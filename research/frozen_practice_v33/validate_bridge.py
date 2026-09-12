from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import adapter
from adapter import HERE, ROOT, SELECTED

sys.path.insert(0, str(ROOT / 'research/shaped_probe_validation_v11'))
from validate_shaped import frozen_verifier
from b2026_robot.client import RobotClient
from b2026_robot.storage import Journal
from mock_server import Fault, MockArena


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists() or not output.is_relative_to(ROOT / 'reports/frozen_practice_v33'):
        raise ValueError('Use a fresh D-drive bridge validation directory')
    frozen = adapter.hashes()
    save = frozen_verifier.experiment.save
    save(output / 'protocol.json', {'created_local': datetime.now().isoformat(), 'source_hashes': frozen, 'official_calls': 0})
    for relative in frozen:
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    chosen = {('q3', 'boundary', 10), ('q3', 'near_origin', 16), ('q4', 'clustered', 16), ('q4', 'boundary', 16)}
    scenes = [scene for scene in frozen_verifier.experiment.make_scenes(361420100, 'adapter', [10, 16]) if (scene['problem'], scene['profile'], scene['count']) in chosen]
    save(output / 'scenes_evaluator_only.json', scenes)
    results = []
    for scene in scenes:
        def factory(port, problem, stations, variant, network, mode):
            selected_class = adapter.ScanEconomyPolicy if problem == 'q3' else adapter.ShapedProbePolicy
            return selected_class(port, problem, stations, variant, network, mode=SELECTED[problem])
        direct, trace = frozen_verifier.experiment.evaluate(scene, SELECTED[scene['problem']], policy_factory=factory, world_factory=frozen_verifier.EvaluationWorld)
        save(output / 'records' / (scene['id'] + '.json'), direct)
        save(output / 'traces' / (scene['id'] + '.json'), trace)
        if not direct['success']:
            raise ValueError('Frozen direct reference failed')
        for recovery in (False, True):
            folder = output / 'http' / (scene['id'] + ('_recovery' if recovery else '_normal'))
            folder.mkdir(parents=True)
            world = frozen_verifier.EvaluationWorld([frozen_verifier.Source(**source) for source in scene['sources']], scene['seed'], scene['error_mode'])
            faults = [Fault('/enter', 'drop_after'), Fault('/measure', 'drop_after'), Fault('/measure', 'truncate_after'), Fault('/clear', 'drop_after'), Fault('/exit', 'drop_after')] if recovery else []
            with MockArena(world, faults=faults) as arena, Journal(folder / 'requests.jsonl') as journal:
                if urlparse(arena.base_url).hostname != '127.0.0.1':
                    raise ValueError('Only the local synthetic interface is permitted')
                client = RobotClient(arena.base_url, arena.robot_id, journal, mock_token=arena.token)
                result = adapter.run_session(client, scene['problem'])
                save(folder / 'session.json', result)
                clock = frozen_verifier.rounded_clock_audit(folder, trace, direct['virtual_seconds'], client.virtual_seconds)
                actions = lambda sequence: [{key: action[key] for key in ('action', 'channel', 'position', 'response')} for action in sequence]
                bodies = {}
                for request in arena.requests:
                    identifier = json.loads(request['body_utf8'])['request_id']
                    bodies.setdefault(identifier, set()).add((request['path'], request['body_utf8']))
                checks = {'normal_exit': result['status'] == 'completed' and client.exited and not result['error'],
                          'all_cleared': not world.remaining and len(client.cleared_channels) == scene['count'],
                          'same_actions': actions(world.trace) == actions(trace), 'authority_exact': client.virtual_seconds == world.virtual_seconds,
                          'single_execution': len(arena.executions) == len(world.trace) + 2,
                          'retry_identity': all(len(values) == 1 for values in bodies.values()), 'faults_consumed': not arena.faults,
                          'expected_retry_count': client.retry_count == (5 if recovery else 0),
                          'no_uncertainty': not any(result.get(key) for key in ('uncertain', 'pending_request', 'watchdog_errors'))}
                save(folder / 'checks.json', checks)
                results.append({'scene_id': scene['id'], 'recovery': recovery, 'passed': all(checks.values()), 'checks': checks, 'clock': clock})
                print('HTTP', scene['problem'], scene['profile'], recovery, all(checks.values()), flush=True)
    verification = {'passed': len(results) == 8 and all(result['passed'] for result in results) and frozen == adapter.hashes(),
                    'official_calls': 0, 'http_results': results, 'source_unchanged': frozen == adapter.hashes()}
    save(output / 'verification.json', verification)
    if not verification['passed']:
        raise ValueError('Bridge validation failed')
    (output.parent / 'LATEST_VALIDATION.txt').write_text(str(output), encoding='utf-8')
    print('VALIDATION', output, flush=True)


if __name__ == '__main__':
    main()
