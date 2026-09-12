from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import statistics
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'research/joint_search_v5'), str(ROOT / 'research/radial_patrol_v7')]
import experiment_v5 as experiment
from radial_policy import RadialPolicy
from policy_v5 import ContinuousPolicy
from final_validation import audit_trace
from geometry import dual_ring_network
from offline_benchmark import OfflineRuleWorld, Source
import numpy as np

sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'research/interface_validation')]
from b2026_robot import session as session_module
from b2026_robot.client import RobotClient
from b2026_robot.policy import make_policy
from b2026_robot.storage import Journal
from mock_server import Fault, MockArena

SELECTED = {'q3': 'pilot1600', 'q4': 'certified_fixed'}


class EvaluationWorld(OfflineRuleWorld):
    def bearing_error(self, channel):
        if self.error_mode == 'spatial_correlated':
            return float(np.sin(self.position[0] / 450 + self.position[1] / 700 + channel * 0.7))
        return super().bearing_error(channel)


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def hashes():
    folders = (HERE, ROOT / 'research/radial_patrol_v7', ROOT / 'src/b2026_robot', ROOT / 'research/interface_validation')
    return {**experiment.source_hashes(), **{str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                            for folder in folders for path in folder.glob('*.py')}}


def evaluate(scene, mode):
    factory = RadialPolicy if mode == 'pilot1600' else None
    return experiment.evaluate(scene, mode, policy_factory=factory, world_factory=EvaluationWorld)


def stress_scenes():
    scenes = []
    for problem_index, problem in enumerate(('q3', 'q4')):
        for count in (10, 16):
            for repetition in range(4):
                seed = 117100000 + problem_index * 1000000 + count * 100 + repetition
                generator = np.random.default_rng(seed)
                rotation = generator.uniform(0, 2 * np.pi)
                channels = generator.choice(np.arange(1, 21), count, replace=False)
                sources = []
                for index, channel in enumerate(channels):
                    distance = (0.001, 5.000001, 20.000001, 999.999999, 1000.000001, 1799.999999)[index % 6]
                    angle = rotation + index * 2 * np.pi / count
                    position = distance * np.array([np.cos(angle), np.sin(angle)])
                    heading = None
                    if problem == 'q4':
                        angles = np.deg2rad(np.arange(720) / 2)
                        directions = np.column_stack((np.cos(angles), np.sin(angles)))
                        vectors = dual_ring_network() - position
                        visible = (directions @ vectors.T >= 0) & (np.linalg.norm(vectors, axis=1) <= 1000)
                        heading = float(np.argmin(visible.sum(axis=1)) / 2)
                    sources.append({'channel': int(channel), 'position': position.tolist(), 'radius': 1000.0, 'heading_deg': heading})
                error_mode = 'constant_extreme' if repetition < 2 else 'spatial_correlated'
                scenes.append({'id': f'jointv7_stress_{problem}_boundary_noise_{count}_{repetition}', 'problem': problem,
                               'count': count, 'seed': seed, 'profile': 'boundary_noise', 'phase': 'stress',
                               'error_mode': error_mode, 'sources': sources})
    return scenes


def http_factory(client, problem, q4_network='dual21'):
    original, metadata = make_policy(client, problem, q4_network)
    base = RadialPolicy if problem == 'q3' else ContinuousPolicy
    adapted = type('FrozenV7LocalAdapter', (base, type(original)), {})
    policy = adapted(original.port, problem, original.stations, 'E_joint' if problem == 'q3' else 'F_route',
                     'grid7' if problem == 'q3' else 'dual21', mode=SELECTED[problem])
    if adapted.account is not type(original).account:
        raise ValueError('Adapter did not reuse authoritative interface accounting')
    metadata.update({'local_candidate_only': True, 'variant': SELECTED[problem]})
    return policy, metadata


def rounded_clock_audit(folder, direct, expected_raw, expected_rounded):
    events = [json.loads(line) for line in (folder / 'requests.jsonl').read_text(encoding='utf-8').splitlines()]
    requests = {event['payload']['request_id']: event for event in events if event['event'] == 'request'}
    responses = {}
    for event in events:
        if event['event'] == 'response':
            try:
                body = json.loads(event['body_utf8'])
            except (ValueError, KeyError):
                continue
            if body.get('accepted'):
                responses[event['request_id']] = body
    position = np.zeros(2)
    receiver = 1
    rounded = 0.0
    raw = 0.0
    action_index = 0
    for event in (event for event in events if event['event'] == 'committed'):
        if event['path'] in ('/measure', '/clear'):
            request = requests[event['request_id']]['payload']
            response = responses[event['request_id']]
            destination = np.array([request['position']['x'], request['position']['y']])
            action = direct[action_index]
            if (action['action'] != event['path'][1:] or action['channel'] != request['channel']
                    or action['position'] != destination.tolist()):
                raise ValueError('HTTP physical action differs from frozen direct run')
            movement = float(np.linalg.norm(destination - position)) / 5
            raw += movement
            rounded += movement
            if event['path'] == '/measure':
                raw += 5
                rounded += 5
                if receiver != request['channel']:
                    raw += 1
                    rounded += 1
                receiver = request['channel']
            else:
                cost = 5 if response['clear_result'] == 'success' else 3
                raw += cost
                rounded += cost
            rounded = round(rounded, 6)
            position = destination
            action_index += 1
        elif event['path'] not in ('/enter', '/exit'):
            raise ValueError('Unknown committed operation')
        if (rounded != event['virtual_time_s'] or receiver != event['receiver_channel']
                or position.tolist() != event['position']):
            raise ValueError('Per-action authoritative rounded state differs')
    if action_index != len(direct) or abs(raw - expected_raw) > 1e-8 or rounded != expected_rounded:
        raise ValueError('Final HTTP clock or action count differs')
    return {'actions': action_index, 'per_action_error': 0.0, 'unrounded_seconds': raw,
            'rounded_seconds': rounded, 'aggregate_rounding_difference': abs(raw - rounded)}


def http_checks(output, scenes, records):
    selected = [scene for scene in scenes if scene['phase'] == 'holdout' and
                (scene['problem'], scene['profile'], scene['count']) in
                (('q3', 'boundary', 10), ('q3', 'near_origin', 16), ('q4', 'clustered', 16), ('q4', 'boundary', 16))]
    results = []
    for scene in selected:
        mode = SELECTED[scene['problem']]
        direct = next(row for row in records if row['scene_id'] == scene['id'] and row['mode'] == mode)
        if not direct['success']:
            results.append({'scene_id': scene['id'], 'passed': False, 'reason': 'direct evaluation failed'})
            continue
        direct_trace = load(output / 'traces' / (scene['id'] + '__' + mode + '.json'))
        for recovery in ((False, True) if scene['profile'] == 'boundary' else (False,)):
            folder = output / 'http' / (scene['id'] + ('_recovery' if recovery else '_normal'))
            folder.mkdir(parents=True, exist_ok=False)
            world = EvaluationWorld([Source(**source) for source in scene['sources']], scene['seed'], scene['error_mode'])
            faults = [Fault('/enter', 'drop_after'), Fault('/measure', 'drop_after'), Fault('/measure', 'truncate_after'),
                      Fault('/clear', 'drop_after'), Fault('/exit', 'drop_after')] if recovery else []
            with MockArena(world, faults=faults) as arena, Journal(folder / 'requests.jsonl') as journal:
                if urlparse(arena.base_url).hostname not in ('127.0.0.1', 'localhost', '::1'):
                    raise ValueError('Only local loopback HTTP is permitted')
                client = RobotClient(arena.base_url, arena.robot_id, journal, mock_token=arena.token)
                with patch.object(session_module, 'make_policy', http_factory):
                    result = session_module.run_session(client, scene['problem'])
                experiment.save(folder / 'session.json', result)
                clock = rounded_clock_audit(folder, direct_trace, direct['virtual_seconds'], client.virtual_seconds)
                actions = lambda trace: [{key: action[key] for key in ('action', 'channel', 'position', 'response')} for action in trace]
                bodies = {}
                for request in arena.requests:
                    identifier = json.loads(request['body_utf8'])['request_id']
                    bodies.setdefault(identifier, set()).add((request['path'], request['body_utf8']))
                checks = {'normal_exit': result['status'] == 'completed' and client.exited and not result['error'],
                          'all_cleared': not world.remaining and len(client.cleared_channels) == scene['count'],
                          'same_actions': actions(world.trace) == actions(direct_trace),
                          'authority_exact': client.virtual_seconds == world.virtual_seconds,
                          'same_position': client.position == tuple(world.position), 'same_receiver': client.receiver_channel == world.receiver_channel,
                          'single_execution': len(arena.executions) == len(world.trace) + 2,
                          'retry_identity': all(len(values) == 1 for values in bodies.values()), 'faults_consumed': not arena.faults,
                          'retries': client.retry_count == (5 if recovery else 0),
                          'no_uncertainty': not any(result.get(key) for key in ('uncertain', 'pending_request', 'watchdog_errors'))}
                experiment.save(folder / 'checks.json', checks)
                experiment.save(folder / 'clock_audit.json', clock)
                results.append({'scene_id': scene['id'], 'recovery': recovery, 'passed': all(checks.values()), 'checks': checks, 'clock': clock})
            experiment.save(output / 'http_results.json', results)
            print('HTTP', scene['id'], recovery, results[-1]['passed'], flush=True)
    return results


def summarize(records):
    summary = {}
    for problem in SELECTED:
        summary[problem] = {}
        for phase in ('holdout', 'stress', 'combined'):
            rows = [row for row in records if row['problem'] == problem and (phase == 'combined' or row['phase'] == phase)]
            baseline = {row['scene_id']: row for row in rows if row['mode'] == 'previous'}
            selected = [row for row in rows if row['mode'] == SELECTED[problem]]
            result = {'scenes': len(selected), 'all_passed': all(row['success'] for row in rows),
                      'candidate_successes': sum(row['success'] for row in selected), 'mode': SELECTED[problem]}
            if result['all_passed']:
                old_mean = statistics.mean(row['seconds_per_source'] for row in baseline.values())
                new_mean = statistics.mean(row['seconds_per_source'] for row in selected)
                changes = [row['seconds_per_source'] / baseline[row['scene_id']]['seconds_per_source'] - 1 for row in selected]
                result.update({'previous_mean': old_mean, 'candidate_mean': new_mean,
                               'improvement_percent': (1 - new_mean / old_mean) * 100,
                               'faster': sum(change < -1e-10 for change in changes), 'slower': sum(change > 1e-10 for change in changes),
                               'tied': sum(abs(change) <= 1e-10 for change in changes), 'worst_regression_percent': max(changes) * 100,
                               'target_met_on_this_phase': new_mean < (180 if problem == 'q3' else 300)})
            summary[problem][phase] = result
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a new D-drive output')
    frozen = hashes()
    experiment.save(output / 'selection.json', {'source_hashes': frozen, 'selected': SELECTED, 'created_local': datetime.now().isoformat(),
                                               'official_calls': 0, 'holdout_seed': 113100000, 'stress_seed': 117100000,
                                               'no_post_holdout_tuning': True, 'objective': 'Q3<180 and Q4<300 case-equal virtual seconds/source'})
    for relative in frozen:
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    shutil.copy2(HERE / 'PROTOCOL.md', output / 'PROTOCOL.md')
    scenes = experiment.make_scenes(113100000, 'holdout', (10, 12, 14, 16)) + stress_scenes()
    for scene in scenes:
        scene['id'] = scene['id'].replace('jointv4_', 'jointv7_')
    folders = ('offline_review', 'plan_trials', 'plan_trials_v2', 'policy_optimization_v3', 'q4_validation_v1',
               'q4_optimization_v2', 'joint_search_v4', 'joint_search_v5', 'flexible_patrol_v6', 'radial_patrol_v7')
    prior_paths = [path for folder in folders for path in (ROOT / 'reports' / folder).rglob('*scenes*.json')]
    prior_seeds = {scene['seed'] for path in prior_paths for scene in load(path)}
    seeds = [scene['seed'] for scene in scenes]
    if len(set(seeds)) != len(seeds) or set(seeds) & prior_seeds:
        raise ValueError('Validation seeds overlap existing development or holdout scenes')
    experiment.save(output / 'seed_audit.json', {'unique': len(seeds), 'prior_count': len(prior_seeds), 'overlap': [],
                                                'prior_files': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in prior_paths}})
    experiment.save(output / 'scenes_evaluator_only.json', scenes)
    records = []
    audits = []
    for scene in scenes:
        for mode in ('previous', SELECTED[scene['problem']]):
            row, trace = evaluate(scene, mode)
            records.append(row)
            audits.append(audit_trace(trace, row['virtual_seconds']))
            name = scene['id'] + '__' + mode + '.json'
            experiment.save(output / 'records' / name, row)
            experiment.save(output / 'traces' / name, trace)
            print(scene['problem'], scene['phase'], scene['profile'], scene['count'], mode,
                  row['failure'] if not row['success'] else round(row['seconds_per_source'], 3), flush=True)
    summary = summarize(records)
    experiment.save(output / 'summary.json', summary)
    replays = []
    for problem in SELECTED:
        for profile, count in (('random', 10), ('near_origin', 16)):
            scene = next(scene for scene in scenes if scene['problem'] == problem and scene['profile'] == profile and scene['count'] == count)
            row = next(row for row in records if row['scene_id'] == scene['id'] and row['mode'] == SELECTED[problem])
            replay, trace = evaluate(scene, SELECTED[problem])
            replays.append({'scene_id': scene['id'], 'passed': replay['success'] and replay['trace_sha256'] == row['trace_sha256']})
    experiment.save(output / 'replays.json', replays)
    http = http_checks(output, scenes, records)
    experiment.save(output / 'verification.json', {'source_hashes_unchanged': frozen == hashes(), 'runs': len(records),
                                                   'all_cleared_runs': sum(row['success'] for row in records),
                                                   'actions_audited': sum(audit['actions'] for audit in audits),
                                                   'max_clock_error': max(audit['max_clock_error'] for audit in audits),
                                                   'replays_passed': sum(row['passed'] for row in replays),
                                                   'http_passed': sum(row['passed'] for row in http), 'http_runs': len(http),
                                                   'official_calls': 0, 'dual_target_met': all(summary[problem]['combined'].get('target_met_on_this_phase', False) for problem in SELECTED)})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
