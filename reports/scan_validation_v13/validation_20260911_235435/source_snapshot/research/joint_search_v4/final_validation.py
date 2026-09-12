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

from experiment import HERE, ROOT, evaluate, hashes, make_scenes, save

sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'research' / 'interface_validation')]

import numpy as np
from belief_policy import SweepPreviousPolicy
from b2026_robot import session as session_module
from b2026_robot.client import RobotClient
from b2026_robot.policy import make_policy
from b2026_robot.storage import Journal
from geometry import dual_ring_network
from mock_server import Fault, MockArena
from offline_benchmark import OfflineRuleWorld, Source
from policy import JointSearchPolicy


SELECTED = {'q3': 'adaptive', 'q4': 'sweep_previous'}


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def stress_scenes():
    scenes = []
    for problem_index, problem in enumerate(('q3', 'q4')):
        for count in (10, 16):
            for repetition in range(4):
                seed = 98100000 + problem_index * 1000000 + count * 100 + repetition
                generator = np.random.default_rng(seed)
                rotation = generator.uniform(0, 360)
                channels = generator.choice(np.arange(1, 21), count, replace=False)
                sources = []
                for index, channel in enumerate(channels):
                    radius = (0.001, 5.000001, 999.999999, 1000.000001, 1799.999999, 1790.0)[index % 6]
                    angle = math.radians(rotation + index * 360 / count)
                    position = np.array([radius * math.cos(angle), radius * math.sin(angle)])
                    heading = None
                    if problem == 'q4' and index:
                        angles = np.deg2rad(np.arange(720) / 2)
                        directions = np.column_stack((np.cos(angles), np.sin(angles)))
                        vectors = dual_ring_network() - position
                        visible = (directions @ vectors.T >= 0) & (np.linalg.norm(vectors, axis=1) <= 1000)
                        heading = float(np.argmin(visible.sum(axis=1)) / 2)
                    sources.append({'channel': int(channel), 'position': position.tolist(), 'radius': 1000.0,
                                    'heading_deg': heading})
                scenes.append({'id': f'jointv4_stress_{problem}_heading_sweep_{count}_{repetition}',
                               'problem': problem, 'count': count, 'seed': seed, 'profile': 'heading_sweep',
                               'phase': 'stress', 'error_mode': 'spatial_extreme', 'sources': sources})
    return scenes


def audit_trace(trace, expected):
    position = np.zeros(2)
    receiver = 1
    elapsed = 0.0
    max_error = 0.0
    for action in trace:
        destination = np.array(action['position'])
        elapsed += float(np.linalg.norm(destination - position)) / 5
        if action['action'] == 'measure':
            elapsed += 5 + int(action['channel'] != receiver)
            receiver = action['channel']
        elif action['action'] == 'clear':
            elapsed += 5 if action['response']['result'] == 'success' else 3
        else:
            raise ValueError('Unknown physical action')
        max_error = max(max_error, abs(elapsed - action['virtual_seconds']))
        position = destination
    if abs(elapsed - expected) > 1e-6 or max_error > 1e-6:
        raise ValueError('Independent trace timing failed')
    return {'actions': len(trace), 'max_clock_error': max_error}


def factory(client, problem, q4_network='dual21'):
    original, metadata = make_policy(client, problem, q4_network)
    base = JointSearchPolicy if problem == 'q3' else SweepPreviousPolicy
    adapted = type('JointV4LocalAdapter', (base, type(original)), {})
    arguments = (original.port, problem, original.stations,
                 'E_joint' if problem == 'q3' else 'F_route', 'grid7' if problem == 'q3' else 'dual21')
    policy = adapted(*arguments, mode='adaptive') if problem == 'q3' else adapted(*arguments)
    if adapted.account is not type(original).account:
        raise ValueError('Interface authoritative accounting not reused')
    metadata.update({'local_candidate_only': True, 'variant': SELECTED[problem]})
    return policy, metadata


def http_checks(output, scenes, records):
    selected = [scene for scene in scenes if scene['phase'] == 'holdout'
                and (scene['problem'], scene['profile'], scene['count']) in
                (('q3', 'random', 10), ('q3', 'near_origin', 16), ('q4', 'clustered', 16), ('q4', 'boundary', 16))]
    results = []
    for scene in selected:
        direct = next(record for record in records if record['scene_id'] == scene['id'] and record['mode'] == SELECTED[scene['problem']])
        if not direct['success']:
            results.append({'scene_id': scene['id'], 'passed': False, 'skipped': 'offline failure'})
            continue
        for recovery in (False, True) if scene['profile'] in ('random', 'boundary') else (False,):
            folder = output / 'http' / (scene['id'] + ('_recovery' if recovery else '_normal'))
            folder.mkdir(parents=True, exist_ok=False)
            world = OfflineRuleWorld([Source(**source) for source in scene['sources']], scene['seed'], scene['error_mode'])
            faults = [Fault('/enter', 'drop_after'), Fault('/measure', 'drop_after'), Fault('/measure', 'truncate_after'),
                      Fault('/clear', 'drop_after'), Fault('/exit', 'drop_after')] if recovery else []
            with MockArena(world, faults=faults) as arena, Journal(folder / 'requests.jsonl') as journal:
                client = RobotClient(arena.base_url, arena.robot_id, journal, mock_token=arena.token)
                with patch.object(session_module, 'make_policy', factory):
                    result = session_module.run_session(client, scene['problem'])
                save(folder / 'session.json', result)
                direct_trace = load(output / 'traces' / (scene['id'] + '__' + SELECTED[scene['problem']] + '.json'))
                actions = lambda trace: [{key: action[key] for key in ('action', 'channel', 'position', 'response')} for action in trace]
                bodies = {}
                for request in arena.requests:
                    identifier = json.loads(request['body_utf8'])['request_id']
                    bodies.setdefault(identifier, set()).add((request['path'], request['body_utf8']))
                checks = {'normal_exit': result['status'] == 'completed' and client.exited and not result['error'],
                          'all_cleared': not world.remaining and len(client.cleared_channels) == scene['count'],
                          'same_actions': actions(world.trace) == actions(direct_trace),
                          'same_clock': abs(client.virtual_seconds - direct['virtual_seconds']) < 1e-6,
                          'authority_exact': client.virtual_seconds == world.virtual_seconds,
                          'same_position': client.position == tuple(world.position),
                          'same_receiver': client.receiver_channel == world.receiver_channel,
                          'single_execution': len(arena.executions) == len(world.trace) + 2,
                          'retry_identity': all(len(values) == 1 for values in bodies.values()),
                          'faults_consumed': not arena.faults,
                          'retries': client.retry_count == (5 if recovery else 0),
                          'no_uncertainty': not any(result.get(key) for key in ('uncertain', 'pending_request', 'watchdog_errors'))}
                save(folder / 'checks.json', checks)
                results.append({'scene_id': scene['id'], 'recovery': recovery, 'passed': all(checks.values()),
                                'checks': checks, 'actions': len(world.trace), 'retries': client.retry_count})
            save(output / 'http_results.json', results)
            print('HTTP', scene['id'], recovery, results[-1]['passed'], flush=True)
    return results


def summarize(records):
    result = {}
    for problem in SELECTED:
        result[problem] = {}
        for phase in ('holdout', 'stress', 'combined'):
            rows = [record for record in records if record['problem'] == problem and (phase == 'combined' or record['phase'] == phase)]
            previous = {record['scene_id']: record for record in rows if record['mode'] == 'previous'}
            candidate = [record for record in rows if record['mode'] == SELECTED[problem]]
            all_passed = all(record['success'] for record in rows)
            summary = {'scenes': len(candidate), 'candidate_successes': sum(record['success'] for record in candidate),
                       'all_passed': all_passed, 'candidate': SELECTED[problem]}
            if all_passed:
                original_mean = statistics.mean(record['seconds_per_source'] for record in previous.values())
                candidate_mean = statistics.mean(record['seconds_per_source'] for record in candidate)
                deltas = [record['seconds_per_source'] - previous[record['scene_id']]['seconds_per_source'] for record in candidate]
                relative = [record['seconds_per_source'] / previous[record['scene_id']]['seconds_per_source'] - 1 for record in candidate]
                summary.update({'previous_mean': original_mean, 'candidate_mean': candidate_mean,
                                'improvement_percent': 100 * (1 - candidate_mean / original_mean),
                                'faster': sum(delta < -1e-6 for delta in deltas), 'slower': sum(delta > 1e-6 for delta in deltas),
                                'worst_slowdown_percent': max(relative) * 100,
                                'median_saved_seconds_per_source': -statistics.median(deltas),
                                'target': 180 if problem == 'q3' else 300,
                                'target_met': candidate_mean < (180 if problem == 'q3' else 300),
                                'profiles': {profile: {
                                    'previous': statistics.mean(previous[record['scene_id']]['seconds_per_source'] for record in candidate if record['profile'] == profile),
                                    'candidate': statistics.mean(record['seconds_per_source'] for record in candidate if record['profile'] == profile)}
                                    for profile in sorted({record['profile'] for record in candidate})}})
            result[problem][phase] = summary
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a new D-drive validation directory')
    frozen = hashes()
    save(output / 'selection.json', {'created_local': datetime.now().isoformat(), 'selected': SELECTED,
                                  'source_hashes': frozen, 'official_calls': 0,
                                  'selection_scope': 'Exploratory development selection; frozen before new evaluation cases.'})
    save(output / 'protocol.json', {'source_hashes': frozen, 'selected': SELECTED,
                                   'holdout_base_seed': 93100000, 'stress_base_seed': 98100000,
                                   'counts': [10, 12, 14, 16], 'scenes_per_problem': 32,
                                   'holdout_scenes_per_problem': 24, 'stress_scenes_per_problem': 8,
                                   'metric': 'Case-equal total virtual seconds/source, never success-only averaging.',
                                   'no_post_holdout_tuning': True, 'official_calls': 0})
    for path in HERE.glob('*'):
        if path.is_file():
            destination = output / 'source_snapshot' / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    scenes = make_scenes(93100000, 'holdout', (10, 12, 14, 16)) + stress_scenes()
    seeds = [scene['seed'] for scene in scenes]
    prior_paths = []
    for folder in ('offline_review', 'plan_trials', 'plan_trials_v2', 'policy_optimization_v3',
                   'q4_validation_v1', 'q4_optimization_v2', 'joint_search_v4'):
        prior_paths.extend(path for path in (ROOT / 'reports' / folder).rglob('*scenes*.json')
                           if not path.is_relative_to(output))
    prior_seeds = {scene['seed'] for path in prior_paths for scene in load(path)}
    if len(set(seeds)) != len(seeds) or set(seeds) & prior_seeds:
        raise ValueError('Evaluation seeds overlap earlier local scenes')
    save(output / 'seed_audit.json', {'unique': len(seeds), 'prior_seeds': len(prior_seeds), 'overlap': [],
                                   'prior_files': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                                   for path in prior_paths}})
    save(output / 'scenes_evaluator_only.json', scenes)
    records = []
    audit = []
    for scene in scenes:
        for mode in ('previous', SELECTED[scene['problem']]):
            record, trace = evaluate(scene, mode)
            records.append(record)
            audit.append(audit_trace(trace, record['virtual_seconds']))
            filename = scene['id'] + '__' + mode + '.json'
            save(output / 'records' / filename, record)
            save(output / 'traces' / filename, trace)
            print(scene['problem'], scene['phase'], scene['profile'], scene['count'], mode,
                  round(record['seconds_per_source'], 3) if record['success'] else record['failure'], flush=True)
    save(output / 'summary.json', summarize(records))
    replays = []
    for problem in SELECTED:
        for profile, count in (('random', 10), ('near_origin', 16)):
            scene = next(scene for scene in scenes if scene['problem'] == problem and scene['profile'] == profile and scene['count'] == count)
            direct = next(record for record in records if record['scene_id'] == scene['id'] and record['mode'] == SELECTED[problem])
            replay, _trace = evaluate(scene, SELECTED[problem])
            replays.append({'scene_id': scene['id'], 'passed': replay['success'] and replay['trace_sha256'] == direct['trace_sha256']})
    save(output / 'replays.json', replays)
    http_results = http_checks(output, scenes, records)
    save(output / 'verification.json', {'source_hashes_unchanged': hashes() == frozen,
                                      'runs': len(records), 'all_cleared_runs': sum(record['success'] for record in records),
                                      'physical_actions_audited': sum(item['actions'] for item in audit),
                                      'max_clock_error': max(item['max_clock_error'] for item in audit),
                                      'replays_passed': sum(item['passed'] for item in replays),
                                      'http_passed': sum(item['passed'] for item in http_results), 'official_calls': 0})
    print(json.dumps(summarize(records), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
