from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research' / 'policy_optimization_v3'))

from candidate import CandidatePolicy
from experiment import (
    Q4OptimizationPolicy,
    action_hash,
    file_hash,
    load,
    record_path,
    save,
    verify,
)
from mock_server import Fault, MockArena
from offline_benchmark import OfflineRuleWorld, Source

from b2026_robot import session as session_module
from b2026_robot.client import RobotClient
from b2026_robot.policy import make_policy
from b2026_robot.storage import Journal


def policy_factory(mode):
    def factory(client, problem, q4_network='dual21'):
        original, metadata = make_policy(client, problem, q4_network)
        if problem != 'q4' or q4_network != 'dual21':
            raise ValueError('Local selected-policy adapter only supports q4/dual21')
        if mode == 'baseline':
            return original, metadata
        base = CandidatePolicy if mode == 'joint' else Q4OptimizationPolicy
        policy_class = type('SelectedQ4LocalAdapter', (base, type(original)), {})
        policy = policy_class(original.port, 'q4', original.stations, 'F_route', 'dual21', mode=mode)
        if policy_class.account is not type(original).account:
            raise ValueError('Selected policy does not reuse authoritative interface accounting')
        metadata.update({'variant': f'F_route+q4_v2_{mode}', 'local_candidate_only': True})
        return policy, metadata
    return factory


def run(output):
    _, scenes = verify(output)
    mode = load(output / 'selection.json')['selected']
    selected_ids = ('q4v2_holdout_q4_random_10_0', 'q4v2_holdout_q4_boundary_16_0',
                    'q4v2_holdout_q4_clustered_16_1', 'q4v2_stress_heading_sweep_16_3')
    recovery = {'q4v2_holdout_q4_boundary_16_0', 'q4v2_stress_heading_sweep_16_3'}
    results = []
    for scene_id in selected_ids:
        scene = next(scene for scene in scenes if scene['id'] == scene_id)
        direct = load(record_path(output, scene, mode))
        if not direct['all_cleared']:
            raise ValueError('Selected direct policy failed; do not conceal failure through HTTP retesting')
        for run_mode in (('normal', 'recovery') if scene_id in recovery else ('normal',)):
            trial = f'{scene_id}_{run_mode}'
            folder = output / 'http' / trial
            folder.mkdir(parents=True, exist_ok=False)
            world = OfflineRuleWorld([Source(**source) for source in scene['sources']], scene['seed'], scene['error_mode'])
            faults = [] if run_mode == 'normal' else [Fault('/enter', 'drop_after'), Fault('/measure', 'drop_after'),
                Fault('/measure', 'truncate_after'), Fault('/clear', 'drop_after'), Fault('/exit', 'drop_after')]
            with MockArena(world, faults=faults) as arena, Journal(folder / 'requests.jsonl') as journal:
                client = RobotClient(arena.base_url, arena.robot_id, journal, mock_token=arena.token)
                with patch.object(session_module, 'make_policy', policy_factory(mode)):
                    summary = session_module.run_session(client, 'q4')
                save(folder / 'summary.json', summary)
                request_bodies = {}
                for request in arena.requests:
                    identifier = json.loads(request['body_utf8'])['request_id']
                    request_bodies.setdefault(identifier, set()).add((request['path'], request['body_utf8']))
                checks = {
                    'normal_exit': summary['status'] == 'completed' and client.exited and not summary['error'],
                    'all_cleared': not world.remaining and len(client.cleared_channels) == scene['count'],
                    'same_actions': action_hash(world.trace) == direct['action_sha256'],
                    'same_policy_stats': summary.get('policy_stats') == direct['policy_stats'],
                    'same_clock': abs(client.virtual_seconds - direct['virtual_seconds']) <= (len(world.trace) + 1) * 1e-6,
                    'authoritative_clock_exact': client.virtual_seconds == world.virtual_seconds,
                    'position_exact': client.position == tuple(world.position),
                    'receiver_exact': client.receiver_channel == world.receiver_channel,
                    'one_execution_per_action': len(arena.executions) == len(world.trace) + 2,
                    'same_retry_id_and_body': all(len(bodies) == 1 for bodies in request_bodies.values()),
                    'all_faults_exercised': not arena.faults,
                    'expected_retries': client.retry_count == (5 if run_mode == 'recovery' else 0),
                    'no_uncertainty': not any(summary.get(key) for key in ('uncertain', 'pending_request', 'watchdog_errors')),
                }
                save(folder / 'checks.json', checks)
                results.append({'trial': trial, 'scene_id': scene_id, 'mode': mode, 'checks': checks,
                                'passed': all(checks.values()), 'cleared': len(client.cleared_channels),
                                'source_count': scene['count'], 'virtual_s': client.virtual_seconds,
                                'physical_actions': len(world.trace), 'retry_count': client.retry_count})
            results[-1]['request_log_sha256'] = file_hash(folder / 'requests.jsonl')
            save(output / 'http_results.json', results)
            print(f'Local HTTP {trial}: passed={results[-1]["passed"]}', flush=True)
    verify(output)
    if len(results) != 6 or not all(row['passed'] for row in results):
        raise AssertionError('Selected candidate local HTTP failed; no official deployment')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Selected q4 policy local MockArena tests only, never official')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    folder = args.output.resolve()
    if folder.drive.upper() != 'D:' or not folder.is_relative_to(ROOT / 'reports' / 'q4_optimization_v2'):
        raise ValueError('Output must remain within the dedicated D-drive reports')
    run(folder)
