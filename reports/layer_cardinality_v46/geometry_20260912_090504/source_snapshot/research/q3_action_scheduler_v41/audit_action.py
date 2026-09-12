from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np
import run_action as runner
from offline_benchmark import OfflineRuleWorld, Source


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit_batch(directory):
    protocol = load(directory / 'protocol.json')
    for relative, digest in protocol['source_hashes'].items():
        for path in (runner.ROOT / relative, directory / 'source_snapshot' / relative):
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Source or snapshot changed: ' + relative)
    scenes = {scene['id']: scene for scene in load(directory / 'scenes_evaluator_only.json')}
    rows = []
    total_actions = 0
    predictions = 0
    max_error = 0.0
    for path in sorted((directory / 'records').glob('*.json')):
        row = load(path)
        trace = load(directory / 'traces' / path.name)
        if not row['success'] or row['cleared_count'] != row['count'] or row['effective_target_radius'] != 2000:
            raise ValueError('Incomplete or wrong-policy run: ' + path.name)
        if hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest() != row['trace_sha256']:
            raise ValueError('Trace digest changed')
        clock = runner.baseline.audit_trace(trace, row['virtual_seconds'])
        total_actions += clock['actions']
        max_error = max(max_error, clock['max_clock_error'])
        predictions += runner.audit_predictions(row['action_events'], trace)
        scene = scenes[row['scene_id']]
        world = OfflineRuleWorld([Source(**source) for source in scene['sources']], scene['seed'], scene['error_mode'])
        for action in trace:
            getattr(world, action['action'])(np.asarray(action['position']), action['channel'])
            if world.trace[-1] != action:
                raise ValueError('Independent physical replay differs')
        if world.remaining:
            raise ValueError('Physical replay left a source')
        for event in row['action_events']:
            if event['active_steps_after'] > 8:
                raise ValueError('Active probe budget exceeded')
            if row['mode'] == 'stepwise_action' and event['predicted_kind'] == 'probe':
                if event['active_steps_after'] - event['active_steps_before'] != 1:
                    raise ValueError('A single active probe was not accounted exactly once')
        rows.append(row)
    if len(rows) != len(scenes) * len(protocol['modes']):
        raise ValueError('Missing registered runs')
    reference = {row['scene_id']: row for row in rows if row['mode'] == 'previous'}
    if any(row['trace_sha256'] != reference[row['scene_id']]['trace_sha256'] for row in rows if row['mode'] == 'identity'):
        raise ValueError('Locked executor is not action-exact')
    summary = load(directory / 'summary.json')
    paired = []
    for mode in protocol['modes']:
        selected = [row for row in rows if row['mode'] == mode]
        mean = statistics.mean(row['seconds_per_source'] for row in selected)
        if not math.isclose(mean, summary[mode]['mean'], abs_tol=1e-9):
            raise ValueError('Case-equal complete-mission mean differs')
        for row in selected:
            stats = row['world_stats']
            paired.append({'profile': row['profile'], 'count': row['count'], 'mode': mode,
                           'seconds_per_source': row['seconds_per_source'],
                           'saved_seconds_per_source': reference[row['scene_id']]['seconds_per_source'] - row['seconds_per_source'],
                           'movement': stats['move_meters'] / (5 * row['count']),
                           'radio': (5 * stats['measure_count'] + stats['switch_count']) / row['count'],
                           'optical': (5 * stats['successful_clear_count'] + 3 * stats['failed_clear_count']) / row['count'],
                           'source_selections': row['policy_stats']['joint_targets'], 'wall_seconds': row['wall_seconds']})
    return {'runs': len(rows), 'successes': len(rows), 'actions_audited': total_actions, 'exact_physics_replays': len(rows),
            'max_clock_error': max_error, 'identity_cases': len(scenes), 'predicted_actions_verified': predictions,
            'source_files': len(protocol['source_hashes']), 'frozen_manifests': runner.frozen_integrity(),
            'summary': summary, 'paired_cases': paired, 'official_calls': 0, 'formal_calls': 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a fresh D-drive audit output')
    result = audit_batch(args.batch.resolve())
    runner.experiment.save(output, result)
    print(json.dumps({key: value for key, value in result.items() if key not in ('summary', 'paired_cases')}, indent=2))


if __name__ == '__main__':
    main()
