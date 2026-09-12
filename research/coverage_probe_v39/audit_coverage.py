from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np
import run_coverage as runner
from coverage_certificate import TriangleCoverage
from offline_benchmark import OfflineRuleWorld, Source


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit_commitment(event, trace):
    anchor = np.array(event['anchor'])
    relative = np.array(event['vertices']) - anchor
    displacement = np.array(event['candidates']) - anchor
    direction = np.array(event['direction'])
    squared_changes = np.sum(displacement**2, axis=1)[None, :] - 2 * relative @ displacement.T
    if np.max(squared_changes) >= 0:
        raise ValueError('The new probe pair does not contract range over its full region')
    if np.min(displacement @ direction) <= 0 or np.min(relative @ direction) <= np.max(displacement @ direction):
        raise ValueError('The new probe segment is not between the anchor and source region')
    cross_first = relative[:, 0] * displacement[0, 1] - relative[:, 1] * displacement[0, 0]
    cross_second = relative[:, 0] * displacement[1, 1] - relative[:, 1] * displacement[1, 0]
    if np.any(event['side'] * cross_first <= 0) or np.any(event['side'] * cross_second >= 0):
        raise ValueError('The probe pair does not bracket every source ray')
    remaining = event['remaining_before']
    if event['kept'] != [index for index in remaining if index != event['station']]:
        raise ValueError('Bad patrol partition')
    prefix = [action for action in trace if action['virtual_seconds'] <= event['start_seconds']]
    negatives = {(tuple(action['position']), action['channel']) for action in prefix
                 if action['action'] == 'measure' and action['response']['result'] == 'no_signal'}
    optical = {(tuple(action['position']), action['channel']) for action in prefix
               if action['action'] == 'clear' and action['response']['result'] == 'no_target_in_range'}
    for point in event['before_radio']:
        if any((tuple(point), channel) not in negatives for channel in event['unknown_before']):
            raise ValueError('Earlier radio coverage uses a future or nonexistent absence')
    for point in event['before_optical']:
        if any((tuple(point), channel) not in optical for channel in event['unknown_before']):
            raise ValueError('Earlier optical coverage uses a future or nonexistent absence')
    trial = TriangleCoverage()
    for point in event['before_radio']:
        trial.observe_absence(point)
    for point in event['before_optical']:
        trial.observe_clear_absence(point)
    trial.observe_absence(event['point'])
    for index in event['kept']:
        trial.observe_absence(event['route'][index])
    if not trial.empty:
        raise ValueError('Continuous coverage is incomplete')
    if event['remaining_after_scan'] != remaining:
        raise ValueError('Station deleted before the scan completed')
    if event['outcome'] == 'deleted_after_real_scan':
        actions = [action for action in trace
                   if event['scan_start_seconds'] < action['virtual_seconds'] <= event['scan_end_seconds']]
        if any(action['action'] != 'measure' or action['position'] != event['point'] for action in actions):
            raise ValueError('Invalid scan interval')
        if sorted(action['channel'] for action in actions) != sorted(event['unknown_before']):
            raise ValueError('Incomplete actual unknown-channel scan')
        if event['remaining_after'] != event['kept']:
            raise ValueError('Deleted set differs from the certificate')
    elif event['remaining_after'] != remaining:
        raise ValueError('Incomplete commitment removed fallback station')
    return {'maximum_range_change': float(np.max(squared_changes)), 'first_failed': event['first_failed']}


def audit_batch(directory):
    protocol = load(directory / 'protocol.json')
    for relative, expected in protocol['source_hashes'].items():
        for path in (runner.ROOT / relative, directory / 'source_snapshot' / relative):
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('Source or snapshot changed: ' + relative)
    scenes = {scene['id']: scene for scene in load(directory / 'scenes_evaluator_only.json')}
    rows = []
    action_count = 0
    commitments = 0
    maximum_error = 0.0
    for path in sorted((directory / 'records').glob('*.json')):
        row = load(path)
        trace = load(directory / 'traces' / path.name)
        if not row['success'] or row['cleared_count'] != row['count']:
            raise ValueError('Failed complete mission: ' + path.name)
        if hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest() != row['trace_sha256']:
            raise ValueError('Trace changed')
        clock = runner.baseline.run_scan_benchmark.audit_trace(trace, row['virtual_seconds'])
        maximum_error = max(maximum_error, clock['max_clock_error'])
        action_count += clock['actions']
        scene = scenes[row['scene_id']]
        world = OfflineRuleWorld([Source(**source) for source in scene['sources']], scene['seed'], scene['error_mode'])
        for action in trace:
            getattr(world, action['action'])(np.asarray(action['position']), action['channel'])
            if world.trace[-1] != action:
                raise ValueError('Independent physical replay differs')
        if world.remaining:
            raise ValueError('Not all sources cleared')
        for event in row['coverage_commitments']:
            audit_commitment(event, trace)
            commitments += 1
        rows.append(row)
    if len(rows) != len(scenes) * len(protocol['modes']):
        raise ValueError('Missing registered runs')
    reference = {row['scene_id']: row for row in rows if row['mode'] == 'previous'}
    for row in rows:
        if row['mode'] == 'identity' and row['trace_sha256'] != reference[row['scene_id']]['trace_sha256']:
            raise ValueError('Identity adapter changed the mission')
    summary = load(directory / 'summary.json')
    paired = []
    for mode in protocol['modes']:
        selected = [row for row in rows if row['mode'] == mode]
        mean = statistics.mean(row['seconds_per_source'] for row in selected)
        if not math.isclose(mean, summary[mode]['mean'], abs_tol=1e-9):
            raise ValueError('Mean differs from complete traces')
        for row in selected:
            stats = row['world_stats']
            paired.append({'profile': row['profile'], 'count': row['count'], 'mode': mode,
                           'seconds_per_source': row['seconds_per_source'],
                           'saved_seconds_per_source': reference[row['scene_id']]['seconds_per_source'] - row['seconds_per_source'],
                           'movement': stats['move_meters'] / (5 * row['count']),
                           'radio': (5 * stats['measure_count'] + stats['switch_count']) / row['count'],
                           'optical': (5 * stats['successful_clear_count'] + 3 * stats['failed_clear_count']) / row['count'],
                           'replacements': row['policy_stats'].get('co_probe_replacements', 0),
                           'wall_seconds': row['wall_seconds']})
    return {'runs': len(rows), 'successes': len(rows), 'exact_physics_replays': len(rows), 'actions_audited': action_count,
            'max_clock_error': maximum_error, 'identity_cases': len(scenes), 'commitments_audited': commitments,
            'source_files': len(protocol['source_hashes']), 'frozen_manifests': runner.baseline.frozen_integrity(),
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
