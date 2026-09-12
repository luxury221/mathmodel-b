from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np
import run_paired as runner
from coverage_certificate import TriangleCoverage
from offline_benchmark import OfflineRuleWorld, Source


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit_commitment(event, trace):
    trial = TriangleCoverage()
    for point in event['before_radio']:
        trial.observe_absence(point)
    for point in event['before_optical']:
        trial.observe_clear_absence(point)
    removed = set(event['removed'])
    if event['kept'] != [index for index in event['remaining_before'] if index not in removed]:
        raise ValueError('Invalid recorded patrol partition')
    for singleton in event['planned_points']:
        partial = copy.deepcopy(trial)
        partial.observe_absence(singleton)
        for index in event['kept']:
            partial.observe_absence(event['route'][index])
        if partial.empty:
            raise ValueError('Claimed joint mechanism was already a single-point replacement')
    for point in event['planned_points']:
        trial.observe_absence(point)
    for index in event['kept']:
        trial.observe_absence(event['route'][index])
    if not trial.empty:
        raise ValueError('Recorded commitment has a continuous hole')
    for checkpoint in event['checkpoints']:
        if checkpoint['remaining'] != event['remaining_before']:
            raise ValueError('Fallback removed before commitment finished')
        actions = [action for action in trace
                   if checkpoint['scan_start_seconds'] < action['virtual_seconds'] <= checkpoint['scan_end_seconds']]
        if checkpoint['actual_scan_recorded']:
            channels = [action['channel'] for action in actions
                        if action['action'] == 'measure' and action['position'] == checkpoint['point']]
            if sorted(channels) != sorted(checkpoint['unknown_before']):
                raise ValueError('A committed scan lacks its actual unknown-channel measurements')
        if any(action['action'] != 'measure' or action['position'] != checkpoint['point'] for action in actions):
            raise ValueError('Unexpected physical action inside a scan')
    if event['outcome'] == 'deleted_after_real_scans':
        if len(event['checkpoints']) != 2 or not all(check['actual_scan_recorded'] for check in event['checkpoints']):
            raise ValueError('Deletion depended on a future scan')
        if event['remaining_after'] != event['kept']:
            raise ValueError('Final patrol does not match the certified kept set')
    elif event['remaining_after'] != event['remaining_before']:
        raise ValueError('An incomplete pair deleted fallback stations')


def audit_batch(directory):
    frozen = load(directory / 'protocol.json')['source_hashes']
    for relative, expected in frozen.items():
        for path in (runner.ROOT / relative, directory / 'source_snapshot' / relative):
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('Source or snapshot hash changed: ' + relative)
    scenes = {scene['id']: scene for scene in load(directory / 'scenes_evaluator_only.json')}
    rows = []
    action_count = 0
    events_audited = 0
    max_clock_error = 0.0
    for path in sorted((directory / 'records').glob('*.json')):
        row = load(path)
        trace = load(directory / 'traces' / path.name)
        if not row['success'] or row['cleared_count'] != row['count']:
            raise ValueError('Failed complete mission retained; batch cannot pass')
        if hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest() != row['trace_sha256']:
            raise ValueError('Trace hash differs')
        clock = runner.run_scan_benchmark.audit_trace(trace, row['virtual_seconds'])
        max_clock_error = max(max_clock_error, clock['max_clock_error'])
        action_count += clock['actions']
        scene = scenes[row['scene_id']]
        world = OfflineRuleWorld([Source(**source) for source in scene['sources']], scene['seed'], scene['error_mode'])
        for action in trace:
            getattr(world, action['action'])(np.asarray(action['position']), action['channel'])
            if world.trace[-1] != action:
                raise ValueError('Independent legal-action replay disagrees')
        if world.remaining:
            raise ValueError('Source remains after complete trace')
        for event in row['commitment_events']:
            audit_commitment(event, trace)
            events_audited += 1
        rows.append(row)
    summary = load(directory / 'summary.json')
    reference = {row['scene_id']: row for row in rows if row['mode'] == 'previous'}
    if len(rows) != 3 * len(scenes):
        raise ValueError('Missing registered runs')
    pairs = []
    for row in rows:
        if row['mode'] == 'identity' and row['trace_sha256'] != reference[row['scene_id']]['trace_sha256']:
            raise ValueError('Identity adapter changed actions')
        if row['mode'] == 'paired':
            previous = reference[row['scene_id']]
            count = row['count']
            pair = {'profile': row['profile'], 'count': count, 'previous': previous['seconds_per_source'],
                    'paired': row['seconds_per_source'], 'trace_identical': row['trace_sha256'] == previous['trace_sha256'],
                    'seconds_saved': previous['seconds_per_source'] - row['seconds_per_source'],
                    'pair_station_deletions': row['policy_stats']['pair_station_deletions'],
                    'wall_seconds': row['wall_seconds'], 'previous_wall_seconds': previous['wall_seconds']}
            for label, record in (('previous', previous), ('paired', row)):
                stats = record['world_stats']
                pair[label + '_movement'] = stats['move_meters'] / (5 * count)
                pair[label + '_radio'] = (5 * stats['measure_count'] + stats['switch_count']) / count
                pair[label + '_optical'] = (5 * stats['successful_clear_count'] + 3 * stats['failed_clear_count']) / count
            pairs.append(pair)
    for mode in ('previous', 'identity', 'paired'):
        selected = [row for row in rows if row['mode'] == mode]
        mean = statistics.mean(row['seconds_per_source'] for row in selected)
        if not math.isclose(mean, summary[mode]['mean'], abs_tol=1e-9):
            raise ValueError('Complete-mission aggregate differs')
    return {'runs': len(rows), 'all_cleared': len(rows), 'actions_audited': action_count,
            'exact_physics_replays': len(rows), 'identity_cases': len(scenes),
            'max_clock_error': max_clock_error, 'commitments_audited': events_audited,
            'frozen_source_files': len(frozen), 'frozen_manifests': runner.frozen_integrity(),
            'summary': summary, 'paired_cases': pairs, 'official_calls': 0, 'formal_calls': 0}


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
