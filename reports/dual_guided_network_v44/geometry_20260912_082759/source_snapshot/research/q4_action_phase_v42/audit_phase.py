from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import run_phase as runner
from coverage_certificate import TriangleCoverage
from geometry import open_route, route_length
from offline_benchmark import OfflineRuleWorld, Source
from phase_policy import PHASES, rotate_stations


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit_event(event, trace):
    before = [action for action in trace if action['virtual_seconds'] <= event['actual_virtual_seconds']]
    if not before or before[-1]['virtual_seconds'] != event['actual_virtual_seconds']:
        raise ValueError('Phase event has no actual clock anchor')
    if event['before_radio'] != [[0.0, 0.0]] or any(action['position'] != [0.0, 0.0] for action in before):
        raise ValueError('Rotation occurred after off-origin evidence')
    for channel in event['unknown_channels']:
        for field, kind, result in (('before_radio', 'measure', 'no_signal'),
                                     ('before_optical', 'clear', 'no_target_in_range')):
            for point in event[field]:
                if not any(action['action'] == kind and action['channel'] == channel and action['position'] == point
                           and action['response']['result'] == result for action in before):
                    raise ValueError('Coverage certificate uses unexecuted or nonnegative observations')
    original = np.asarray(event['original_route'])
    selected = np.asarray(event['selected_route'])
    if not np.array_equal(selected, rotate_stations(original, event['phase_degrees'])):
        raise ValueError('The selected layout is not the registered rigid rotation')
    if [candidate['phase_degrees'] for candidate in event['candidates']] != list(PHASES):
        raise ValueError('Unregistered candidate phases')
    costs = []
    for candidate in event['candidates']:
        proposal = rotate_stations(original, candidate['phase_degrees'])
        trial = TriangleCoverage()
        for point in event['before_radio']:
            trial.observe_absence(point)
        for point in event['before_optical']:
            trial.observe_clear_absence(point)
        for point in proposal[1:]:
            trial.observe_absence(point)
        if candidate['complete'] != trial.empty or not math.isclose(candidate['remaining_area'], trial.area, abs_tol=1e-6):
            raise ValueError('Continuous certificate differs in independent reconstruction')
        if trial.empty:
            points = np.vstack((proposal[1:], [target['position'] for target in event['targets']]))
            cost = route_length(open_route(points, event['actual_position']), event['actual_position'])
            if not math.isclose(cost, candidate['proxy_meters'], abs_tol=1e-7):
                raise ValueError('Joint action-route proxy differs')
            costs.append((cost, candidate['phase_degrees']))
    chosen = costs[0]
    for candidate in costs[1:]:
        if candidate[0] < chosen[0] - 1e-6:
            chosen = candidate
    if event['phase_degrees'] != chosen[1] or not math.isclose(chosen[0], event['selected_proxy_meters'], abs_tol=1e-7):
        raise ValueError('Chosen phase violates preregistered deterministic selection')
    actual_new_stations = 0
    if event['phase_degrees']:
        for point in selected[1:]:
            if any(action['action'] == 'measure' and action['position'] == point.tolist()
                   and action['virtual_seconds'] > event['actual_virtual_seconds'] for action in trace):
                actual_new_stations += 1
    return {'candidate_checks': len(event['candidates']), 'actual_new_stations': actual_new_stations}


def audit_batch(directory):
    protocol = load(directory / 'protocol.json')
    for relative, digest in protocol['source_hashes'].items():
        for path in (runner.ROOT / relative, directory / 'source_snapshot' / relative):
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Source or snapshot changed: ' + relative)
    scenes = {scene['id']: scene for scene in load(directory / 'scenes_evaluator_only.json')}
    rows, paired = [], []
    actions, candidates, actual_stations, events = 0, 0, 0, 0
    clock_error = 0.0
    for path in sorted((directory / 'records').glob('*.json')):
        row = load(path)
        trace = load(directory / 'traces' / path.name)
        if not row['success'] or row['cleared_count'] != row['count']:
            raise ValueError('Incomplete task: ' + path.name)
        if hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest() != row['trace_sha256']:
            raise ValueError('Trace digest differs')
        clock = runner.baseline.baseline.audit_trace(trace, row['virtual_seconds'])
        actions += clock['actions']
        clock_error = max(clock_error, clock['max_clock_error'])
        scene = scenes[row['scene_id']]
        world = OfflineRuleWorld([Source(**source) for source in scene['sources']], scene['seed'], scene['error_mode'])
        for action in trace:
            getattr(world, action['action'])(np.asarray(action['position']), action['channel'])
            if world.trace[-1] != action:
                raise ValueError('Physical replay differs')
        if world.remaining:
            raise ValueError('A source remains after replay')
        for event in row['phase_events']:
            check = audit_event(event, trace)
            events += 1
            candidates += check['candidate_checks']
            actual_stations += check['actual_new_stations']
        rows.append(row)
    if len(rows) != len(scenes) * len(protocol['modes']):
        raise ValueError('Registered task is missing')
    reference = {row['scene_id']: row for row in rows if row['mode'] == 'previous'}
    for row in rows:
        if row['mode'] == 'identity' and row['trace_sha256'] != reference[row['scene_id']]['trace_sha256']:
            raise ValueError('Identity is not action exact')
        stats = row['world_stats']
        paired.append({'profile': row['profile'], 'count': row['count'], 'mode': row['mode'],
                       'seconds_per_source': row['seconds_per_source'],
                       'saved_seconds_per_source': reference[row['scene_id']]['seconds_per_source'] - row['seconds_per_source'],
                       'movement': stats['move_meters'] / (5 * row['count']),
                       'radio': (5 * stats['measure_count'] + stats['switch_count']) / row['count'],
                       'optical': (5 * stats['successful_clear_count'] + 3 * stats['failed_clear_count']) / row['count'],
                       'phase': row['policy_stats'].get('phase_degrees', 0), 'wall_seconds': row['wall_seconds']})
    summary = runner.summarize(rows, protocol['modes'])
    if summary != load(directory / 'summary.json'):
        raise ValueError('Case-equal complete-mission summary differs')
    return {'runs': len(rows), 'successes': len(rows), 'actions_audited': actions, 'max_clock_error': clock_error,
            'exact_physics_replays': len(rows), 'identity_cases': len(scenes), 'phase_events_audited': events,
            'candidate_coverage_checks': candidates, 'actual_rotated_stations_with_measurements': actual_stations,
            'source_files': len(protocol['source_hashes']), 'frozen_manifests': runner.frozen_integrity(),
            'summary': summary, 'paired_cases': paired, 'official_calls': 0, 'formal_calls': 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=Path, required=True)
    args = parser.parse_args()
    directory = args.batch.resolve()
    output = directory / 'analysis.json'
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Audit destination must be a fresh D-drive file')
    result = audit_batch(directory)
    runner.experiment.save(output, result)
    print(json.dumps({key: value for key, value in result.items() if key not in ('summary', 'paired_cases')}, indent=2))


if __name__ == '__main__':
    main()
