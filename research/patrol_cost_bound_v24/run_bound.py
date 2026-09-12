from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / 'research/offline_validation')]
import numpy as np
from geometry import dual_ring_network
from route_bound import neighborhood_costs, open_path_lower_bound


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--time-limit', type=float, default=20.0)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a new D-drive output')
    source_paths = [*HERE.glob('*.py'), ROOT / 'research/offline_validation/geometry.py']
    frozen = {str(path.relative_to(ROOT)): digest(path) for path in source_paths}
    save(output / 'protocol.json', {'created_local': datetime.now().isoformat(), 'source_hashes': frozen,
                                    'official_calls': 0, 'scene_source_tables_read': False,
                                    'scope': 'conditional mandatory-21-station and stricter full-absent-channel-scan classes'})
    shutil.copy2(HERE / 'PROTOCOL.md', output / 'PROTOCOL.md')
    for relative in frozen:
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    stations = dual_ring_network()
    records = []
    seen = set()
    station_bound = open_path_lower_bound(neighborhood_costs(stations, np.zeros(len(stations))), args.time_limit)
    save(output / 'station_bound.json', station_bound)
    print('public_network_lower_bound_meters', station_bound['lower_bound_meters'], flush=True)
    for directory in args.input:
        for path in sorted((directory / 'records').glob('*__previous.json')):
            row = load(path)
            if row['problem'] != 'q4' or row['count'] == 16:
                continue
            if not row['success'] or row['scene_id'] in seen or row['count'] not in (10, 13):
                raise ValueError('Require unique successful 10/13-source development traces')
            seen.add(row['scene_id'])
            trace_path = directory / 'traces' / path.name
            trace = load(trace_path)
            successes = [action for action in trace if action['action'] == 'clear' and action['response']['result'] == 'success']
            if len(successes) != row['count'] or len({action['channel'] for action in successes}) != row['count']:
                raise ValueError('Successful-clear log disagrees with recorded count')
            centers = np.vstack((stations, np.array([action['position'] for action in successes])))
            radii = np.r_[np.zeros(len(stations)), np.full(len(successes), 40.0)]
            bound = open_path_lower_bound(neighborhood_costs(centers, radii), args.time_limit)
            count = row['count']
            movement = bound['lower_bound_meters'] / 5 / count
            absent_radio = 5 * len(stations) * (20 - count) / count
            record = {'scene_id': row['scene_id'], 'profile': row['profile'], 'count': count,
                      'record_path': str(path.resolve()), 'record_sha256': digest(path),
                      'trace_path': str(trace_path.resolve()), 'trace_sha256': digest(trace_path),
                      'observed_baseline_seconds_per_source': row['seconds_per_source'],
                      'conditional_travel_bound_seconds_per_source': movement,
                      'conditional_travel_plus_clear_bound': movement + 5,
                      'stricter_full_absent_scan_bound': movement + 5 + absent_radio,
                      'stricter_absent_measurement_floor': absent_radio, 'solver': bound}
            records.append(record)
            save(output / 'records' / path.name, record)
            print(row['profile'], count, 'travel+clear', round(movement + 5, 3),
                  'full-scan', round(movement + 5 + absent_radio, 3), flush=True)
    if len(records) != 12:
        raise ValueError(f'Expected the 12 existing development cases, got {len(records)}')
    summary = {key: statistics.mean(row[key] for row in records) for key in
               ('observed_baseline_seconds_per_source', 'conditional_travel_plus_clear_bound', 'stricter_full_absent_scan_bound')}
    summary.update({'cases': len(records), 'not_a_universal_algorithm_bound': True,
                    'not_an_actual_policy_performance_result': True,
                    'conditional_cases_above_300': sum(row['conditional_travel_plus_clear_bound'] > 300 for row in records),
                    'stricter_cases_above_300': sum(row['stricter_full_absent_scan_bound'] > 300 for row in records),
                    'source_integrity': frozen == {str(path.relative_to(ROOT)): digest(path) for path in source_paths}})
    save(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
