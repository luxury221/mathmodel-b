from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a fresh D-drive result path')
    batch = ROOT / 'reports/frozen_practice_v33/batch_20260912_040922'
    bound_directory = ROOT / 'reports/patrol_cost_bound_v24/diagnostic_20260912_021849'
    bound_path = bound_directory / 'station_bound.json'
    verification_path = ROOT / 'reports/frozen_practice_v33/audit_20260912_044703/final_verification.json'
    verification = load(verification_path)
    if not verification['passed'] or verification['exact_replays'] != 10:
        raise ValueError('Official baseline audit did not pass')
    for directory in (batch, bound_directory):
        for relative, expected in load(directory / 'protocol.json')['source_hashes'].items():
            for path in (ROOT / relative, directory / 'source_snapshot' / relative):
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise ValueError('Source snapshot changed: ' + str(path))
    bound = load(bound_path)
    lower = bound['lower_bound_meters']
    if not bound['connected_relaxation_optimal'] or not 0 < lower < bound['raw_solver_dual']:
        raise ValueError('Unexpected conditional lower-bound evidence')
    selected = [row for row in load(batch / 'runs.json') if row['problem'] == 'q4']
    if len(selected) != 5 or any(row['state'] != 'completed' for row in selected):
        raise ValueError('Incomplete Q4 official batch')
    cases = []
    for row in selected:
        count = row['source_count']
        stats = row['audit']['policy_stats']
        if stats['visited_stations'] != 20 or stats['certified_station_prunes'] or stats['station_replacements']:
            raise ValueError('Observed batch does not match the fixed-network diagnostic')
        travel_clear = (lower / 5 + count * 5) / count
        absent_scans = travel_clear + (20 - count) * 21 * 5 / count
        cases.append({'case_code': row['case_code'], 'public_result_source_count': count,
                      'conditional_fixed21_travel_clear': travel_clear,
                      'conditional_fixed21_full_absent_scans': absent_scans})
    result = {'diagnostic_only': True, 'official_calls': 0, 'new_performance_tests': 0,
              'station_lower_bound_meters': lower,
              'fixed21_travel_clear_mean': statistics.mean(case['conditional_fixed21_travel_clear'] for case in cases),
              'fixed21_full_absent_scans_mean': statistics.mean(case['conditional_fixed21_full_absent_scans'] for case in cases),
              'cases': cases,
              'conditions': ['Forced visit to all 21 fixed positions', 'Clear all sources',
                             'The second bound additionally forces every absent channel to be measured at all 21 positions'],
              'not_a_universal_algorithm_bound': True,
              'not_an_official_score': True,
              'bound_evidence': str(bound_path), 'bound_sha256': hashlib.sha256(bound_path.read_bytes()).hexdigest(),
              'official_verification_sha256': hashlib.sha256(verification_path.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('fixed21_travel_clear_mean', 'fixed21_full_absent_scans_mean', 'not_a_universal_algorithm_bound')}, indent=2))


if __name__ == '__main__':
    main()
