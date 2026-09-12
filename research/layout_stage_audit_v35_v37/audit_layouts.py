from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'research/hybrid_discovery_v35'), str(ROOT / 'research/motion_stage_audit'),
                str(ROOT / 'research/practice_baseline')]
import numpy as np
from audit_motion import audit_directory, comparisons
from baseline import verify_hashes
from geometry_hybrid import mixed_price, residual_after_radio
from topology import uncovered_witnesses


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def verify_snapshot(directory):
    hashes = load(directory / 'protocol.json')['source_hashes']
    for relative, expected in hashes.items():
        for path in (ROOT / relative, directory / 'source_snapshot' / relative):
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('Source snapshot changed: ' + str(path))
    return len(hashes)


def test_positions():
    axis = np.arange(-1800, 1801, 60, dtype=float)
    grid = np.array([(horizontal, vertical) for horizontal in axis for vertical in axis
                     if horizontal**2 + vertical**2 <= 1800**2])
    angles = np.arange(1024) * (2 * math.pi / 1024) + 0.000123
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    circles = np.vstack([radius * directions for radius in (19.999, 20.001, 80, 250, 500, 750, 900, 999.9,
                                                            1000.1, 1125, 1250, 1500, 1700, 1799.99, 1800)])
    return np.vstack((grid, circles))


def audit_hybrid(directory):
    source_count = verify_snapshot(directory)
    rows = [load(path) for path in sorted((directory / 'certificates').glob('*.json'))]
    positions = test_positions()
    complete = []
    for row in rows:
        if not row['complete']:
            continue
        coverage = residual_after_radio(row['stations'])
        if not math.isclose(coverage.area, row['radio_remaining_area'], abs_tol=1e-6):
            raise ValueError('Recorded radio remainder does not reproduce')
        patched = np.linalg.norm(positions, axis=1) <= 19.99
        for point in row['optical_centers']:
            coverage.observe_clear_absence(point)
            patched |= np.linalg.norm(positions - point, axis=1) <= 19.99
        if not coverage.empty:
            raise ValueError('Complete hybrid record retained a continuous hole')
        missing = uncovered_witnesses(row['stations'], positions) & ~patched
        if missing.any():
            raise ValueError('Independent continuous-heading sample check found an uncovered position')
        price = mixed_price(row['stations'], row['optical_centers'])
        if not math.isclose(price['proxy_cost_meters'], row['proxy_cost_meters'], abs_tol=1e-6):
            raise ValueError('Mixed route price differs')
        complete.append({'id': row['id'], 'radio_count': row['radio_count'],
                         'additional_optical_stops': len(row['optical_centers']),
                         'proxy_cost_meters': price['proxy_cost_meters']})
    if len(complete) != load(directory / 'summary.json')['complete_hybrid_layouts']:
        raise ValueError('Hybrid complete count differs')
    return {'source_files': source_count, 'geometry_records': len(rows), 'complete_reproduced': len(complete),
            'positions_checked_per_complete_layout': len(positions), 'heading_check': 'exact_maximum_angular_gap_at_each_check_position',
            'sample_check_does_not_replace_continuous_certificate': True, 'complete_layouts': complete}


def audit_repair(directory):
    source_count = verify_snapshot(directory)
    rows = [load(path) for path in sorted((directory / 'iterations').glob('*.json'))]
    checked = 0
    for row in rows:
        if 'stations' not in row:
            continue
        coverage = residual_after_radio(row['stations'])
        if coverage.empty != row['complete']:
            raise ValueError('Repair certificate result differs')
        if 'remaining_area' in row and not math.isclose(coverage.area, row['remaining_area'], abs_tol=1e-6):
            raise ValueError('Repair remainder area differs')
        checked += 1
    return {'source_files': source_count, 'iterations': len(rows), 'geometry_records_reproduced': checked,
            'complete': sum(row['complete'] for row in rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hybrid', type=Path, required=True)
    parser.add_argument('--rotation', type=Path, required=True)
    parser.add_argument('--repair', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a fresh D-drive audit output')
    rotation, rows = audit_directory(args.rotation)
    if rotation['current_changes'] or rotation['archived_changes'] or rotation['tiny_distinct_remeasurements']:
        raise ValueError('Rotation audit failed')
    if rotation['successes'] != rotation['runs'] or rotation['max_clock_error'] > 1e-6:
        raise ValueError('Rotation clearance/clock audit failed')
    reference = {row['scene_id']: row for row in rows if row['mode'] == 'previous'}
    identities = [row for row in rows if row['mode'] == 'identity']
    if len(identities) != 6 or any(row['trace_sha256'] != reference[row['scene_id']]['trace_sha256'] for row in identities):
        raise ValueError('Identity wrapper was not action-exact')
    frozen = load(ROOT / 'reports/scan_validation_v13/validation_20260911_235435/selection.json')['source_hashes']
    for relative, digest in frozen.items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen candidate dependency changed')
    verify_hashes()
    report = {'official_calls': 0, 'new_holdout_used': False, 'frozen_files_verified': len(frozen),
              'hybrid': audit_hybrid(args.hybrid), 'rotation': rotation, 'rotation_comparison': comparisons(rows),
              'identity_exact_cases': len(identities), 'repair': audit_repair(args.repair),
              'passed': True, 'auditor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.mkdir(parents=True)
    (output / 'analysis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'passed': True, 'hybrid_complete_reproduced': report['hybrid']['complete_reproduced'],
                      'repair_records_reproduced': report['repair']['geometry_records_reproduced'],
                      'rotation_runs': rotation['runs'], 'rotation_actions': rotation['actions'],
                      'max_clock_error': rotation['max_clock_error'], 'frozen_files_verified': len(frozen)}, indent=2))


if __name__ == '__main__':
    main()
