from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import shapely


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'research/cover_subsumption_audit_v45'),
                str(ROOT / 'research/layer_cardinality_audit_v46')]
import verify as coverage_audit
from verify_cuts import verify_combination, verify_cut


reference = coverage_audit.reference


def audit_model_rows(pool, reduced, cuts, rows, minimum=19, maximum=20):
    expected = [{'kind': 'fixed_origin', 'members': [0], 'lower': 1, 'upper': 1},
                {'kind': 'cardinality', 'members': list(range(len(pool))), 'lower': minimum, 'upper': maximum}]
    expected.extend({'kind': 'cover', 'members': list(group), 'lower': 1, 'upper': max(len(group), 1)} for group in reduced)
    for first in range(len(pool)):
        for second in range(first + 1, len(pool)):
            squared = sum(float(value)**2 for value in pool[first] - pool[second])
            if 0 < squared < 2500:
                expected.append({'kind': 'spacing', 'members': [first, second], 'lower': 0, 'upper': 1})
    for cut in cuts:
        if cut['required_points'] > 0:
            expected.append({'kind': 'layer_cut', 'members': cut['point_indices'],
                             'lower': cut['required_points'], 'upper': len(cut['point_indices'])})
    if expected != rows:
        raise ValueError('Linear integer model differs from the registered Boolean model')
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    batch, output = arguments.batch.resolve(), arguments.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a fresh D-drive audit file')
    load = reference.provenance.load
    protocol = load(batch / 'protocol.json')
    for relative, expected in protocol['source_hashes'].items():
        for source in (ROOT / relative, batch / 'source_snapshot' / relative):
            if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                raise ValueError('Frozen source changed: ' + str(source))
    for relative, expected in protocol['input_hashes'].items():
        for source in (batch / relative, ROOT / protocol['input_batch'] / relative):
            if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                raise ValueError('Frozen input changed: ' + str(source))
    if (protocol['time_limit'], protocol['mip_rel_gap'], protocol['minimum_points'], protocol['maximum_points']) != (30, 0, 19, 20):
        raise ValueError('Registered integer model or budget changed')
    pool = np.asarray(load(batch / 'pool.json'))
    if not np.array_equal(pool, reference.pool_points()):
        raise ValueError('Point pool changed')
    groups = load(batch / 'groups' / '00.json')
    proof = load(batch / 'proofs' / '00.json')
    coverage_audit.audit_implications(groups, proof)
    positions, headings = reference.initial_witnesses()
    if set(tuple(group) for group in groups) != coverage_audit.independent_groups(pool, positions, headings):
        raise ValueError('Raw coverage witnesses differ')
    cuts = load(batch / 'interval_certificates.json')
    budget_checks = sum(verify_cut(groups, cut) for cut in cuts)
    verify_combination(cuts, load(batch / 'interval_combination.json'))
    reduced = [groups[index] for index in proof['kept_indices']]
    rows = load(batch / 'model_rows.json')
    model_checks = audit_model_rows(pool, reduced, cuts, rows)
    row = load(batch / 'iterations' / '00.json')
    if 'selected_indices' in row:
        selected = set(row['selected_indices'])
        if len(selected) != len(row['selected_indices']) or not 19 <= len(selected) <= 20 or 0 not in selected:
            raise ValueError('Candidate cardinality failed')
        if not all(set(group).intersection(selected) for group in groups):
            raise ValueError('Candidate violates original coverage')
        for model_row in rows:
            count = len(selected.intersection(model_row['members']))
            if not model_row['lower'] <= count <= model_row['upper']:
                raise ValueError('Candidate violates an exact model row')
        points = pool[row['selected_indices']]
        if not np.array_equal(points, np.asarray(row['points'])):
            raise ValueError('Candidate coordinate list differs')
        coverage = reference.certify(points)
        if coverage.empty != row['complete'] or not coverage.region.equals(shapely.from_wkt(row['remaining_geometry'])):
            raise ValueError('Continuous certificate differs')
    summary = load(batch / 'summary.json')
    if not summary['source_hashes_unchanged'] or summary['statuses'] != [row['status']]:
        raise ValueError('Summary differs from raw result')
    result = {
        'status': 'passed', 'source_files_checked': len(protocol['source_hashes']),
        'original_groups_rebuilt': len(groups), 'exact_point_budget_checks': budget_checks,
        'integer_model_rows_checked': model_checks, 'solver_status': row['status'],
        'returned_candidates': summary['returned_candidates'], 'continuous_complete': summary['continuous_complete'],
        'eligible_count': summary['eligible_count'], 'official_calls': 0, 'formal_calls': 0,
        'policy_runs': 0, 'new_holdout': False, 'continuous_plane_infeasibility_claimed': False,
    }
    reference.provenance.save(output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
