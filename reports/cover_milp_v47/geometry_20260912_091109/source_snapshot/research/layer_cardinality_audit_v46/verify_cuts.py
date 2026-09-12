from __future__ import annotations

import argparse
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research/dual_guided_network_v44'))
import search_network as reference


def verify_cut(groups, cut):
    points = cut['point_indices']
    if len(points) != len(set(points)) or any(not isinstance(point, int) or point <= 0 for point in points):
        raise ValueError('Point indices must be distinct and exclude the fixed origin')
    loads = {point: 0 for point in points}
    denominator = cut['denominator']
    if not isinstance(denominator, int) or denominator <= 0:
        raise ValueError('Invalid denominator')
    total = 0
    seen = set()
    for group_index, numerator in cut['weighted_groups']:
        if not isinstance(numerator, int) or numerator <= 0 or group_index in seen or not 0 <= group_index < len(groups):
            raise ValueError('Invalid weighted group')
        members = set(groups[group_index])
        if not members or not members.issubset(loads):
            raise ValueError('A weighted group leaves the certified point subset')
        seen.add(group_index)
        total += numerator
        for member in members:
            loads[member] += numerator
    if any(load > denominator for load in loads.values()):
        raise ValueError('Exact point budget exceeded')
    bound = Fraction(total, denominator)
    integer_lower = -(-bound.numerator // bound.denominator)
    if total != cut['total_numerator'] or max(loads.values(), default=0) != cut['maximum_point_load']:
        raise ValueError('Recorded integer totals do not match')
    if integer_lower != cut['required_points']:
        raise ValueError('Recorded integer lower bound does not match')
    return len(points)


def verify_combination(cuts, combination):
    occupied = set()
    result = 1
    for index in combination['disjoint_cut_indices']:
        if not isinstance(index, int) or not 0 <= index < len(cuts):
            raise ValueError('Invalid combined cut index')
        cut = cuts[index]
        layers = set(range(cut['first_layer'], cut['last_layer'] + 1))
        if occupied.intersection(layers):
            raise ValueError('Overlapping intervals cannot have their lower bounds added')
        occupied.update(layers)
        result += cut['required_points']
    if result != combination['with_forced_origin_lower'] or result - 1 != combination['nonorigin_lower']:
        raise ValueError('Combined lower bound differs')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
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
    groups = load(batch / 'groups' / '00.json')
    pool = reference.np.asarray(load(batch / 'pool.json'))
    reference_sites = reference.dual_ring_network()
    reference_indices = {int(reference.np.argmin(reference.np.linalg.norm(pool - point, axis=1))) for point in reference_sites}
    cuts = load(batch / 'interval_certificates.json')
    expected_intervals = {(first, last) for first in range(9) for last in range(first, 9)}
    if len(cuts) != 45 or {(cut['first_layer'], cut['last_layer']) for cut in cuts} != expected_intervals:
        raise ValueError('Registered intervals changed')
    budgets_checked = 0
    for cut in cuts:
        points = list(range(1 + cut['first_layer'] * 72, 1 + (cut['last_layer'] + 1) * 72))
        if points != cut['point_indices']:
            raise ValueError('Point list differs from registered layer interval')
        budgets_checked += verify_cut(groups, cut)
        if len(reference_indices.intersection(points)) < cut['required_points']:
            raise ValueError('Known reference violates a claimed valid cut')
    combined = verify_combination(cuts, load(batch / 'interval_combination.json'))
    row = load(batch / 'iterations' / '00.json')
    if 'selected_indices' in row:
        selected = set(row['selected_indices'])
        if not all(len(selected.intersection(cut['point_indices'])) >= cut['required_points'] for cut in cuts):
            raise ValueError('Returned candidate violates a valid cut')
    result = {
        'status': 'passed', 'source_files_checked': len(protocol['source_hashes']),
        'interval_certificates': len(cuts), 'exact_point_budgets_checked': budgets_checked,
        'positive_cardinality_cuts': sum(cut['required_points'] > 0 for cut in cuts),
        'global_integer_lower': combined, 'known_21_point_reference_satisfies_all_cuts': True,
        'solver_status': row['status'], 'pool_19_20_ruled_out_by_integer_certificate': combined > 20,
        'continuous_plane_infeasibility_claimed': False, 'policy_runs': 0,
        'new_holdout': False, 'official_calls': 0, 'formal_calls': 0,
    }
    reference.provenance.save(output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
