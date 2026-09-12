from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research/dual_guided_network_v44'))
import search_network as reference
import numpy as np
import shapely


def audit_implications(groups, proof):
    if proof['original_count'] != len(groups) or len(proof['implications']) != len(groups):
        raise ValueError('Original group count mismatch')
    normalized = [set(group) for group in groups]
    retained = proof['kept_indices']
    if retained != sorted(set(retained)) or len(retained) != proof['retained_count']:
        raise ValueError('Invalid retained index list')
    if any(index < 0 or index >= len(groups) for index in retained):
        raise ValueError('Retained group outside the original list')
    kept_set = set(retained)
    fixed = set(proof['forced_true'])
    for original_index, implication in enumerate(proof['implications']):
        if implication['kind'] == 'fixed_true':
            if implication['point'] not in fixed.intersection(normalized[original_index]):
                raise ValueError('Invalid fixed-point implication')
        elif implication['kind'] == 'retained_subset':
            witness = implication['index']
            if witness not in kept_set or not normalized[witness].issubset(normalized[original_index]):
                raise ValueError('Invalid subset implication')
        else:
            raise ValueError('Unknown implication type')
    fixed_count = sum(item['kind'] == 'fixed_true' for item in proof['implications'])
    if fixed_count != proof['fixed_satisfied_count'] or len(groups) - fixed_count - len(retained) != proof['subsumed_count']:
        raise ValueError('Deletion counts are inconsistent')
    return True


def independent_groups(pool, positions, headings):
    groups = set()
    for position, heading in zip(positions, headings):
        offsets = pool - position
        eligible = np.flatnonzero((np.einsum('ij,ij->i', offsets, offsets) <= 999.9**2)
                                  & (offsets @ heading >= 0.01))
        groups.add(tuple(int(index) for index in eligible))
    return groups


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
                raise ValueError('Source integrity failed: ' + str(source))
    if (protocol['forced_true'], protocol['minimum_points'], protocol['maximum_points'],
        protocol['seconds_per_round'], protocol['solver_workers'], protocol['solver_seed']) != ([0], 19, 20, 30, 1, 0):
        raise ValueError('Registered solver configuration changed')
    pool = np.asarray(load(batch / 'pool.json'))
    if not np.array_equal(pool, reference.pool_points()):
        raise ValueError('Registered pool changed')
    reference_sites = reference.dual_ring_network()
    reference_indices = set()
    for point in reference_sites:
        distances = np.linalg.norm(pool - point, axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] >= 1e-8:
            raise ValueError('Reference not in point pool')
        reference_indices.add(nearest)
    if len(reference_indices) != 21 or not reference.certify(reference_sites).empty:
        raise ValueError('Known reference certificate failed')
    positions, headings = reference.initial_witnesses()
    deleted_checked = 0
    all_checked = 0
    summaries = []
    for iteration_file in sorted((batch / 'iterations').glob('*.json')):
        row = load(iteration_file)
        groups = load(batch / 'groups' / iteration_file.name)
        proof = load(batch / 'proofs' / iteration_file.name)
        if proof['forced_true'] != [0] or proof['point_count'] != len(pool):
            raise ValueError('Fixed variables or pool count changed')
        if set(tuple(group) for group in groups) != independent_groups(pool, positions, headings):
            raise ValueError('Raw geometry groups differ from independent rebuild')
        audit_implications(groups, proof)
        if not all(reference_indices.intersection(group) for group in groups):
            raise ValueError('Known 21-point reference violates a group')
        all_checked += len(groups)
        deleted_checked += len(groups) - len(proof['kept_indices'])
        if 'selected_indices' in row:
            selected = row['selected_indices']
            chosen_set = set(selected)
            if len(selected) != len(chosen_set) or not 19 <= len(selected) <= 20 or 0 not in chosen_set:
                raise ValueError('Candidate cardinality failed')
            if not all(chosen_set.intersection(group) for group in groups):
                raise ValueError('Candidate misses an original constraint')
            points = pool[selected]
            pair_distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
            if np.any(pair_distances[np.triu_indices(len(points), 1)] < 50):
                raise ValueError('Candidate spacing failed')
            coverage = reference.certify(points)
            if coverage.empty != row['complete'] or not coverage.region.equals(shapely.from_wkt(row['remaining_geometry'])):
                raise ValueError('Continuous certificate differs')
        for example in row.get('counterexamples', []):
            positions = np.vstack((positions, example['position']))
            headings = np.vstack((headings, example['heading']))
        summaries.append({'iteration': row['iteration'], 'status': row['status'],
                          'original_groups': len(groups), 'retained_groups': proof['retained_count'],
                          'returned_candidate': 'selected_indices' in row})
    summary = load(batch / 'summary.json')
    if summary['iterations'] != len(summaries) or not summary['source_hashes_unchanged']:
        raise ValueError('Summary integrity failed')
    result = {
        'status': 'passed', 'iterations': summaries, 'source_files_checked': len(protocol['source_hashes']),
        'original_groups_independently_rebuilt': all_checked, 'deletion_implications_checked': deleted_checked,
        'known_21_point_reference_passed': True, 'continuous_candidates': summary['continuous_complete'],
        'new_holdout': False, 'official_calls': 0, 'formal_calls': 0, 'policy_runs': 0,
        'conclusion': 'Subsumption preserves feasible assignments; solver status is not a complete-mission score',
    }
    reference.provenance.save(output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
