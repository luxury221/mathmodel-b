from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'research/dual_guided_network_v44'))
import search_network as previous
from reduction import reduce_cover


def source_hashes():
    combined = previous.hashes()
    old_protocol = previous.provenance.load(ROOT / 'reports/dual_guided_network_v44/geometry_20260912_082759/protocol.json')
    for relative, expected in old_protocol['source_hashes'].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen v44 source changed: ' + relative)
        combined[relative] = expected
    files = [*HERE.glob('*.py'), HERE / 'PROTOCOL.md',
             *sorted((ROOT / 'research/cover_subsumption_audit_v45').glob('*.py'))]
    return {**combined, **{str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a fresh D-drive output directory')
    frozen = source_hashes()
    previous.provenance.save(output / 'protocol.json', {
        'created_local': datetime.now().isoformat(), 'source_hashes': frozen,
        'reference_batch': 'reports/dual_guided_network_v44/geometry_20260912_082759',
        'maximum_rounds': 8, 'seconds_per_round': 30, 'solver_workers': 1, 'solver_seed': 0,
        'minimum_points': 19, 'maximum_points': 20, 'forced_true': [0],
        'new_holdout': False, 'policy_runs': 0, 'official_calls': 0, 'formal_calls': 0,
    })
    for relative in frozen:
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    pool = previous.pool_points()
    positions, headings = previous.initial_witnesses()
    previous.provenance.save(output / 'pool.json', pool.tolist())
    reference = previous.geometry_price(previous.dual_ring_network())
    previous.provenance.save(output / 'reference.json', reference)
    rows, forbidden, eligible = [], [], []
    for iteration in range(8):
        groups = previous.visible_groups(pool, positions, headings)
        started = time.perf_counter()
        proof = reduce_cover(groups, len(pool), forced_true=(0,))
        reduction_seconds = time.perf_counter() - started
        reduced = [groups[index] for index in proof['kept_indices']]
        previous.provenance.save(output / 'groups' / f'{iteration:02d}.json', groups)
        previous.provenance.save(output / 'proofs' / f'{iteration:02d}.json', proof)
        selected, row = previous.solve_cover(pool, reduced, forbidden)
        row.update({
            'iteration': iteration, 'source_witness_count': len(positions),
            'original_groups': len(groups), 'retained_groups': len(reduced),
            'fixed_satisfied_groups': proof['fixed_satisfied_count'],
            'subsumed_groups': proof['subsumed_count'], 'reduction_seconds': reduction_seconds,
        })
        if selected is not None:
            points = pool[selected]
            coverage = previous.certify(points)
            row.update({'selected_indices': selected, 'points': points.tolist(), 'complete': coverage.empty,
                        'remaining_area': coverage.area, 'remaining_geometry': coverage.region.wkt,
                        'geometry_price': previous.geometry_price(points)})
            improvement = 100 * (1 - row['geometry_price']['proxy_cost_meters'] / reference['proxy_cost_meters'])
            row['proxy_improvement_percent'] = improvement
            if coverage.empty and improvement >= 1:
                eligible.append(row)
            if not coverage.empty:
                examples = previous.counterexamples(coverage.region, points, limit=64, receiving_radius=999.9)
                row['counterexamples'] = [{'position': item[1].tolist(), 'heading': item[2].tolist()} for item in examples]
                if examples:
                    positions = previous.np.vstack((positions, [item[1] for item in examples]))
                    headings = previous.np.vstack((headings, [item[2] for item in examples]))
            forbidden.append(selected)
        rows.append(row)
        previous.provenance.save(output / 'iterations' / f'{iteration:02d}.json', row)
        print(json.dumps({key: row[key] for key in ('iteration', 'status', 'original_groups', 'retained_groups',
                                                  'reduction_seconds', 'wall_seconds', 'branches')}, ensure_ascii=False), flush=True)
        if selected is None or eligible:
            break
    summary = {
        'iterations': len(rows), 'statuses': [row['status'] for row in rows],
        'returned_candidates': sum('points' in row for row in rows),
        'continuous_complete': sum(row.get('complete', False) for row in rows),
        'eligible_count': len(eligible), 'eligible': eligible[0] if eligible else None,
        'source_hashes_unchanged': frozen == source_hashes(),
        'original_groups_first_round': rows[0]['original_groups'],
        'retained_groups_first_round': rows[0]['retained_groups'],
        'new_holdout': False, 'policy_runs': 0, 'official_calls': 0, 'formal_calls': 0,
        'scope': 'Equivalent discrete covering model, not a mission-time result or global infeasibility proof',
    }
    previous.provenance.save(output / 'summary.json', summary)
    print(json.dumps({key: value for key, value in summary.items() if key != 'eligible'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
