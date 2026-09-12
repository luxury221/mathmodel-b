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
sys.path[:0] = [str(ROOT / 'research/radial_dual_bound_v43'), str(ROOT / 'research/network_synthesis_v8'),
                str(ROOT / 'research/annular_topology_v25')]
import run_bound as provenance
import numpy as np
from ortools.sat.python import cp_model
from synthesis import certify, counterexamples, initial_witnesses
from geometry import dual_ring_network, ring
from topology import geometry_price


RADII = (201, 955, 957, 1000, 1081, 1371, 1497, 1870, 1903)


def pool_points():
    return np.vstack(([0.0, 0.0], *(ring(radius, 72) for radius in RADII)))


def visible_groups(pool, positions, headings):
    matrix = []
    for point in pool:
        vectors = point - positions
        matrix.append((np.sum(vectors * vectors, axis=1) <= 999.9**2)
                      & (np.sum(vectors * headings, axis=1) >= 0.01))
    matrix = np.asarray(matrix).T
    if not matrix.any(axis=1).all():
        raise ValueError('Point pool cannot satisfy a registered safety witness')
    _packed, indices = np.unique(np.packbits(matrix, axis=1), axis=0, return_index=True)
    return [np.flatnonzero(matrix[index]).tolist() for index in indices]


def solve_cover(pool, groups, forbidden, minimum=19, maximum=20, seconds=30):
    model = cp_model.CpModel()
    chosen = [model.new_bool_var('point_' + str(index)) for index in range(len(pool))]
    model.add(chosen[0] == 1)
    model.add(sum(chosen) >= minimum)
    model.add(sum(chosen) <= maximum)
    for group in groups:
        model.add_bool_or([chosen[index] for index in group])
    for bad in forbidden:
        members = set(bad)
        model.add_bool_or([chosen[index].Not() if index in members else chosen[index] for index in range(1, len(pool))])
    distances = np.linalg.norm(pool[:, None] - pool[None, :], axis=2)
    conflicts = np.argwhere(np.triu((distances < 50) & (distances > 0), 1))
    for first, second in conflicts:
        model.add(chosen[int(first)] + chosen[int(second)] <= 1)
    model.minimize(sum(chosen))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    started = time.perf_counter()
    status = solver.solve(model)
    metadata = {'status': solver.status_name(status), 'wall_seconds': time.perf_counter() - started,
                'point_pool_size': len(pool), 'witness_groups': len(groups), 'minimum_spacing_constraints': len(conflicts),
                'objective_bound': solver.best_objective_bound, 'branches': solver.num_branches, 'conflicts': solver.num_conflicts}
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, metadata
    selected = [index for index, variable in enumerate(chosen) if solver.value(variable)]
    metadata['selected_count'] = len(selected)
    return selected, metadata


def hashes():
    provenance.frozen_integrity()
    combined = {}
    for relative in ('reports/q4_action_phase_v42/dev1_20260912_074048/protocol.json',
                     'reports/radial_dual_bound_v43/bound_20260912_081400/protocol.json'):
        for name, digest in provenance.load(ROOT / relative)['source_hashes'].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
                raise ValueError('Prior frozen source changed: ' + name)
            combined[name] = digest
    files = [*HERE.glob('*.py'), HERE / 'PROTOCOL.md', ROOT / 'research/network_synthesis_v8/synthesis.py',
             ROOT / 'research/annular_topology_v25/topology.py']
    return {**combined, **{str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a fresh D-drive geometry output')
    frozen = hashes()
    provenance.save(output / 'protocol.json', {'created_local': datetime.now().isoformat(), 'source_hashes': frozen,
                                              'maximum_rounds': 8, 'seconds_per_round': 30, 'source_counts': None,
                                              'policy_runs': 0, 'official_calls': 0, 'formal_calls': 0, 'new_holdout': False})
    for relative in frozen:
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    pool = pool_points()
    positions, headings = initial_witnesses()
    provenance.save(output / 'pool.json', pool.tolist())
    reference = geometry_price(dual_ring_network())
    provenance.save(output / 'reference.json', reference)
    rows, forbidden = [], []
    eligible = []
    for iteration in range(8):
        groups = visible_groups(pool, positions, headings)
        selected, row = solve_cover(pool, groups, forbidden)
        row['iteration'] = iteration
        row['source_witness_count'] = len(positions)
        if selected is None:
            rows.append(row)
            provenance.save(output / 'iterations' / f'{iteration:02d}.json', row)
            break
        points = pool[selected]
        coverage = certify(points)
        row.update({'selected_indices': selected, 'points': points.tolist(), 'complete': coverage.empty,
                    'remaining_area': coverage.area, 'remaining_geometry': coverage.region.wkt,
                    'geometry_price': geometry_price(points)})
        improvement = 100 * (1 - row['geometry_price']['proxy_cost_meters'] / reference['proxy_cost_meters'])
        row['proxy_improvement_percent'] = improvement
        if coverage.empty and improvement >= 1:
            eligible.append(row)
        if not coverage.empty:
            examples = counterexamples(coverage.region, points, limit=64, receiving_radius=999.9)
            row['counterexamples'] = [{'position': item[1].tolist(), 'heading': item[2].tolist()} for item in examples]
            if examples:
                positions = np.vstack((positions, [item[1] for item in examples]))
                headings = np.vstack((headings, [item[2] for item in examples]))
        forbidden.append(selected)
        rows.append(row)
        provenance.save(output / 'iterations' / f'{iteration:02d}.json', row)
        print(iteration, row['status'], len(selected), 'complete', coverage.empty, 'area', round(coverage.area, 6),
              'proxy', round(row['geometry_price']['proxy_cost_meters'], 3), flush=True)
        if eligible:
            break
    summary = {'iterations': len(rows), 'returned_candidates': sum('points' in row for row in rows),
               'continuous_complete': sum(row.get('complete', False) for row in rows), 'eligible_count': len(eligible),
               'eligible': eligible[0] if eligible else None, 'statuses': [row['status'] for row in rows],
               'source_hashes_unchanged': frozen == hashes(), 'new_holdout': False,
               'official_calls': 0, 'formal_calls': 0, 'policy_runs': 0,
               'scope': 'restricted radial point pool and time-capped candidate search, not global infeasibility'}
    provenance.save(output / 'summary.json', summary)
    print(json.dumps({key: value for key, value in summary.items() if key != 'eligible'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
