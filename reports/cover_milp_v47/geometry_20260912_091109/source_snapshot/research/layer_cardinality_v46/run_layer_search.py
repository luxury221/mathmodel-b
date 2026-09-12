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
sys.path.insert(0, str(ROOT / 'research/cover_subsumption_v45'))
import run_search as reduced_stage
from reduction import reduce_cover
from layer_bounds import disjoint_interval_lower, generate_interval_bounds


previous = reduced_stage.previous
np = previous.np
cp_model = previous.cp_model


def source_hashes():
    combined = reduced_stage.source_hashes()
    old_manifest = previous.provenance.load(ROOT / 'reports/cover_subsumption_v45/geometry_20260912_085728/protocol.json')
    for relative, expected in old_manifest['source_hashes'].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen v45 source changed: ' + relative)
    files = [*HERE.glob('*.py'), HERE / 'PROTOCOL.md',
             *sorted((ROOT / 'research/layer_cardinality_audit_v46').glob('*.py'))]
    return {**combined, **{str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}


def solve_with_cuts(pool, groups, cuts, minimum=19, maximum=20, seconds=30):
    model = cp_model.CpModel()
    chosen = [model.new_bool_var('point_' + str(index)) for index in range(len(pool))]
    model.add(chosen[0] == 1)
    model.add(sum(chosen) >= minimum)
    model.add(sum(chosen) <= maximum)
    for group in groups:
        model.add_bool_or([chosen[index] for index in group])
    distances = np.linalg.norm(pool[:, None] - pool[None, :], axis=2)
    conflicts = np.argwhere(np.triu((distances < 50) & (distances > 0), 1))
    for first, second in conflicts:
        model.add(chosen[int(first)] + chosen[int(second)] <= 1)
    active_cuts = [cut for cut in cuts if cut['required_points'] > 0]
    for cut in active_cuts:
        model.add(sum(chosen[index] for index in cut['point_indices']) >= cut['required_points'])
    model.minimize(sum(chosen))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    started = time.perf_counter()
    status = solver.solve(model)
    row = {
        'status': solver.status_name(status), 'wall_seconds': time.perf_counter() - started,
        'point_pool_size': len(pool), 'witness_groups': len(groups),
        'minimum_spacing_constraints': len(conflicts), 'added_cardinality_cuts': len(active_cuts),
        'objective_bound': solver.best_objective_bound, 'branches': solver.num_branches,
        'conflicts': solver.num_conflicts, 'model_sha256': hashlib.sha256(str(model.proto).encode()).hexdigest(),
    }
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, row
    selected = [index for index, variable in enumerate(chosen) if solver.value(variable)]
    row['selected_count'] = len(selected)
    return selected, row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a fresh D-drive output directory')
    frozen = source_hashes()
    save = previous.provenance.save
    save(output / 'protocol.json', {
        'created_local': datetime.now().isoformat(), 'source_hashes': frozen,
        'maximum_rounds': 1, 'seconds_per_round': 30, 'solver_workers': 1, 'solver_seed': 0,
        'minimum_points': 19, 'maximum_points': 20, 'forced_true': [0],
        'interval_count': 45, 'lp_seconds_per_interval': 3,
        'policy_runs': 0, 'new_holdout': False, 'official_calls': 0, 'formal_calls': 0,
    })
    for relative in frozen:
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    pool = previous.pool_points()
    positions, headings = previous.initial_witnesses()
    groups = previous.visible_groups(pool, positions, headings)
    proof = reduce_cover(groups, len(pool))
    reduced = [groups[index] for index in proof['kept_indices']]
    save(output / 'pool.json', pool.tolist())
    save(output / 'groups' / '00.json', groups)
    save(output / 'proofs' / '00.json', proof)
    reference = previous.geometry_price(previous.dual_ring_network())
    save(output / 'reference.json', reference)
    cuts = generate_interval_bounds(groups)
    save(output / 'interval_certificates.json', cuts)
    global_lower = disjoint_interval_lower(cuts)
    save(output / 'interval_combination.json', global_lower)
    print(json.dumps({'intervals': len(cuts), 'positive_cuts': sum(cut['required_points'] > 0 for cut in cuts),
                      'global_integer_lower': global_lower['with_forced_origin_lower'],
                      'lp_seconds_total': sum(cut['lp_seconds'] for cut in cuts)}), flush=True)
    selected, row = solve_with_cuts(pool, reduced, cuts)
    row.update({'iteration': 0, 'source_witness_count': len(positions),
                'original_groups': len(groups), 'retained_groups': len(reduced)})
    eligible = None
    if selected is not None:
        points = pool[selected]
        coverage = previous.certify(points)
        row.update({'selected_indices': selected, 'points': points.tolist(), 'complete': coverage.empty,
                    'remaining_area': coverage.area, 'remaining_geometry': coverage.region.wkt,
                    'geometry_price': previous.geometry_price(points)})
        improvement = 100 * (1 - row['geometry_price']['proxy_cost_meters'] / reference['proxy_cost_meters'])
        row['proxy_improvement_percent'] = improvement
        if coverage.empty and improvement >= 1:
            eligible = row
        if not coverage.empty:
            examples = previous.counterexamples(coverage.region, points, limit=64, receiving_radius=999.9)
            row['counterexamples'] = [{'position': item[1].tolist(), 'heading': item[2].tolist()} for item in examples]
    save(output / 'iterations' / '00.json', row)
    summary = {
        'iterations': 1, 'statuses': [row['status']], 'returned_candidates': int(selected is not None),
        'continuous_complete': int(row.get('complete', False)), 'eligible_count': int(eligible is not None),
        'eligible': eligible, 'source_hashes_unchanged': frozen == source_hashes(),
        'global_integer_lower': global_lower['with_forced_origin_lower'],
        'new_holdout': False, 'policy_runs': 0, 'official_calls': 0, 'formal_calls': 0,
        'scope': 'Exact rational valid cuts on a finite point pool, not a complete-mission score',
    }
    save(output / 'summary.json', summary)
    print(json.dumps({key: value for key, value in summary.items() if key != 'eligible'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
