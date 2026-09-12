from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'research/layer_cardinality_v46'))
import run_layer_search as prior


INPUT = ROOT / 'reports/layer_cardinality_v46/geometry_20260912_090504'
previous = prior.previous


def source_hashes():
    combined = prior.source_hashes()
    manifest = previous.provenance.load(INPUT / 'protocol.json')
    for relative, expected in manifest['source_hashes'].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen v46 source changed: ' + relative)
    files = [*HERE.glob('*.py'), HERE / 'PROTOCOL.md',
             *sorted((ROOT / 'research/cover_milp_audit_v47').glob('*.py'))]
    return {**combined, **{str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}


def model_rows(pool, groups, cuts, minimum=19, maximum=20):
    rows = [{'kind': 'fixed_origin', 'members': [0], 'lower': 1, 'upper': 1},
            {'kind': 'cardinality', 'members': list(range(len(pool))), 'lower': minimum, 'upper': maximum}]
    rows.extend({'kind': 'cover', 'members': list(group), 'lower': 1, 'upper': max(1, len(group))} for group in groups)
    distances = np.linalg.norm(pool[:, None] - pool[None, :], axis=2)
    conflicts = np.argwhere(np.triu((distances < 50) & (distances > 0), 1))
    rows.extend({'kind': 'spacing', 'members': [int(first), int(second)], 'lower': 0, 'upper': 1}
                for first, second in conflicts)
    rows.extend({'kind': 'layer_cut', 'members': cut['point_indices'],
                 'lower': cut['required_points'], 'upper': len(cut['point_indices'])}
                for cut in cuts if cut['required_points'] > 0)
    return rows


def accepted_binary_candidate(values, rows, point_count):
    array = np.asarray(values, dtype=float)
    if array.shape != (point_count,) or not np.isfinite(array).all():
        raise ValueError('Invalid solver variable vector')
    rounded = np.rint(array)
    if np.max(np.abs(array - rounded)) > 1e-6 or not np.isin(rounded, [0, 1]).all():
        raise ValueError('Solver candidate is not binary')
    selected = set(int(index) for index in np.flatnonzero(rounded))
    for row in rows:
        total = len(selected.intersection(row['members']))
        if not row['lower'] <= total <= row['upper']:
            raise ValueError('Integer row check failed: ' + row['kind'])
    return sorted(selected)


def solve_milp(rows, point_count, seconds=30):
    row_indices, column_indices = [], []
    for row_index, row in enumerate(rows):
        if len(row['members']) != len(set(row['members'])):
            raise ValueError('Model rows require distinct indices')
        row_indices.extend([row_index] * len(row['members']))
        column_indices.extend(row['members'])
    matrix = csc_matrix((np.ones(len(row_indices)), (row_indices, column_indices)), shape=(len(rows), point_count))
    constraint = LinearConstraint(matrix, np.array([row['lower'] for row in rows]), np.array([row['upper'] for row in rows]))
    started = time.perf_counter()
    result = milp(np.ones(point_count), integrality=np.ones(point_count), bounds=Bounds(0, 1),
                  constraints=constraint, options={'time_limit': seconds, 'mip_rel_gap': 0.0})
    metadata = {
        'solver_status_code': int(result.status), 'solver_message': str(result.message),
        'wall_seconds': time.perf_counter() - started, 'model_rows': len(rows),
        'mip_node_count': getattr(result, 'mip_node_count', None),
        'objective_bound': getattr(result, 'mip_dual_bound', None), 'mip_gap': getattr(result, 'mip_gap', None),
    }
    selected = None
    if result.x is not None:
        selected = accepted_binary_candidate(result.x, rows, point_count)
    if result.status == 0:
        if selected is None:
            raise ValueError('Optimal status without an integer solution')
        metadata['status'] = 'OPTIMAL'
    elif result.status == 1:
        metadata['status'] = 'FEASIBLE_TIME_LIMIT' if selected is not None else 'UNKNOWN'
    elif result.status == 2:
        metadata['status'] = 'INFEASIBLE'
    else:
        metadata['status'] = 'SOLVER_ERROR'
    for key in ('objective_bound', 'mip_gap'):
        if metadata[key] is not None and not np.isfinite(metadata[key]):
            metadata[key] = None
    return selected, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a fresh D-drive output directory')
    load, save = previous.provenance.load, previous.provenance.save
    if load(INPUT / 'coverage_audit.json')['status'] != 'passed' or load(INPUT / 'cut_audit.json')['status'] != 'passed':
        raise ValueError('Prior independent audits must pass')
    frozen = source_hashes()
    input_files = ['pool.json', 'groups/00.json', 'proofs/00.json', 'interval_certificates.json', 'interval_combination.json']
    input_hashes = {relative: hashlib.sha256((INPUT / relative).read_bytes()).hexdigest() for relative in input_files}
    save(output / 'protocol.json', {
        'created_local': datetime.now().isoformat(), 'source_hashes': frozen,
        'input_batch': str(INPUT.relative_to(ROOT)), 'input_hashes': input_hashes,
        'backend': 'scipy.optimize.milp/HiGHS', 'time_limit': 30, 'mip_rel_gap': 0.0,
        'threads': 'installed_backend_default', 'minimum_points': 19, 'maximum_points': 20,
        'official_calls': 0, 'formal_calls': 0, 'policy_runs': 0, 'new_holdout': False,
    })
    for relative in frozen:
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    for relative in input_files:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(INPUT / relative, destination)
    pool = np.asarray(load(output / 'pool.json'))
    groups = load(output / 'groups' / '00.json')
    proof = load(output / 'proofs' / '00.json')
    reduced = [groups[index] for index in proof['kept_indices']]
    cuts = load(output / 'interval_certificates.json')
    rows = model_rows(pool, reduced, cuts)
    save(output / 'model_rows.json', rows)
    selected, row = solve_milp(rows, len(pool))
    row['iteration'] = 0
    eligible = None
    if selected is not None:
        if not all(set(selected).intersection(group) for group in groups):
            raise ValueError('Candidate misses an original group')
        points = pool[selected]
        coverage = previous.certify(points)
        reference_cost = previous.geometry_price(previous.dual_ring_network())
        price = previous.geometry_price(points)
        improvement = 100 * (1 - price['proxy_cost_meters'] / reference_cost['proxy_cost_meters'])
        row.update({'selected_indices': selected, 'selected_count': len(selected), 'points': points.tolist(),
                    'complete': coverage.empty, 'remaining_area': coverage.area, 'remaining_geometry': coverage.region.wkt,
                    'geometry_price': price, 'proxy_improvement_percent': improvement})
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
        'official_calls': 0, 'formal_calls': 0, 'policy_runs': 0, 'new_holdout': False,
        'scope': 'Same finite integer covering model, different solver mechanism, not a mission score',
    }
    save(output / 'summary.json', summary)
    print(json.dumps({key: value for key, value in summary.items() if key != 'eligible'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
