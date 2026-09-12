from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from datetime import datetime
from pathlib import Path

import mpmath
import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from radial_geometry import coefficient_matrix, columns_for, integer_time_bound, mission_lower, movement_lower, rational_budget_checks, rows_for


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFESTS = ('reports/scan_validation_v13/validation_20260911_235435/selection.json',
             'reports/frozen_practice_v33/batch_20260912_040922/protocol.json',
             'reports/paired_scan_commitment_v38/dev1_20260912_060916/protocol.json',
             'reports/coverage_probe_v39/dev1_20260912_064358/protocol.json',
             'reports/feasible_segment_bound_v40/bound_20260912_070654/protocol.json',
             'reports/q3_action_scheduler_v41/dev1_20260912_071723/protocol.json',
             'reports/q4_action_phase_v42/dev1_20260912_074048/protocol.json',
             'reports/q4_action_phase_v42/dev2_20260912_074431/protocol.json')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def frozen_integrity():
    counts = {}
    for relative in MANIFESTS:
        hashes = load(ROOT / relative)['source_hashes']
        for name, digest in hashes.items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
                raise ValueError('Frozen source changed: ' + name)
        counts[relative] = len(hashes)
    return counts


def source_hashes():
    frozen_integrity()
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [*HERE.glob('*.py'), HERE / 'PROTOCOL.md']}


def solve_certificate(problem, output, columns):
    started = time.perf_counter()
    rows = rows_for(problem)
    matrix = coefficient_matrix(rows, columns)
    costs = np.asarray([column['seconds'] for column in columns], dtype=float)
    solution = linprog(-np.ones(len(rows)), A_ub=csr_matrix(matrix), b_ub=costs,
                       bounds=(0, None), method='highs')
    if not solution.success or not np.isfinite(solution.x).all():
        raise ValueError('Dual proposal failed: ' + solution.message)
    numerators = np.floor(np.maximum(0.0, solution.x) * 1000000).astype(np.int64)
    active = np.flatnonzero(numerators)
    active_rows = [rows[index] for index in active]
    active_numerators = numerators[active]
    certified = coefficient_matrix(active_rows, columns, rigorous=True)
    initial_ratio, _checks = rational_budget_checks(certified, active_numerators, columns)
    scale = 1.0
    if initial_ratio > 1:
        scale = math.nextafter(1 / float(initial_ratio), 0.0) * (1 - 1e-9)
        active_numerators = np.floor(active_numerators * scale).astype(np.int64)
    maximum, checks = rational_budget_checks(certified, active_numerators, columns)
    if maximum > 1:
        raise ValueError('Interval-envelope rational dual certificate is infeasible')
    rational, integer_bound = integer_time_bound(active_numerators)
    np.save(output / (problem + '_upper_coefficients.npy'), certified, allow_pickle=False)
    matrix_file = output / (problem + '_upper_coefficients.npy')
    result = {'problem': problem, 'rows': rows, 'active_indices': active.tolist(), 'active_rows': active_rows,
              'weight_numerators': active_numerators.tolist(), 'weight_denominator': 1000000,
              'proposed_dual_value': -float(solution.fun), 'weights_scale': scale,
              'verified_dual_value': float(rational), 'rational_lower_numerator': rational.numerator,
              'rational_lower_denominator': rational.denominator, 'integer_absent_channel_seconds_lower': integer_bound,
              'maximum_budget_ratio': float(maximum), 'exact_budget_numerator': maximum.numerator,
              'exact_budget_denominator': maximum.denominator, 'coefficients_file': matrix_file.name,
              'coefficients_sha256': hashlib.sha256(matrix_file.read_bytes()).hexdigest(),
              'binding_cells': [{'column': index, 'budget_ratio': float(checks[index]), **columns[index]}
                                for index in sorted(range(len(checks)), key=lambda index: checks[index], reverse=True)[:16]],
              'movement_lower': movement_lower(problem),
              'by_source_count': [mission_lower(problem, count, integer_bound) for count in range(10, 17)],
              'solver_status': solution.message, 'optimality_scope': 'only finite radial-envelope LP, not the search algorithm',
              'wall_seconds': time.perf_counter() - started}
    save(output / (problem + '_certificate.json'), result)
    print(problem, 'dual', result['verified_dual_value'], 'integer_seconds', integer_bound,
          'active_rows', len(active), 'max_budget', float(maximum), 'wall', round(result['wall_seconds'], 2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a fresh D-drive output')
    hashes = source_hashes()
    save(output / 'protocol.json', {'created_local': datetime.now().isoformat(), 'source_hashes': hashes,
                                   'frozen_manifests': frozen_integrity(), 'practice_calls': 0, 'formal_calls': 0,
                                   'policy_runs': 0, 'new_holdout': False,
                                   'libraries': {'numpy': np.__version__, 'scipy': scipy.__version__, 'mpmath': mpmath.__version__}})
    for relative in hashes:
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    columns = columns_for()
    save(output / 'columns.json', columns)
    results = {problem: solve_certificate(problem, output, columns) for problem in ('q3', 'q4')}
    official_path = ROOT / 'reports/frozen_practice_v33/batch_20260912_040922/runs.json'
    official_rows = load(official_path)
    diagnostic = []
    for row in official_rows:
        if row['client_status'] != 'completed' or not row['all_cleared']:
            raise ValueError('The official reference batch is incomplete')
        problem = row['problem']
        bound = mission_lower(problem, row['source_count'], results[problem]['integer_absent_channel_seconds_lower'])
        diagnostic.append({'problem': problem, 'attempt': row['attempt'], 'source_count': row['source_count'],
                           'observed_seconds_per_source': row['seconds_per_source'], **bound})
    save(output / 'official_count_diagnostic.json', {'basis': 'public completed source counts only; no new runs or hidden locations',
                                                   'input_sha256': hashlib.sha256(official_path.read_bytes()).hexdigest(), 'cases': diagnostic})
    save(output / 'integrity.json', {'unchanged': hashes == source_hashes(), 'frozen_manifests': frozen_integrity()})
    print('output', output, flush=True)


if __name__ == '__main__':
    main()
