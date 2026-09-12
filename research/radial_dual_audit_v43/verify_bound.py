from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from fractions import Fraction
from pathlib import Path

import mpmath
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
mpmath.iv.dps = 60
mpmath.mp.dps = 80


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def ivalue(value):
    rational = Fraction(value)
    return mpmath.iv.mpf(rational.numerator) / rational.denominator


def upper(value):
    return math.nextafter(float(value.b), math.inf)


def lower(value):
    return math.nextafter(float(value.a), -math.inf)


def cap_angle(cosine):
    cosine = Fraction(cosine)
    if cosine >= 1:
        return mpmath.iv.mpf(0)
    if cosine <= -1:
        return mpmath.iv.pi
    return mpmath.iv.atan2(mpmath.iv.sqrt(ivalue(1 - cosine * cosine)), ivalue(cosine))


def distance_cap(radius, column, reach):
    radius, reach = Fraction(radius), Fraction(reach)
    start, end = Fraction(column['lower']), Fraction(column['upper'])
    if end < radius - reach or start > radius + reach:
        return mpmath.iv.mpf(0)
    critical = radius * radius - reach * reach
    if critical > 0 and start * start <= critical <= end * end:
        return mpmath.iv.atan2(ivalue(reach), mpmath.iv.sqrt(ivalue(critical)))
    if start == 0 and radius <= reach:
        return mpmath.iv.pi if radius < reach else mpmath.iv.pi / 2
    receiver = start if start * start >= critical else end
    return cap_angle((radius * radius + receiver * receiver - reach * reach) / (2 * radius * receiver))


def independent_coefficient(row, column):
    radius = Fraction(row['radius'])
    optical = column['kind'] == 'optical'
    reach = 20 if optical else 1000
    if column['origin']:
        active = radius <= reach
        if row['kind'] == 'oriented' and not optical:
            active = active and Fraction(row['cosine']) <= 0
        divisor = 3 if row['kind'] == 'triple_range' and not optical else 1
        return max(0.0, min(1.0, upper(ivalue(Fraction(int(active), divisor)))))
    first = distance_cap(radius, column, reach)
    if row['kind'] != 'oriented' or optical:
        divisor = 3 if row['kind'] == 'triple_range' and not optical else 1
        return max(0.0, min(1.0, upper(first / (divisor * mpmath.iv.pi))))
    cosine = Fraction(row['cosine'])
    receiver = Fraction(column['upper'] if cosine > 0 else column['lower'])
    if cosine == 0:
        second = mpmath.iv.pi / 2
    elif receiver == 0:
        second = mpmath.iv.pi
    else:
        second = cap_angle(radius * cosine / receiver)
    separation = cap_angle(cosine)
    straight = max(0.0, min(upper(2 * first), upper(2 * second), upper(first + second - separation)))
    wrapped = max(0.0, upper(first + second + separation - 2 * mpmath.iv.pi))
    fraction = math.nextafter((straight + wrapped) / lower(2 * mpmath.iv.pi), math.inf)
    return min(1.0, max(0.0, fraction))


def point_fraction(row, receiver, kind):
    radius = mpmath.mpf(row['radius'])
    receiver = mpmath.mpf(str(receiver))
    optical = kind == 'optical'
    reach = 20 if optical else 1000
    if receiver == 0:
        value = mpmath.mpf(int(radius <= reach))
        if row['kind'] == 'oriented' and not optical and Fraction(row['cosine']) > 0:
            value = mpmath.mpf(0)
    else:
        quotient = (radius * radius + receiver * receiver - reach * reach) / (2 * radius * receiver)
        alpha = mpmath.acos(max(-1, min(1, quotient)))
        value = alpha / mpmath.pi
        if row['kind'] == 'oriented' and not optical:
            cosine = Fraction(row['cosine'])
            exact_cosine = mpmath.mpf(cosine.numerator) / cosine.denominator
            beta = mpmath.acos(max(-1, min(1, radius * exact_cosine / receiver)))
            separation = mpmath.acos(exact_cosine)
            overlap = max(0, min(2 * alpha, 2 * beta, alpha + beta - separation))
            overlap += max(0, alpha + beta + separation - 2 * mpmath.pi)
            value = overlap / (2 * mpmath.pi)
    value = min(mpmath.mpf(1), max(mpmath.mpf(0), value))
    if row['kind'] == 'triple_range' and not optical:
        value /= 3
    return value


def verify_partition(columns):
    if len(columns) != 2312:
        raise ValueError('Unexpected radial partition size')
    for kind, end, cost in (('radio', 2800, 5), ('optical', 1820, 3)):
        chosen = [column for column in columns if column['kind'] == kind]
        if chosen[0] != {'kind': kind, 'lower': 0, 'upper': 0, 'origin': True, 'seconds': cost}:
            raise ValueError('Missing origin atom')
        if chosen[1]['lower'] != 0 or chosen[-1]['upper'] != end:
            raise ValueError('Incomplete continuous radial domain')
        if any(column['origin'] or column['upper'] - column['lower'] != 2 or column['seconds'] != cost for column in chosen[1:]):
            raise ValueError('Changed radial cell definition')
        if any(first['upper'] != second['lower'] for first, second in zip(chosen[1:], chosen[2:])):
            raise ValueError('Gap in receiver radius coverage')


def verify_certificate(problem, directory, output, columns):
    certificate = load(directory / (problem + '_certificate.json'))
    denominator = certificate['weight_denominator']
    if denominator != 1000000 or any(not isinstance(value, int) or value < 0 for value in certificate['weight_numerators']):
        raise ValueError('Invalid rational weights')
    rows = certificate['active_rows']
    weights = [Fraction(value, denominator) for value in certificate['weight_numerators']]
    if len(rows) != len(weights):
        raise ValueError('Weight count differs from row count')
    path = directory / certificate['coefficients_file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != certificate['coefficients_sha256']:
        raise ValueError('Stored coefficient matrix digest differs')
    stored = np.load(path, allow_pickle=False)
    if stored.shape != (len(columns), len(rows)) or not np.isfinite(stored).all():
        raise ValueError('Invalid stored coefficient shape')
    fresh = np.array([[independent_coefficient(row, column) for row in rows] for column in columns])
    maximum = Fraction()
    maximum_stored = Fraction()
    point_checks = 0
    for index, column in enumerate(columns):
        total = sum((Fraction.from_float(float(value)) * weight for value, weight in zip(fresh[index], weights)), Fraction())
        saved_total = sum((Fraction.from_float(float(value)) * weight for value, weight in zip(stored[index], weights)), Fraction())
        maximum = max(maximum, total / column['seconds'])
        maximum_stored = max(maximum_stored, saved_total / column['seconds'])
        if total > column['seconds'] or saved_total > column['seconds']:
            raise ValueError('A continuous receiver cell exceeds the exact dual budget')
        if column['origin'] or index % 29 == 0:
            radii = [0] if column['origin'] else [max(0.00001, column['lower']), (column['lower'] + column['upper']) / 2, column['upper']]
            for radius in radii:
                for row_index, row in enumerate(rows):
                    value = point_fraction(row, radius, column['kind'])
                    if value > mpmath.mpf(float(fresh[index, row_index])):
                        raise ValueError('Direct 80-digit receiver fraction exceeds its interval envelope')
                    point_checks += 1
    total_weight = sum(weights, Fraction())
    integer_bound = -(-total_weight.numerator // total_weight.denominator)
    if total_weight != Fraction(certificate['rational_lower_numerator'], certificate['rational_lower_denominator']):
        raise ValueError('Reported rational lower bound differs from weights')
    if integer_bound != certificate['integer_absent_channel_seconds_lower']:
        raise ValueError('Integer operation bound differs')
    baseline_radii = [0, *([1125] * 6)] if problem == 'q3' else [0, *([1000] * 8), *([1870] * 12)]
    minimum_baseline_cover = math.inf
    for row in rows:
        cover = sum(point_fraction(row, radius, 'radio') for radius in baseline_radii)
        if problem == 'q4':
            cover += point_fraction(row, 0, 'optical')
        if cover < 1 - mpmath.mpf('1e-70'):
            raise ValueError('A necessary inequality rejects an existing complete network')
        minimum_baseline_cover = min(minimum_baseline_cover, float(cover))
    known_query_cost = 35 if problem == 'q3' else 108
    if integer_bound > known_query_cost:
        raise ValueError('The lower bound exceeds a known feasible query upper bound')
    movement = (1 + mpmath.sqrt(3) + 7 * mpmath.pi / 6) * 1780 / 5 if problem == 'q4' else mpmath.pi * (1800**2 - 1000**2) / 10000
    if mpmath.mpf(certificate['movement_lower']['seconds']) > movement:
        raise ValueError('Movement lower bound was rounded upward')
    for row in certificate['by_source_count']:
        count = row['source_count']
        expected = movement + (20 - count) * integer_bound + 5 * count if count < 16 else mpmath.mpf(80)
        if mpmath.mpf(row['total_lower_seconds']) > expected:
            raise ValueError('Combined mission lower bound is too high')
        if row['absence_bound_applicable'] != (count < 16):
            raise ValueError('The 16-source stopping exception was ignored')
    np.save(output / (problem + '_independent_upper.npy'), fresh, allow_pickle=False)
    return {'problem': problem, 'active_rows': len(rows), 'continuous_cell_checks': len(columns),
            'direct_high_precision_point_checks': point_checks, 'maximum_independent_budget_ratio': float(maximum),
            'maximum_original_budget_ratio': float(maximum_stored), 'exact_budget_numerator': maximum.numerator,
            'exact_budget_denominator': maximum.denominator, 'dual_value': float(total_weight),
            'integer_absent_channel_seconds_lower': integer_bound, 'known_feasible_query_cost_upper': known_query_cost,
            'minimum_known_network_integrated_coverage': minimum_baseline_cover,
            'maximum_coefficient_difference': float(np.max(np.abs(fresh - stored)))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a fresh D-drive independent audit output')
    output.mkdir(parents=True)
    directory = args.batch.resolve()
    protocol = load(directory / 'protocol.json')
    for relative, digest in protocol['source_hashes'].items():
        for path in (ROOT / relative, directory / 'source_snapshot' / relative):
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Frozen producer source or snapshot changed')
    for relative in protocol['frozen_manifests']:
        for name, digest in load(ROOT / relative)['source_hashes'].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
                raise ValueError('Historical frozen code changed')
    columns = load(directory / 'columns.json')
    verify_partition(columns)
    reports = [verify_certificate(problem, directory, output, columns) for problem in ('q3', 'q4')]
    counts = load(directory / 'official_count_diagnostic.json')
    official_path = ROOT / 'reports/frozen_practice_v33/batch_20260912_040922/runs.json'
    if counts['input_sha256'] != hashlib.sha256(official_path.read_bytes()).hexdigest():
        raise ValueError('Official public result input changed')
    official = load(official_path)
    by_problem = {}
    for problem, target in (('q3', 180), ('q4', 300)):
        cases = [row for row in counts['cases'] if row['problem'] == problem]
        source_rows = [row for row in official if row['problem'] == problem]
        if len(cases) != 5 or [row['source_count'] for row in cases] != [row['source_count'] for row in source_rows]:
            raise ValueError('Count diagnostic differs from the completed official batch')
        if any(row['seconds_per_source_lower'] > row['observed_seconds_per_source'] for row in cases):
            raise ValueError('Lower bound exceeds an observed feasible complete mission')
        bound = math.nextafter(statistics.mean(row['seconds_per_source_lower'] for row in cases), -math.inf)
        by_problem[problem] = {'source_counts': [row['source_count'] for row in cases],
                               'batch_count_conditioned_lower_mean': bound,
                               'observed_official_mean': statistics.mean(row['observed_seconds_per_source'] for row in cases),
                               'target': target, 'bound_excludes_target_for_this_count_mix': bound >= target}
    result = {'status': 'passed', 'certificates': reports, 'official_count_diagnostic': by_problem,
              'source_hashes': {str(Path(__file__).relative_to(ROOT)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              'producer_protocol_sha256': hashlib.sha256((directory / 'protocol.json').read_bytes()).hexdigest(),
              'new_policy_runs': 0, 'new_holdout': False, 'official_calls': 0, 'formal_calls': 0,
              'scope': 'necessary guarantee-search bounds, not achieved performance or a complete mission optimizer'}
    write(output / 'analysis.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
