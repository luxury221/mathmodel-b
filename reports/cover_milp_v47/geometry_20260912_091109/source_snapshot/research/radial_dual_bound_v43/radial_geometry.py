from __future__ import annotations

import math
from fractions import Fraction
from functools import lru_cache

import mpmath
import numpy as np


RADII = ('20.1', '50', '100', '200', '400', '600', '800', '1000', '1200', '1400', '1600', '1800')
COSINES = ('1', '3/4', '1/2', '1/4', '1/20', '0', '-1/4', '-1/2', '-3/4', '-1')
RADIAL_STEP = 2
mpmath.iv.dps = 60


def up(value):
    return math.nextafter(float(value), math.inf)


def down(value):
    return math.nextafter(float(value), -math.inf)


def interval_fraction(value):
    value = Fraction(value)
    return mpmath.iv.mpf(value.numerator) / value.denominator


def endpoint_bounds(value):
    return down(float(value.a)), up(float(value.b))


@lru_cache(maxsize=None)
def acos_bounds(value):
    value = Fraction(value)
    if value >= 1:
        return 0.0, 0.0
    if value <= -1:
        return endpoint_bounds(mpmath.iv.pi)
    sine = mpmath.iv.sqrt(interval_fraction(1 - value * value))
    return endpoint_bounds(mpmath.iv.atan2(sine, interval_fraction(value)))


def point_range_angle(source_radius, receiver_radius, receiving_radius):
    source = Fraction(source_radius)
    receiver = Fraction(receiver_radius)
    reach = Fraction(receiving_radius)
    if receiver == 0:
        return math.pi if source <= reach else 0.0
    cosine = (receiver * receiver + source * source - reach * reach) / (2 * receiver * source)
    return math.acos(float(max(Fraction(-1), min(Fraction(1), cosine))))


def range_upper(source_radius, lower, upper, receiving_radius, rigorous=False):
    source = Fraction(source_radius)
    lower, upper = Fraction(lower), Fraction(upper)
    reach = Fraction(receiving_radius)
    if lower > source + reach or upper < source - reach:
        return 0.0
    critical_squared = source * source - reach * reach
    if critical_squared > 0 and lower * lower <= critical_squared <= upper * upper:
        if rigorous:
            return endpoint_bounds(mpmath.iv.atan2(interval_fraction(reach), mpmath.iv.sqrt(interval_fraction(critical_squared))))[1]
        return math.asin(float(reach / source))
    if lower == 0 and source <= reach:
        angle = mpmath.iv.pi if source < reach else mpmath.iv.pi / 2
        return endpoint_bounds(angle)[1] if rigorous else math.pi if source < reach else math.pi / 2
    receiver = lower if critical_squared <= lower * lower else upper
    cosine = (receiver * receiver + source * source - reach * reach) / (2 * receiver * source)
    return acos_bounds(cosine)[1] if rigorous else math.acos(float(max(Fraction(-1), min(Fraction(1), cosine))))


def heading_upper(source_radius, cosine, lower, upper, rigorous=False):
    cosine = Fraction(cosine)
    if cosine == 0:
        return endpoint_bounds(mpmath.iv.pi / 2)[1] if rigorous else math.pi / 2
    receiver = Fraction(upper if cosine > 0 else lower)
    if receiver == 0:
        return endpoint_bounds(mpmath.iv.pi)[1] if rigorous else math.pi
    threshold = Fraction(source_radius) * cosine / receiver
    return acos_bounds(threshold)[1] if rigorous else math.acos(float(max(Fraction(-1), min(Fraction(1), threshold))))


def arc_fraction(first_width, second_width, center, rigorous=False):
    if first_width <= 0 or second_width <= 0:
        return 0.0
    if not rigorous:
        angle = float(center)
        overlap = sum(max(0.0, min(first_width, angle + second_width + shift * 2 * math.pi)
                          - max(-first_width, angle - second_width + shift * 2 * math.pi)) for shift in (-1, 0, 1))
        return min(1.0, overlap / (2 * math.pi))
    center_low, center_high = center
    pi_low, pi_high = endpoint_bounds(mpmath.iv.pi)
    overlap = 0.0
    for shift in (-1, 0, 1):
        shift_low = down(2 * shift * (pi_high if shift < 0 else pi_low))
        shift_high = up(2 * shift * (pi_low if shift < 0 else pi_high))
        left = down(down(center_low - second_width) + shift_low)
        right = up(up(center_high + second_width) + shift_high)
        length = max(0.0, up(min(first_width, right) - max(-first_width, left)))
        overlap = up(overlap + length)
    return min(1.0, max(0.0, up(overlap / down(2 * pi_low))))


def rows_for(problem):
    rows = []
    for radius in RADII:
        rows.append({'radius': radius, 'kind': 'range' if problem == 'q3' else 'triple_range'})
        if problem == 'q4':
            rows.extend({'radius': radius, 'kind': 'oriented', 'cosine': cosine} for cosine in COSINES)
    return rows


def columns_for(step=RADIAL_STEP):
    columns = []
    for kind, limit, seconds in (('radio', 2800, 5), ('optical', 1820, 3)):
        columns.append({'kind': kind, 'lower': 0, 'upper': 0, 'origin': True, 'seconds': seconds})
        for lower in range(0, limit, step):
            columns.append({'kind': kind, 'lower': lower, 'upper': min(lower + step, limit), 'origin': False, 'seconds': seconds})
    return columns


def coefficient(row, column, rigorous=False):
    optical = column['kind'] == 'optical'
    reach = 20 if optical else 1000
    source = Fraction(row['radius'])
    if column['origin']:
        visible = source <= reach
        if not optical and row['kind'] == 'oriented':
            visible = visible and Fraction(row['cosine']) <= 0
        divisor = 3 if not optical and row['kind'] == 'triple_range' else 1
        value = int(visible) / divisor
        return up(value) if rigorous and value not in (0.0, 1.0) else value
    width = range_upper(row['radius'], column['lower'], column['upper'], reach, rigorous)
    if optical or row['kind'] != 'oriented':
        divisor = 3 if not optical and row['kind'] == 'triple_range' else 1
        if rigorous:
            pi_low = endpoint_bounds(mpmath.iv.pi)[0]
            return min(1.0, max(0.0, up(width / down(divisor * pi_low))))
        return width / (divisor * math.pi)
    beta = heading_upper(row['radius'], row['cosine'], column['lower'], column['upper'], rigorous)
    center = acos_bounds(Fraction(row['cosine'])) if rigorous else math.acos(float(Fraction(row['cosine'])))
    return arc_fraction(width, beta, center, rigorous)


def coefficient_matrix(rows, columns, rigorous=False):
    return np.asarray([[coefficient(row, column, rigorous) for row in rows] for column in columns])


def rational_budget_checks(matrix, numerators, columns, denominator=1000000):
    if denominator <= 0 or any(int(numerator) != numerator or numerator < 0 for numerator in numerators):
        raise ValueError('Certificate weights must be nonnegative rationals')
    if np.shape(matrix) != (len(columns), len(numerators)) or not np.isfinite(matrix).all():
        raise ValueError('Invalid certificate coefficient matrix')
    if np.any(matrix < 0) or np.any(matrix > 1):
        raise ValueError('Coverage proportions must lie between zero and one')
    weights = [Fraction(int(numerator), denominator) for numerator in numerators]
    checks = []
    for values, column in zip(matrix, columns):
        total = sum((Fraction.from_float(float(value)) * weight for value, weight in zip(values, weights)), Fraction())
        checks.append(total / column['seconds'])
    return max(checks), checks


def integer_time_bound(numerators, denominator=1000000):
    total = Fraction(sum(int(numerator) for numerator in numerators), denominator)
    return total, -(-total.numerator // total.denominator)


def movement_lower(problem):
    if problem == 'q4':
        length = (1 + mpmath.iv.sqrt(3) + 7 * mpmath.iv.pi / 6) * 1780
    else:
        length = mpmath.iv.pi * (1800**2 - 1000**2) / 2000
    return {'meters': endpoint_bounds(length)[0], 'seconds': endpoint_bounds(length / 5)[0]}


def mission_lower(problem, count, absent_seconds):
    if count == 16:
        return {'source_count': count, 'absence_bound_applicable': False, 'total_lower_seconds': 5 * count,
                'seconds_per_source_lower': 5,
                'scope': 'only successful optical clearance; absence and inspection not applicable'}
    if not 10 <= count <= 15:
        raise ValueError('Only legal competition source counts are allowed')
    move = movement_lower(problem)['seconds']
    total = down(down(move + (20 - count) * absent_seconds) + 5 * count)
    return {'source_count': count, 'absence_bound_applicable': True, 'total_lower_seconds': total,
            'seconds_per_source_lower': down(total / count), 'movement_lower_seconds': move,
            'absent_channel_operation_lower_seconds': (20 - count) * absent_seconds,
            'successful_clear_lower_seconds': 5 * count, 'switch_lower_seconds': 0}
