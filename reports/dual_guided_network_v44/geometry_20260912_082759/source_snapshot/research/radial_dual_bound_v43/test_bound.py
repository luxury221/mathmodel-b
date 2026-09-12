from __future__ import annotations

import math
import sys
from fractions import Fraction
from pathlib import Path

import mpmath
import numpy as np
import pytest

from radial_geometry import arc_fraction, coefficient, columns_for, heading_upper, integer_time_bound, mission_lower, movement_lower, range_upper, rational_budget_checks


def sampled_fraction(row, receiver, optical=False, angles=40000):
    phase = (np.arange(angles) + 0.5) * 2 * np.pi / angles
    radius = float(Fraction(row['radius']))
    positions = radius * np.column_stack((np.cos(phase), np.sin(phase)))
    vectors = np.array([receiver, 0]) - positions
    visible = np.linalg.norm(vectors, axis=1) <= (20 if optical else 1000)
    if not optical and row['kind'] == 'oriented':
        offset = math.acos(float(Fraction(row['cosine'])))
        headings = np.column_stack((np.cos(phase + offset), np.sin(phase + offset)))
        visible &= np.sum(vectors * headings, axis=1) >= 0
    divisor = 3 if not optical and row['kind'] == 'triple_range' else 1
    return float(visible.mean()) / divisor


def test_arc_intersection_handles_wrap_and_full_circles():
    assert math.isclose(arc_fraction(math.pi, math.pi / 2, math.pi), 0.5)
    assert math.isclose(arc_fraction(math.pi / 2, math.pi / 2, 0), 0.5)
    assert arc_fraction(math.pi / 4, math.pi / 4, math.pi) == 0
    assert arc_fraction(math.pi, math.pi, 0) == 1


def test_origin_is_separate_from_tangent_positive_radius_limit():
    row = {'radius': '500', 'kind': 'oriented', 'cosine': '0'}
    origin = {'kind': 'radio', 'lower': 0, 'upper': 0, 'origin': True}
    first_cell = {'kind': 'radio', 'lower': 0, 'upper': 2, 'origin': False}
    assert coefficient(row, origin, True) == 1
    assert 0.5 <= coefficient(row, first_cell, True) < 0.50000001
    row['cosine'] = '1'
    assert coefficient(row, origin, True) == 0


def test_range_cell_bound_includes_interior_radial_maximum():
    value = range_upper('1800', 1400, 1600, 1000, True)
    assert value >= math.asin(1000 / 1800)
    assert value < math.asin(1000 / 1800) + 1e-12
    assert range_upper('1800', 0, 799, 1000, True) == 0
    assert range_upper('1800', 2801, 3000, 1000, True) == 0


def test_heading_envelope_selects_correct_radial_endpoint():
    assert heading_upper('1000', '1/2', 1000, 1200, True) >= math.acos(500 / 1200)
    assert heading_upper('1000', '-1/2', 1000, 1200, True) >= math.acos(-500 / 1000)
    assert heading_upper('1000', '0', 0, 2, True) >= math.pi / 2


@pytest.mark.parametrize('kind,cosine', [('range', None), ('triple_range', None), ('oriented', '1'), ('oriented', '1/20'), ('oriented', '-1/2')])
def test_continuous_cell_envelopes_dominate_direct_visibility_samples(kind, cosine):
    for radius in ('20.1', '800', '1800'):
        row = {'radius': radius, 'kind': kind}
        if cosine is not None:
            row['cosine'] = cosine
        for lower, upper in ((0, 2), (798, 802), (1798, 1802), (2058, 2060)):
            for optical in (False, True):
                column = {'kind': 'optical' if optical else 'radio', 'lower': lower, 'upper': upper, 'origin': False}
                bound = coefficient(row, column, True)
                for receiver in np.linspace(max(lower, 0.001), upper, 5):
                    observed = sampled_fraction(row, receiver, optical)
                    assert observed <= bound + 3 / 40000


def test_optical_radius_is_twenty_not_the_radio_radius():
    row = {'radius': '50', 'kind': 'oriented', 'cosine': '1'}
    origin = {'kind': 'optical', 'lower': 0, 'upper': 0, 'origin': True}
    assert coefficient(row, origin, True) == 0
    assert coefficient(row, {**origin, 'origin': False, 'lower': 49, 'upper': 51}, True) > 0


def test_rational_budgets_detect_invalid_or_infeasible_weights():
    columns = [{'seconds': 5}, {'seconds': 3}]
    matrix = np.array([[1.0], [0.5]])
    assert rational_budget_checks(matrix, [5000000], columns)[0] == 1
    assert rational_budget_checks(matrix, [5000001], columns)[0] > 1
    with pytest.raises(ValueError, match='nonnegative'):
        rational_budget_checks(matrix, [-1], columns)


def test_integer_operation_bound_and_maximum_count_exception():
    rational, integer = integer_time_bound([5000001])
    assert rational == Fraction(5000001, 1000000) and integer == 6
    assert integer_time_bound([5000000])[1] == 5
    assert not mission_lower('q4', 16, 1000)['absence_bound_applicable']
    assert mission_lower('q4', 16, 1000)['seconds_per_source_lower'] == 5
    with pytest.raises(ValueError, match='legal'):
        mission_lower('q4', 17, 80)


def test_movement_constants_are_rounded_below_high_precision_values():
    mpmath.mp.dps = 80
    expected = (1 + mpmath.sqrt(3) + 7 * mpmath.pi / 6) * 1780 / 5
    assert mpmath.mpf(movement_lower('q4')['seconds']) <= expected
    assert expected - movement_lower('q4')['seconds'] < mpmath.mpf('1e-9')


def test_radial_cells_cover_the_entire_relevant_space_without_gaps():
    columns = columns_for()
    for kind, limit in (('radio', 2800), ('optical', 1820)):
        chosen = [column for column in columns if column['kind'] == kind]
        assert chosen[0]['origin'] and chosen[1]['lower'] == 0
        assert chosen[-1]['upper'] == limit
        assert all(first['upper'] == second['lower'] for first, second in zip(chosen[1:], chosen[2:]))


def test_optical_channels_do_not_imply_receiver_switches():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / 'research/offline_validation'))
    from offline_benchmark import OfflineRuleWorld, Source
    sources = [Source(channel, (0.0, 0.0), 1000.0, 0.0 if channel == 1 else None) for channel in range(1, 17)]
    world = OfflineRuleWorld(sources, 471842503)
    for channel in range(1, 17):
        assert world.clear(np.zeros(2), channel)['result'] == 'success'
    assert world.stats['switch_count'] == 0 and world.receiver_channel == 1
    assert world.virtual_seconds == 80 and not world.remaining
    assert mission_lower('q4', 16, 1000)['total_lower_seconds'] == world.virtual_seconds
