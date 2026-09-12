from __future__ import annotations

import math

import mpmath
import verify_bound as audit


def test_full_circle_probability_cannot_exceed_one():
    for radius in ('200', '400'):
        row = {'radius': radius, 'kind': 'oriented', 'cosine': '-1'}
        for receiver in (56, 57, 58):
            assert audit.point_fraction(row, receiver, 'radio') == 1


def test_disjoint_angular_caps_are_zero():
    row = {'radius': '1800', 'kind': 'oriented', 'cosine': '1'}
    assert audit.point_fraction(row, 1000, 'radio') == 0
    assert audit.point_fraction(row, 1870, 'radio') > 0


def test_closed_overlap_envelope_covers_endpoint_and_middle():
    row = {'radius': '1800', 'kind': 'oriented', 'cosine': '1'}
    column = {'kind': 'radio', 'lower': 2058, 'upper': 2060, 'origin': False}
    bound = audit.independent_coefficient(row, column)
    for receiver in (2058, math.sqrt(1800**2 + 1000**2), 2059, 2060):
        assert audit.point_fraction(row, receiver, 'radio') <= mpmath.mpf(bound)
