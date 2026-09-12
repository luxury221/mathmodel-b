from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / 'research/joint_search_v5'), str(ROOT / 'research/offline_validation')]
from check_layouts import certify
from geometry import dual_ring_network, open_route, ring, route_length
from strong_routes import StrongRouter


def test_small_open_tours_match_exhaustive_search():
    generator = np.random.default_rng(252511101)
    router = StrongRouter()
    for count in range(2, 8):
        points = generator.uniform(-100, 100, (count, 2))
        start = generator.uniform(-100, 100, 2)
        optimum = min(route_length(points[list(order)], start) for order in itertools.permutations(range(count)))
        assert abs(route_length(router(points, start), start) - optimum) < 1e-8


@pytest.mark.parametrize('mode', ['multistart', 'guided'])
def test_larger_tours_are_complete_no_worse_and_repeatable(mode):
    generator = np.random.default_rng(252511102)
    points = generator.uniform(-1800, 1800, (20, 2))
    start = np.array([100, -200])
    first, second = StrongRouter(mode), StrongRouter(mode)
    route = first(points, start)
    assert np.array_equal(route, second(points, start))
    assert sorted(map(tuple, route)) == sorted(map(tuple, points))
    assert route_length(route, start) <= route_length(open_route(points, start), start) + 1e-7
    copy = first(points, start)
    copy[0] = [9999, 9999]
    assert np.array_equal(first(points, start), route)


def test_duplicates_and_empty_inputs_preserve_the_input_multiset():
    router = StrongRouter()
    assert router([]).shape == (0, 2)
    points = np.array([[1, 2], [1, 2], [3, 4]])
    assert sorted(map(tuple, router(points))) == sorted(map(tuple, points))


def test_new_layout_requires_complete_origin_evidence():
    points = np.vstack(([0, 0], ring(1011.5, 7), ring(1860, 13, 15)))
    assert certify(points).empty
    assert not certify(points, origin_radio=False).empty
    assert certify(dual_ring_network()).empty
