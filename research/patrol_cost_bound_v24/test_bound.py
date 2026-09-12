from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from route_bound import neighborhood_costs, open_path_lower_bound


def exact_path(costs):
    return min(sum(costs[first, second] for first, second in zip((0, *order), order))
               for order in itertools.permutations(range(1, len(costs))))


@pytest.mark.parametrize('count', [2, 4, 7])
def test_lower_bound_matches_small_exact_relaxation(count):
    generator = np.random.default_rng(361412000 + count)
    centers = generator.uniform(-100, 100, (count, 2))
    radii = np.r_[0.0, generator.uniform(0, 15, count - 1)]
    costs = neighborhood_costs(centers, radii)
    result = open_path_lower_bound(costs, 10)
    exact = exact_path(costs)
    assert result['connected_relaxation_optimal']
    assert result['lower_bound_meters'] <= exact + 1e-7
    assert abs(result['lower_bound_meters'] - exact) <= 1.1e-4


def test_subtours_are_not_treated_as_open_paths():
    centers = np.array([[0, 0], [1, 0], [0, 1], [1000, 1000], [1001, 1000], [1000, 1001]])
    costs = neighborhood_costs(centers, np.zeros(len(centers)))
    result = open_path_lower_bound(costs, 10)
    assert result['subtour_cuts'] > 0
    assert abs(result['lower_bound_meters'] - exact_path(costs)) <= 1.1e-4


def test_neighborhoods_are_optimistic_and_reject_bad_inputs():
    centers = np.array([[0, 0], [30, 0], [100, 0]])
    costs = neighborhood_costs(centers, [0, 40, 40])
    assert costs[0, 1] == 0
    assert costs[0, 2] == 60
    assert costs[1, 2] == 0
    with pytest.raises(ValueError):
        neighborhood_costs(centers, [0, -1, 0])


def test_zero_budget_does_not_use_a_feasible_objective_as_lower_bound():
    result = open_path_lower_bound(np.array([[0, 100], [100, 0]]), 0)
    assert result['lower_bound_meters'] == 0
    assert result['rounds'] == []
    assert not result['connected_relaxation_optimal']
