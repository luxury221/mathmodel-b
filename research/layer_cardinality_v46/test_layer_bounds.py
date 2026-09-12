from __future__ import annotations

import copy
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

from layer_bounds import disjoint_interval_lower, generate_interval_bounds, integer_certificate
from run_layer_search import solve_with_cuts


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'layer_cardinality_audit_v46'))
from verify_cuts import verify_combination, verify_cut


def test_triangle_fractional_dual_rounds_to_two_and_all_assignments_obey():
    groups = [[1, 2], [1, 3], [2, 3]]
    cut = integer_certificate(groups, [0, 1, 2], [1, 2, 3], [0.5, 0.5, 0.5])
    assert cut['fractional_lower'] == 1.5 and cut['required_points'] == 2
    assert verify_cut(groups, cut) == 3
    for flags in itertools.product((False, True), repeat=3):
        selected = set(itertools.compress([1, 2, 3], flags))
        if all(selected.intersection(group) for group in groups):
            assert len(selected) >= cut['required_points']


def test_exact_renormalization_handles_floating_budget_overshoot():
    groups = [[1, 2], [1, 3], [2, 3]]
    cut = integer_certificate(groups, [0, 1, 2], [1, 2, 3], [0.500001, 0.500001, 0.500001])
    assert cut['denominator'] > 100000000
    assert cut['required_points'] == 2 and verify_cut(groups, cut) == 3


def test_lp_proposal_is_only_used_through_integer_certificate():
    groups = [[1, 2], [1, 3], [2, 3]]
    cuts = generate_interval_bounds(groups, layer_size=1, layer_count=3)
    total = next(cut for cut in cuts if cut['first_layer'] == 0 and cut['last_layer'] == 2)
    assert total['lp_status'] == 0 and total['required_points'] == 2
    assert all(verify_cut(groups, cut) >= 1 for cut in cuts)
    assert disjoint_interval_lower(cuts, layer_count=3)['with_forced_origin_lower'] == 3


def test_empty_interval_contributes_no_station_lower_bound():
    cuts = generate_interval_bounds([], layer_size=2, layer_count=2)
    assert len(cuts) == 3 and all(cut['required_points'] == 0 for cut in cuts)
    assert disjoint_interval_lower(cuts, layer_count=2)['with_forced_origin_lower'] == 1


def test_disjoint_interval_bounds_add_but_overlaps_do_not():
    cuts = [{'first_layer': 0, 'last_layer': 1, 'required_points': 3},
            {'first_layer': 1, 'last_layer': 2, 'required_points': 3},
            {'first_layer': 2, 'last_layer': 2, 'required_points': 2}]
    combination = disjoint_interval_lower(cuts, layer_count=3)
    assert combination['with_forced_origin_lower'] == 6
    assert verify_combination(cuts, combination) == 6
    feasible_totals = [sum(counts) for counts in itertools.product(range(6), repeat=3)
                       if all(sum(counts[cut['first_layer']:cut['last_layer'] + 1]) >= cut['required_points'] for cut in cuts)]
    assert min(feasible_totals) == combination['nonorigin_lower']


def test_independent_combination_verifier_rejects_overlap():
    cuts = [{'first_layer': 0, 'last_layer': 1, 'required_points': 3},
            {'first_layer': 1, 'last_layer': 2, 'required_points': 3}]
    damaged = {'disjoint_cut_indices': [0, 1], 'with_forced_origin_lower': 7, 'nonorigin_lower': 6}
    with pytest.raises(ValueError, match='Overlapping'):
        verify_combination(cuts, damaged)


def test_independent_verifier_rejects_over_budget_certificate():
    groups = [[1, 2], [1, 3], [2, 3]]
    cut = integer_certificate(groups, [0, 1, 2], [1, 2, 3], [0.5, 0.5, 0.5])
    cut['denominator'] -= 1
    with pytest.raises(ValueError, match='budget'):
        verify_cut(groups, cut)


def test_independent_verifier_rejects_invented_required_count():
    groups = [[1, 2]]
    cut = integer_certificate(groups, [0], [1, 2], [1.0])
    damaged = copy.deepcopy(cut)
    damaged['required_points'] += 1
    with pytest.raises(ValueError, match='lower bound'):
        verify_cut(groups, damaged)


def test_groups_crossing_interval_are_not_valid_dual_rows():
    with pytest.raises(ValueError, match='inside'):
        integer_certificate([[1, 2]], [0], [1], [0.5])


def test_strengthened_cover_solver_obeys_exact_cardinality_cut():
    pool = np.array([[0, 0], [100, 0], [0, 100], [-100, 0]], dtype=float)
    groups = [[1, 2], [1, 3], [2, 3]]
    cut = integer_certificate(groups, [0, 1, 2], [1, 2, 3], [0.5, 0.5, 0.5])
    selected, metadata = solve_with_cuts(pool, groups, [cut], minimum=1, maximum=4, seconds=2)
    assert metadata['status'] == 'OPTIMAL' and len(selected) == 3 and 0 in selected
    assert metadata['added_cardinality_cuts'] == 1
