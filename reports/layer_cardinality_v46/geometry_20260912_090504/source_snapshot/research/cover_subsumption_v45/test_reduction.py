from __future__ import annotations

import copy
import itertools
import sys
from pathlib import Path

import pytest

from reduction import reduce_cover


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'cover_subsumption_audit_v45'))
from verify import audit_implications


def satisfies(groups, selection):
    chosen = set(selection)
    return all(chosen.intersection(group) for group in groups)


def test_all_three_point_families_and_assignments_are_equivalent():
    point_count = 3
    possible_groups = [list(itertools.compress(range(point_count), flags))
                       for flags in itertools.product((False, True), repeat=point_count)]
    for family_flags in itertools.product((False, True), repeat=len(possible_groups)):
        groups = list(itertools.compress(possible_groups, family_flags))
        for forced in ((), (0,), (1, 2)):
            proof = reduce_cover(groups, point_count, forced)
            assert audit_implications(groups, proof)
            reduced = [groups[index] for index in proof['kept_indices']]
            for selection in possible_groups:
                if set(forced).issubset(selection):
                    assert satisfies(groups, selection) == satisfies(reduced, selection)


def test_repeated_nested_and_unordered_groups_have_final_subset_witnesses():
    groups = [[4, 2, 1], [2, 1], [1], [3, 4], [4, 3], [0, 2], [1, 1]]
    proof = reduce_cover(groups, 5)
    assert proof['kept_indices'] == [2, 3]
    assert proof['fixed_satisfied_count'] == 1
    assert proof['subsumed_count'] == 4
    assert proof == reduce_cover(groups, 5)
    assert audit_implications(groups, proof)


def test_empty_group_is_preserved_as_contradiction():
    groups = [[1, 2], [], [2], [0], []]
    proof = reduce_cover(groups, 3)
    assert proof['kept_indices'] == [1]
    assert audit_implications(groups, proof)
    assert not satisfies([groups[index] for index in proof['kept_indices']], [0, 1, 2])


@pytest.mark.parametrize('groups,point_count,forced', [([[3]], 3, (0,)), ([[-1]], 3, (0,)),
                                                    ([[1.5]], 3, (0,)), ([[True]], 3, (0,)),
                                                    ([[1]], 3, (4,)), ([], 0, ())])
def test_invalid_indices_and_counts_are_rejected(groups, point_count, forced):
    with pytest.raises(ValueError):
        reduce_cover(groups, point_count, forced)


def test_independent_verifier_rejects_deleted_superset_as_witness():
    groups = [[1], [1, 2], [2, 3]]
    proof = reduce_cover(groups, 4)
    damaged = copy.deepcopy(proof)
    damaged['implications'][0] = {'kind': 'retained_subset', 'index': 2}
    with pytest.raises(ValueError, match='subset'):
        audit_implications(groups, damaged)


def test_independent_verifier_rejects_fixed_point_not_in_group():
    groups = [[1], [0, 1]]
    proof = reduce_cover(groups, 2)
    proof['implications'][0] = {'kind': 'fixed_true', 'point': 0}
    with pytest.raises(ValueError, match='fixed-point'):
        audit_implications(groups, proof)


def test_independent_verifier_rejects_inconsistent_counts():
    groups = [[1], [0, 1]]
    proof = reduce_cover(groups, 2)
    proof['subsumed_count'] += 1
    with pytest.raises(ValueError, match='counts'):
        audit_implications(groups, proof)
