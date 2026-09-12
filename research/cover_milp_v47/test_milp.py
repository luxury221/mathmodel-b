from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

from milp_search import accepted_binary_candidate, model_rows, solve_milp


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'cover_milp_audit_v47'))
from verify_milp import audit_model_rows


def toy_model():
    pool = np.array([[0, 0], [100, 0], [102, 0], [0, 100], [-100, 0]], dtype=float)
    groups = [[1, 2], [3], [4]]
    cuts = [{'point_indices': [1, 2, 3, 4], 'required_points': 3}]
    rows = model_rows(pool, groups, cuts, minimum=4, maximum=4)
    return pool, groups, cuts, rows


def test_milp_finds_valid_integer_assignment_with_spacing():
    pool, groups, cuts, rows = toy_model()
    assert audit_model_rows(pool, groups, cuts, rows, minimum=4, maximum=4) == 7
    chosen, metadata = solve_milp(rows, len(pool), seconds=2)
    assert metadata['status'] == 'OPTIMAL' and len(chosen) == 4 and 0 in chosen
    assert not {1, 2}.issubset(chosen)


def test_independent_row_audit_rejects_omitted_conflict():
    pool, groups, cuts, rows = toy_model()
    damaged = [row for row in rows if row['kind'] != 'spacing']
    with pytest.raises(ValueError, match='differs'):
        audit_model_rows(pool, groups, cuts, damaged, minimum=4, maximum=4)


def test_solver_fractional_or_constraint_violating_vector_is_rejected():
    pool, _groups, _cuts, rows = toy_model()
    with pytest.raises(ValueError, match='binary'):
        accepted_binary_candidate([1, 0.5, 0.5, 1, 1], rows, len(pool))
    with pytest.raises(ValueError, match='row check'):
        accepted_binary_candidate([1, 1, 1, 1, 0], rows, len(pool))


def test_empty_cover_group_is_reported_infeasible_not_dropped():
    pool = np.array([[0, 0], [100, 0]], dtype=float)
    rows = model_rows(pool, [[]], [], minimum=1, maximum=2)
    chosen, metadata = solve_milp(rows, len(pool), seconds=2)
    assert chosen is None and metadata['status'] == 'INFEASIBLE'


def test_independent_row_audit_rejects_relaxed_layer_quota():
    pool, groups, cuts, rows = toy_model()
    damaged = copy.deepcopy(rows)
    damaged[-1]['lower'] -= 1
    with pytest.raises(ValueError, match='differs'):
        audit_model_rows(pool, groups, cuts, damaged, minimum=4, maximum=4)
