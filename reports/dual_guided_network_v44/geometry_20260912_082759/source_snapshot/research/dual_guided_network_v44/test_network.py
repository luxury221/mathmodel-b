from __future__ import annotations

import numpy as np
import search_network as search
from geometry import dual_ring_network


def test_pool_retains_reference_sites_and_has_no_source_inputs():
    pool = search.pool_points()
    assert len(pool) == 649 and np.array_equal(pool[0], [0.0, 0.0])
    for point in dual_ring_network():
        assert np.min(np.linalg.norm(pool - point, axis=1)) < 1e-8


def test_cover_solver_obeys_groups_cardinality_and_spacing():
    pool = np.array([[0.0, 0.0], [100, 0], [102, 0], [0, 100], [-100, 0]])
    groups = [[1, 2], [3], [4]]
    chosen, result = search.solve_cover(pool, groups, [], minimum=4, maximum=4, seconds=2)
    assert result['status'] == 'OPTIMAL' and len(chosen) == 4
    assert not {1, 2} <= set(chosen)
    assert all(set(group) & set(chosen) for group in groups)


def test_blocked_geometry_set_is_not_returned_again():
    pool = np.array([[0.0, 0.0], [100, 0], [0, 100]])
    chosen, result = search.solve_cover(pool, [[1], [2]], [[0, 1, 2]], minimum=3, maximum=3, seconds=2)
    assert chosen is None and result['status'] == 'INFEASIBLE'


def test_original_network_still_has_a_continuous_certificate():
    assert search.certify(dual_ring_network()).empty
