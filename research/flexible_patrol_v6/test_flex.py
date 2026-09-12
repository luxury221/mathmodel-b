from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiment import FlexiblePolicy
from flex_planner import CellAssignment, clip_halfplane
from geometry import seven_network, dual_ring_network
from offline_benchmark import OfflineRuleWorld
import numpy as np
from shapely.geometry import Polygon


def fixture_policy(problem):
    policy = FlexiblePolicy(OfflineRuleWorld([], 108131200).port(), problem,
                            seven_network() if problem == 'q3' else dual_ring_network(),
                            'E_joint' if problem == 'q3' else 'F_route', 'grid7' if problem == 'q3' else 'dual21')
    policy.coverage.observe_clear_absence([0, 0])
    policy.coverage.observe_absence([0, 0])
    return policy


def test_halfplane_crossing_never_extrapolates():
    vertices = np.array([[0.9e-10, 0], [1.1e-10, 1], [2, 1], [2, 0]])
    clipped = clip_halfplane(vertices, [1, 0], 0)
    assert len(clipped) == 0
    square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    for scale in (1e-8, 1, 1e8):
        clipped = clip_halfplane(square, [scale, scale], 0)
        assert np.max(clipped.sum(axis=1)) <= 1e-12
        assert Polygon(square).covers(Polygon(clipped))


def test_omni_cell_assignment_contains_only_receivable_vertices():
    policy = fixture_policy('q3')
    assignment = CellAssignment(policy.coverage, policy.route[1:], 'q3')
    assert assignment.build()
    for station, vertices in zip(policy.route[1:], assignment.vertices):
        assert vertices
        assert np.linalg.norm(np.asarray(vertices) - station, axis=1).max() < 999.9


def test_flexible_omni_layout_is_certified_and_shorter():
    policy = fixture_policy('q3')
    layout = policy.patrol.propose(policy.coverage, [[1600, 300], [100, 1600], [-1550, 400]])
    assert layout is not None
    assert layout.new_meters < layout.old_meters - 100
    assert layout.distance_limit < 1000
    assert layout.halfplane_violation < 1e-7


def test_directional_assignment_covers_continuous_heading_intervals():
    policy = fixture_policy('q4')
    assignment = CellAssignment(policy.coverage, policy.route[1:], 'q4')
    assert assignment.build()
    assert assignment.maximum_depth > 0
    for station, vertices, planes in zip(policy.route[1:], assignment.vertices, assignment.halfplanes):
        if vertices:
            assert np.linalg.norm(np.asarray(vertices) - station, axis=1).max() <= 999.90001
        for normal, bound in planes:
            assert np.dot(normal, station) >= bound - 1e-8


def test_nearest_eligible_directional_cells_remain_feasible():
    policy = fixture_policy('q4')
    assignment = CellAssignment(policy.coverage, policy.route[1:], 'q4', assignment='nearest')
    assert assignment.build()
    for station, vertices, planes in zip(policy.route[1:], assignment.vertices, assignment.halfplanes):
        if vertices:
            assert np.linalg.norm(np.asarray(vertices) - station, axis=1).max() <= 999.90001
        for normal, bound in planes:
            assert np.dot(normal, station) >= bound - 1e-8
