from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT / 'research/joint_search_v5')]
import experiment_v5
from prior_policy import PriorPolicy
from geometry import seven_network
from offline_benchmark import OfflineRuleWorld, Source
from plan_geometry import hull_vertices
import numpy as np


def test_prior_does_not_cut_true_belief_and_radius_remains_conservative():
    world = OfflineRuleWorld([Source(1, (1750, 0), 1500, None)], 121011200)
    policy = PriorPolicy(world.port(), 'q3', seven_network(), 'E_joint', 'grid7')
    policy.measure(1, [600, 0])
    policy.clear_positions = [np.array([1740.0, 0]), np.array([0.0, 1750]), np.array([-1760.0, 0])]
    state = policy.states[1]
    original = state.region.wkb
    revision = state.revision
    estimate = policy.estimate(state)
    assert estimate is not None
    center, radius = policy.center_radius(state)
    assert state.region.wkb == original
    assert state.revision == revision
    assert np.linalg.norm(hull_vertices(state.region) - center, axis=1).max() <= radius + 1e-8


def test_no_prior_without_three_successful_clear_positions():
    world = OfflineRuleWorld([Source(1, (1750, 0), 1500, None)], 121011201)
    policy = PriorPolicy(world.port(), 'q3', seven_network(), 'E_joint', 'grid7')
    policy.measure(1, [600, 0])
    policy.clear_positions = [np.array([1750.0, 0]), np.array([0.0, 1750])]
    assert policy.estimate(policy.states[1]) is None
