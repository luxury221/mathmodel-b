from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_point_benchmark import EmpiricalPointPolicy
from geometry import seven_network
from offline_benchmark import OfflineRuleWorld, Source
from plan_geometry import hull_vertices


def prepared(target=(1768, 220), mode='empirical_point'):
    supports = [(1782, 0), (0, 1782), (-1782, 0)]
    sources = [Source(1, target, 1000, None), *[Source(index + 2, point, 1000, None) for index, point in enumerate(supports)]]
    world = OfflineRuleWorld(sources, 255511101)
    policy = EmpiricalPointPolicy(world.port(), 'q3', seven_network(), 'E_joint', 'grid7', mode=mode)
    for index, point in enumerate(supports):
        assert policy.clear(index + 2, point)
    policy.measure(1, [1000, 0])
    return world, policy


def test_empirical_center_keeps_full_true_feasible_region():
    world, policy = prepared()
    state = policy.states[1]
    before = (state.region.wkb, state.revision, len(world.trace), world.virtual_seconds)
    point, radius = policy.center_radius(state)
    assert policy.estimate(state) is not None
    assert np.max(np.linalg.norm(hull_vertices(state.region) - point, axis=1)) <= radius + 1e-8
    assert (state.region.wkb, state.revision, len(world.trace), world.virtual_seconds) == before


def test_less_than_three_successes_produces_no_prior():
    world = OfflineRuleWorld([Source(1, (1768, 220), 1000, None)], 255511102)
    policy = EmpiricalPointPolicy(world.port(), 'q3', seven_network(), 'E_joint', 'grid7')
    policy.measure(1, [1000, 0])
    assert policy.estimate(policy.states[1]) is None


def test_wrong_prior_trial_is_charged_and_falls_back_to_real_localization():
    world, policy = prepared(target=(1640, 150), mode='empirical_trial')
    policy.localize(policy.states[1])
    assert policy.states[1].status == 'CLEARED'
    assert policy.stats['empirical_trials'] == 1
    assert policy.stats['empirical_failed_trials'] == 1
    assert world.stats['failed_clear_count'] >= 1
    assert policy.virtual_seconds == world.virtual_seconds
