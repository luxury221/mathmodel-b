from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_latency
from geometry import seven_network
from latency_policy import DiscoveryOrderQ3
from latency_route import RouteObjective, improve_route
from offline_benchmark import OfflineRuleWorld, Source
from population import cumulative_subset_mass, population_posterior, public_particles
import policy as original_policy_module


def test_population_marginals_match_enumerated_memberships():
    survival = np.array([[0.2, 0.6, 0.8], [0.7, 0.3, 0.9]])
    prior = np.array([0.4, 0.6])
    joint, presence = population_posterior(survival, np.zeros(2), 0, 0, prior, 1, 2, 3)
    expected_joint = np.zeros_like(joint)
    expected_presence = np.zeros_like(presence)
    for family in range(2):
        for count in (1, 2):
            for members in itertools.combinations(range(3), count):
                weight = prior[family] * np.prod(survival[family, list(members)]) / math.comb(3, count)
                expected_joint[family, count] += weight
                expected_presence[family, list(members)] += weight
    normalizer = expected_joint.sum()
    assert np.allclose(joint, expected_joint / normalizer)
    assert np.allclose(presence, expected_presence / normalizer)
    assert math.isclose(presence.sum(), float(joint @ np.arange(3) @ np.ones(2)))


def test_subset_transform_and_route_objective_match_direct_computation():
    masks = np.array([0, 1, 2, 3])
    weights = np.array([0.5, 2.0, 1.0, 0.5])
    cumulative = cumulative_subset_mass(masks, weights, 2)
    for mask in range(4):
        assert cumulative[mask] == sum(weight for visible, weight in zip(masks, weights) if visible & ~mask == 0)
    points = np.array([[100, 0], [50, 80], [200, 30], [-40, 90]])
    objective = RouteObjective(points, np.zeros(2), [0, 2], cumulative, 2.0, True)
    for permutation in itertools.permutations(range(4)):
        prefix, radio = 0, 0.0
        travel = sum(np.linalg.norm(points[second] - (np.zeros(2) if first == 4 else points[first]))
                     for first, second in zip((4, *permutation), permutation)) / 5
        for node in permutation:
            if node in (0, 2):
                radio += 6 * (2 + sum(weight for mask, weight in zip(masks, weights) if mask & prefix == 0))
                prefix |= 1 if node == 0 else 2
        assert math.isclose(float(objective.scores(permutation)[0]), travel + radio)
    selected, score = improve_route(objective, np.arange(4), [0, 2])
    assert sorted(selected.tolist()) == list(range(4))
    assert score <= objective.scores(np.arange(4))[0] + 1e-8


@pytest.mark.parametrize('problem', ['q3', 'q4'])
def test_public_particles_keep_radius_uncertainty_and_near_origin_mass(problem):
    positions, headings, radii, weights = public_particles(problem)
    assert len(np.unique(radii)) == 3
    assert radii.min() > 1000 and radii.max() < 1500
    assert np.allclose(weights.sum(axis=1), 1)
    assert math.isclose(float(weights[0, np.linalg.norm(positions, axis=1) <= 20].sum()), 20**2 / 1800**2)
    assert weights[1, np.linalg.norm(positions, axis=1) < 1500].sum() == 0


@pytest.mark.parametrize('problem', ['q3', 'q4'])
def test_identity_preserves_all_actions_and_module_globals(problem):
    original_router = original_policy_module.open_route
    scene = next(scene for scene in run_latency.experiment.make_scenes(361417100, 'unit', (10,))
                 if scene['problem'] == problem and scene['profile'] == 'random')
    previous, _trace = run_latency.evaluate(scene, 'previous')
    identity, _trace = run_latency.evaluate(scene, 'identity')
    assert previous['success'] and identity['success']
    assert previous['trace_sha256'] == identity['trace_sha256']
    assert identity['policy_stats']['latency_dispatch_calls'] > 0
    assert original_policy_module.open_route is original_router


def test_plans_do_not_become_observations_and_cached_radio_is_not_new_evidence():
    world = OfflineRuleWorld([Source(channel, (1200.0, 0.0), 1000, None) for channel in range(1, 11)], 361417001)
    policy = DiscoveryOrderQ3(world.port(), 'q3', seven_network(), 'E_joint', 'grid7', mode='mixture')
    policy.measure(1, np.zeros(2))
    revision = policy.population.revision
    clock = world.virtual_seconds
    policy.measure(1, np.zeros(2))
    assert policy.population.revision == revision
    before = {channel: value.copy() for channel, value in policy.population.alive.items()}
    coverage = policy.coverage.region.wkb
    regions = [state.region.wkb for state in policy.states.values()]
    result = policy.order_router(policy.route[1:], policy.position)
    assert len(result) == len(policy.route) - 1
    assert world.virtual_seconds == clock
    assert policy.population.revision == revision
    assert all(np.array_equal(value, policy.population.alive[channel]) for channel, value in before.items())
    assert policy.coverage.region.wkb == coverage
    assert [state.region.wkb for state in policy.states.values()] == regions
