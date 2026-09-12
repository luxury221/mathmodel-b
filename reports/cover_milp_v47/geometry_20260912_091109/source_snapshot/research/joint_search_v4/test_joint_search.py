from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT / 'research' / 'offline_validation')]

import numpy as np
import pytest
from coverage import DirectionalBelief, DirectionalCoverage, OmniCoverage
from geometry import dual_ring_network, seven_network
from offline_benchmark import OfflineRuleWorld, Source
from plan_geometry import geometry_contains, initial_outer_region
from policy import JointSearchPolicy


def test_directional_cut_is_inside_every_heading():
    coverage = DirectionalCoverage()
    for index, offsets in enumerate(coverage.offsets):
        assert np.linalg.norm(offsets, axis=1).max() < 1000
        for angle in np.deg2rad(index * 10 + np.linspace(-5, 5, 21)):
            direction = np.array([np.cos(angle), np.sin(angle)])
            assert np.min(-offsets @ direction) >= -1e-9


def test_directional_negative_does_not_mean_empty_disk():
    coverage = DirectionalCoverage()
    coverage.observe_absence(np.array([0.0, 0.0]))
    assert coverage.contains_hypothesis([100.0, 0.0], 0)
    assert not coverage.contains_hypothesis([100.0, 0.0], 180)
    assert not coverage.empty


def test_continuous_heading_boundaries_preserved():
    generator = np.random.default_rng(85101991)
    for heading in (-0.0000001, 0, 4.9999999, 5, 5.0000001, 175, 180, 355, 359.9999999):
        position = np.array([100.0, -100.0])
        coverage = DirectionalCoverage()
        direction = np.array([np.cos(np.deg2rad(heading)), np.sin(np.deg2rad(heading))])
        for receiver in generator.uniform(-1700, 1700, (25, 2)):
            visible = np.linalg.norm(receiver - position) <= 1000 and np.dot(direction, receiver - position) >= 0
            if not visible:
                coverage.observe_absence(receiver)
                assert coverage.contains_hypothesis(position, heading)


def test_omni_original_search_proves_absence():
    coverage = OmniCoverage()
    for position in seven_network():
        coverage.observe_absence(position)
    assert coverage.empty


@pytest.mark.parametrize('problem', ['q3', 'q4'])
def test_near_source_clears_and_clock_matches(problem):
    world = OfflineRuleWorld([Source(3, (0.001, 0.001), 1000.0, None)], 85312222)
    policy = JointSearchPolicy(world.port(), problem, seven_network() if problem == 'q3' else dual_ring_network(),
                               'E_joint' if problem == 'q3' else 'F_route', 'grid7' if problem == 'q3' else 'dual21')
    result = policy.run()
    assert not world.remaining
    assert result['cleared_channels'] == [3]
    assert abs(result['virtual_seconds'] - world.virtual_seconds) < 1e-7


def test_stopping_requires_certificate():
    world = OfflineRuleWorld([], 85312223)
    policy = JointSearchPolicy(world.port(), 'q4', dual_ring_network(), 'F_route', 'dual21')
    with pytest.raises(ValueError, match='Nonempty'):
        policy.finish_discovery('coverage')
    with pytest.raises(ValueError, match='not observed'):
        policy.finish_discovery('maximum_count')


def test_omni_belief_survives_surrounding_positives():
    belief = DirectionalBelief()
    positives = [(np.array([100.0, 0.0]), 180), (np.array([-100.0, 0.0]), 0),
                 (np.array([0.0, 100.0]), 270), (np.array([0.0, -100.0]), 90)]
    posterior = belief.update(initial_outer_region(), positives, [])
    assert geometry_contains(posterior, [0.0, 0.0])


def test_directional_belief_conservatively_contains_source():
    generator = np.random.default_rng(85712778)
    for heading in (None, 0, 5, 123.456, 355.0):
        world = OfflineRuleWorld([Source(3, (840.0, 720.0), 1111.0, heading)], 8571777)
        belief = DirectionalBelief()
        positives = []
        negatives = []
        for receiver in generator.uniform(-1700, 1700, (30, 2)):
            response = world.measure(receiver, 3)
            if response['result'] == 'direction':
                positives.append((receiver, response['bearing_deg']))
            elif response['result'] == 'no_signal':
                negatives.append(receiver)
            posterior = belief.update(initial_outer_region(), positives, negatives)
            assert geometry_contains(posterior, [840.0, 720.0])


@pytest.mark.parametrize('count', [13, 16])
def test_mixed_geometry_certificate_regression(count):
    from experiment import evaluate, make_scenes
    scene = next(scene for scene in make_scenes() if scene['problem'] == 'q4'
                 and scene['profile'] == 'near_origin' and scene['count'] == count)
    record, _trace = evaluate(scene, 'adaptive_short')
    assert record['success'], record['failure']
    assert record['checks']['covers'] > 0


def test_clear_sweep_discovers_backside_source_without_switch():
    from experiment import SweepPreviousPolicy
    world = OfflineRuleWorld([Source(1, (0.001, 0.001), 1000, None),
                              Source(3, (0.001, 0.001), 1000, 45)], 85815122)
    policy = SweepPreviousPolicy(world.port(), 'q4', dual_ring_network(), 'F_route', 'dual21')
    assert policy.measure(1, np.zeros(2))['result'] == 'near'
    assert policy.measure(3, np.zeros(2))['result'] == 'no_signal'
    assert policy.clear(1, np.zeros(2))
    assert not world.remaining
    assert policy.states[3].status == 'CLEARED'
    assert policy.stats['sweep_discoveries'] == 1
    assert policy.receiver_channel == world.receiver_channel == 3
    assert policy.virtual_seconds == world.virtual_seconds
