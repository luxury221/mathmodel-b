import math
import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_range import DirectionalRangePolicy, OmniRangePolicy, evaluate, experiment
from geometry import dual_ring_network, seven_network
from offline_benchmark import OfflineRuleWorld, Source
from plan_geometry import geometry_contains, initial_outer_region
from plan_policy import ChannelBelief
from range_geometry import RangeCoupler, conservative_update, guaranteed_visibility, range_bisector


def test_distance_order_has_correct_polarity_and_really_cuts():
    state = ChannelBelief(1, initial_outer_region(), status='DETECTED')
    state.positives = [(np.array([0, 0]), 0.0)]
    state.negatives = [np.array([700, 1000])]
    coupler = RangeCoupler()
    region = coupler.refine_omni(state)
    assert region.covers(Point(900, 0))
    assert not region.covers(Point(1500, 0))
    assert coupler.effective_cuts == 1
    assert region.area < state.region.area
    state.region = region
    assert coupler.refine_omni(state).equals(region)
    assert coupler.pairs == 1


def test_random_omni_fixed_radius_truth_and_algebra():
    generator = np.random.default_rng(321411201)
    for _trial in range(120):
        source = generator.uniform(-1000, 1000, 2)
        radius = generator.uniform(1000, 1500)
        positive_angle, negative_angle = generator.uniform(0, 2 * np.pi, 2)
        positive = source + generator.uniform(10, radius) * np.array([math.cos(positive_angle), math.sin(positive_angle)])
        negative = source + generator.uniform(radius + 0.001, radius + 700) * np.array([math.cos(negative_angle), math.sin(negative_angle)])
        normal, bound = range_bisector(positive, negative)
        assert normal @ source <= bound + 1e-8
        state = ChannelBelief(1, initial_outer_region(), status='DETECTED', positives=[(positive, 0.0)], negatives=[negative])
        assert geometry_contains(RangeCoupler().refine_omni(state), source)


def test_inactive_halfplane_does_not_reclip_the_source_domain_boundary():
    region = initial_outer_region().buffer(1e-9)
    state = ChannelBelief(1, region, status='DETECTED',
                         positives=[(np.array([0.0, 0.0]), 0.0)], negatives=[np.array([0.0, 10000.0])])
    coupler = RangeCoupler()
    assert coupler.refine_omni(state) is region
    assert coupler.effective_cuts == 0


def test_roundoff_guard_keeps_a_larger_feasible_region_not_a_smaller_one():
    original = Point(0, 0).buffer(10)
    tiny = original.buffer(-1e-10)
    significant = original.buffer(-0.1)
    assert conservative_update(original, tiny) is original
    assert conservative_update(original, significant) is significant


def test_q3_does_not_assume_radius_equals_minimum():
    world = OfflineRuleWorld([Source(1, (1400, 0), 1500, None)], 321411202)
    policy = OmniRangePolicy(world.port(), 'q3', seven_network(), 'E_joint', 'grid7')
    policy.measure(1, [0, 0])
    assert geometry_contains(policy.states[1].region, [1400, 0])
    policy.measure(1, [-200, 600])
    assert geometry_contains(policy.states[1].region, [1400, 0])


def test_q4_back_facing_negative_does_not_apply_omni_cut():
    world = OfflineRuleWorld([Source(1, (900, 0), 1000, 180)], 321411203)
    policy = DirectionalRangePolicy(world.port(), 'q4', dual_ring_network(), 'F_route', 'dual21')
    assert policy.measure(1, [0, 0])['result'] == 'direction'
    assert policy.measure(1, [1200, 0])['result'] == 'no_signal'
    assert geometry_contains(policy.states[1].region, [900, 0])
    assert not policy.direction_beliefs[1].omni.covers(Point(900, 0))


@pytest.mark.parametrize('index', [0, 8, 18, 35])
def test_visibility_is_guaranteed_over_entire_direction_interval(index):
    negative = np.array([700.0, 1000.0])
    region = guaranteed_visibility(negative, index, 36)
    generator = np.random.default_rng(321411210 + index)
    samples = generator.uniform(-1800, 1800, (600, 2))
    for sample in samples:
        if region.covers(Point(sample)):
            for angle in np.deg2rad(np.linspace(index * 10 - 5, index * 10 + 5, 41)):
                assert (negative - sample) @ np.array([math.cos(angle), math.sin(angle)]) >= -1e-8


@pytest.mark.parametrize('heading', [0.000001, 4.999999, 175.000001, 359.999999])
def test_directional_sequence_keeps_truth_at_bin_boundaries(heading):
    source = np.array([450.0, -520.0])
    world = OfflineRuleWorld([Source(1, tuple(source), 1200, heading)], 321411251, 'constant_extreme')
    policy = DirectionalRangePolicy(world.port(), 'q4', dual_ring_network(), 'F_route', 'dual21')
    angles = np.deg2rad(heading + np.array([0, 80, -80, 100, -100, 30, 150, -150]))
    distances = (800, 1100, 1000, 600, 900, 1400, 1350, 1450)
    for angle, distance in zip(angles, distances):
        point = source + distance * np.array([math.cos(angle), math.sin(angle)])
        policy.measure(1, point)
        assert geometry_contains(policy.states[1].region, source)


@pytest.mark.parametrize('problem', ['q3', 'q4'])
def test_full_task_clearance_and_clock(problem):
    scene = next(scene for scene in experiment.make_scenes(321411300, 'unit_test', (10,))
                 if scene['problem'] == problem and scene['profile'] == 'random')
    row, _trace = evaluate(scene, 'coupled')
    assert row['success'], row['failure']
    assert row['cleared_count'] == 10
    assert row['policy_stats']['range_pairs'] > 0
