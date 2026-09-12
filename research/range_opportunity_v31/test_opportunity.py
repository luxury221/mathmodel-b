from __future__ import annotations

import numpy as np
import pytest
import run_opportunity as runner
from geometry import dual_ring_network, seven_network
from offline_benchmark import OfflineRuleWorld, Source
from opportunity_policy import RangeOpportunityQ3, RangeOpportunityQ4


@pytest.mark.parametrize('problem', ['q3', 'q4'])
def test_identity_is_action_exact(problem):
    scene = next(scene for scene in runner.experiment.make_scenes(361418100, 'unit', [10]) if scene['problem'] == problem and scene['profile'] == 'random')
    previous, trace = runner.evaluate(scene, 'previous')
    identity, same = runner.evaluate(scene, 'identity')
    assert previous['success'] and identity['success']
    assert trace == same
    assert identity['effective_target_radius'] == 2000


@pytest.mark.parametrize('problem', ['q3', 'q4'])
def test_extended_reuses_actual_point_beyond_guaranteed_radius(problem):
    points = seven_network() if problem == 'q3' else dual_ring_network()
    policy_class = RangeOpportunityQ3 if problem == 'q3' else RangeOpportunityQ4
    outcomes = []
    for mode in ('identity', 'extended'):
        world = OfflineRuleWorld([Source(1, (1300.0, 0.0), 1450.0, None)], 361418001)
        policy = policy_class(world.port(), problem, points, 'E_joint' if problem == 'q3' else 'F_route', 'grid7' if problem == 'q3' else 'dual21', mode=mode)
        policy.measure(1, np.zeros(2))
        stop = np.array([850.0, 1100.0])
        policy.measure(20, stop)
        count = world.stats['measure_count']
        policy.reuse_directions()
        outcomes.append(world.stats['measure_count'] - count)
        if mode == 'extended':
            assert policy.extended_events[0]['lower_distance'] > 1000
            assert policy.extended_events[0]['response'] == 'direction'
            assert np.array_equal(world.trace[-1]['position'], stop)
            assert policy.extended_events[0]['radius_after'] < policy.extended_events[0]['radius_before']
            previous_trace = list(world.trace)
            policy.reuse_directions()
            assert world.trace == previous_trace
    assert outcomes == [0, 1]


@pytest.mark.parametrize('problem', ['q3', 'q4'])
def test_guarded_full_mission_keeps_all_sources(problem):
    scene = next(scene for scene in runner.experiment.make_scenes(361418100, 'unit', [10]) if scene['problem'] == problem and scene['profile'] == 'random')
    row, _trace = runner.evaluate(scene, 'guarded')
    assert row['success'], row['failure']
    assert row['cleared_count'] == 10
    assert all(event['receive_probability'] >= 0.6 and event['predicted_radius'] <= 0.65 * event['radius_before'] for event in row['extended_events'])
