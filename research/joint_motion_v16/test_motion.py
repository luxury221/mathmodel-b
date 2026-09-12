import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_motion import JointMotionPolicy, evaluate, experiment
from geometry import dual_ring_network
from offline_benchmark import OfflineRuleWorld, Source


def make_policy(mode='combined'):
    world = OfflineRuleWorld([Source(1, (850, 100), 1000, 180)], 281411201)
    policy = JointMotionPolicy(world.port(), 'q4', dual_ring_network(), 'F_route', 'dual21', mode=mode)
    policy.clear(1, [0, 0])
    policy.measure(1, [0, 0])
    return world, policy


def test_proposals_are_not_observations():
    world, policy = make_policy()
    state = policy.states[1]
    kind, point, extra = policy.next_target_action(state)
    before = (state.region.wkb, state.revision, world.virtual_seconds, len(world.trace), len(policy.coverage.observations))
    policy.route_positions([(kind, 1, point, extra)])
    policy.transit_proposal(np.array([1000, 600]), [(kind, 1, point, extra)])
    assert before == (state.region.wkb, state.revision, world.virtual_seconds, len(world.trace), len(policy.coverage.observations))


def test_terminal_routing_does_not_replace_actual_action():
    _world, policy = make_policy('terminal')
    state = policy.states[1]
    kind, point, extra = policy.next_target_action(state)
    assert kind == 'v_probe'
    positions = policy.route_positions([(kind, 1, point, extra)])
    assert np.array_equal(positions[0], policy.center_radius(state)[0])
    assert policy.transit_proposal(point, [(kind, 1, point, extra)]) is None


def test_transit_budget_and_nearby_guard():
    _world, policy = make_policy()
    state = policy.states[1]
    kind, point, extra = policy.next_target_action(state)
    stops = [(kind, 1, point, extra)]
    assert policy.transit_proposal(np.array([30, 20]), stops) is None
    policy.transit_counts[1] = 3
    assert policy.transit_proposal(np.array([1000, 600]), stops) is None


@pytest.mark.parametrize('mode', ['terminal', 'transit', 'combined'])
def test_full_task_and_clock(mode):
    scene = experiment.make_scenes(281411300, 'unit_test', (10,))[6]
    row, trace = evaluate(scene, mode)
    assert row['success'], row['failure']
    assert row['cleared_count'] == 10
    assert trace
    for event in row['motion_events']:
        origin = np.asarray(event['origin'])
        stop = np.asarray(event['actual_stop'])
        destination = np.asarray(event['planned_destination'])
        assert np.linalg.norm(stop - origin) >= 60 - 1e-7
        assert abs(np.linalg.norm(stop - origin) + np.linalg.norm(destination - stop)
                   - np.linalg.norm(destination - origin)) < 1e-6


def test_no_early_stop_without_sources():
    world = OfflineRuleWorld([], 281411204)
    policy = JointMotionPolicy(world.port(), 'q4', dual_ring_network(), 'F_route', 'dual21', mode='terminal')
    result = policy.run()
    assert policy.coverage.empty
    assert len(result['declared_absent']) == 20
    assert world.virtual_seconds == result['virtual_seconds']
    assert world.stats['measure_count'] > 20
