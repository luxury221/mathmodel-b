import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_commit import CommittedPatrolPolicy, evaluate, experiment
from geometry import dual_ring_network
from offline_benchmark import OfflineRuleWorld, Source


def test_patrol_is_complete_and_scheduling_has_no_evidence_effect():
    world = OfflineRuleWorld([Source(1, (850, 100), 1000, 180)], 311411201)
    policy = CommittedPatrolPolicy(world.port(), 'q4', dual_ring_network(), 'F_route', 'dual21')
    assert sorted(policy.patrol_order) == list(range(1, 21))
    policy.clear(1, [0, 0])
    policy.measure(1, [0, 0])
    state = policy.states[1]
    kind, point, extra = policy.next_target_action(state)
    stops = [('fixed', index, policy.route[index], None) for index in policy.remaining_stations]
    stops.append((kind, 1, point, extra))
    before = (state.region.wkb, state.revision, world.virtual_seconds, len(world.trace))
    allowed, _current = policy.scheduled_stops(stops)
    assert len([stop for stop in allowed if stop[0] == 'fixed']) == 1
    assert (state.region.wkb, state.revision, world.virtual_seconds, len(world.trace)) == before
    policy.due_channels.add(1)
    allowed, _current = policy.scheduled_stops(stops)
    assert len(allowed) == 1 and allowed[0][1] == 1


def test_closest_segment_uses_entire_remaining_route():
    world = OfflineRuleWorld([], 311411202)
    policy = CommittedPatrolPolicy(world.port(), 'q4', dual_ring_network(), 'F_route', 'dual21')
    policy.route = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    assert policy.closest_segment(np.array([40, 3]), [1, 2, 3]) == 0
    assert policy.closest_segment(np.array([105, 70]), [1, 2, 3]) == 1
    assert policy.closest_segment(np.array([30, 101]), [1, 2, 3]) == 2


def test_full_task_and_patrol_order():
    scene = experiment.make_scenes(311411300, 'unit_test', (10,))[6]
    row, _trace = evaluate(scene, 'committed')
    assert row['success'], row['failure']
    assert row['cleared_count'] == 10
    world = OfflineRuleWorld([], 311411204)
    policy = CommittedPatrolPolicy(world.port(), 'q4', dual_ring_network(), 'F_route', 'dual21')
    result = policy.run()
    assert policy.coverage.empty
    assert world.virtual_seconds == result['virtual_seconds']
    assert [event['identifier'] for event in policy.commit_events if event['action'] == 'fixed'] == policy.patrol_order
