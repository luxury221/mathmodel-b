from __future__ import annotations

import numpy as np
import pytest
import run_bounded as runner
from bounded_policy import BoundedProbePolicy
from geometry import dual_ring_network
from offline_benchmark import OfflineRuleWorld, Source


def test_identity_is_action_exact():
    scene = next(scene for scene in runner.experiment.make_scenes(361419100, 'unit', [10]) if scene['problem'] == 'q4' and scene['profile'] == 'random')
    previous, trace = runner.evaluate(scene, 'previous')
    identity, same = runner.evaluate(scene, 'identity')
    assert previous['success'] and identity['success']
    assert trace == same


def test_planning_does_not_consume_action_allowance():
    world = OfflineRuleWorld([Source(1, (900.0, 0.0), 1300.0, 180.0)], 361419001)
    policy = BoundedProbePolicy(world.port(), 'q4', dual_ring_network(), 'F_route', 'dual21', mode='single')
    policy.measure(1, np.zeros(2))
    assert not policy.clear(1, np.zeros(2))
    state = policy.states[1]
    trace = list(world.trace)
    revision = state.revision
    first = policy.next_target_action(state)
    second = policy.next_target_action(state)
    assert first[0] == second[0] == 'probe'
    assert np.array_equal(first[1], second[1])
    assert state.revision == revision and world.trace == trace
    assert not policy.bounded_used and policy.stats['bounded_actions'] == 0
    policy.measure(1, first[1], active=True)
    assert policy.bounded_used == {1}
    assert len(policy.bounded_events) == 1
    assert policy.stats['bounded_actions'] == 1
    assert not policy.bounded_pending
    assert not policy.pending_probes


@pytest.mark.parametrize('mode', ['single', 'cost_guarded'])
def test_full_mission_keeps_clearance_and_one_action_limit(mode):
    scene = next(scene for scene in runner.experiment.make_scenes(361419100, 'unit', [10]) if scene['problem'] == 'q4' and scene['profile'] == 'random')
    row, _trace = runner.evaluate(scene, mode)
    assert row['success'], row['failure']
    channels = [event['channel'] for event in row['bounded_events']]
    assert len(set(channels)) == len(channels)
    assert row['cleared_count'] == 10
