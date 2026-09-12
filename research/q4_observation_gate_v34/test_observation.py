from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import run_observation as runner
from geometry import dual_ring_network
from observation_policy import ObservationGatePolicy
from offline_benchmark import OfflineRuleWorld, Source


def policy_fixture(mode):
    world = OfflineRuleWorld([Source(1, (900.0, 0.0), 1300.0, 180.0)], 361421001)
    return ObservationGatePolicy(world.port(), 'q4', dual_ring_network(), 'F_route', 'dual21', mode=mode), world


def test_identity_is_action_exact():
    scene = next(scene for scene in runner.experiment.make_scenes(361421100, 'unit', [10]) if scene['problem'] == 'q4' and scene['profile'] == 'random')
    previous, trace = runner.evaluate(scene, 'previous')
    identity, same = runner.evaluate(scene, 'identity')
    assert previous['success'] and identity['success']
    assert trace == same


@pytest.mark.parametrize('mode', ['two_bearings', 'clear_ready'])
def test_defer_single_bearing_but_keep_recovery_and_finish(mode):
    policy, world = policy_fixture(mode)
    policy.measure(1, np.zeros(2))
    state = policy.states[1]
    assert len(state.positives) == 1
    assert not policy.admitted(state, 'v_probe')
    for kind in ('clear', 'anchor_clear', 'near', 'v_mirror'):
        assert policy.admitted(state, kind)
    trace = list(world.trace)
    revision = state.revision
    for _iteration in range(3):
        assert not policy.admitted(state, 'probe')
    assert world.trace == trace and state.revision == revision
    policy.discovery_done = True
    assert policy.admitted(state, 'v_probe')


def test_two_bearings_and_clear_ready_have_distinct_admission():
    paired, _world = policy_fixture('two_bearings')
    deferred, _other_world = policy_fixture('clear_ready')
    state = SimpleNamespace(positives=[object(), object()])
    assert paired.admitted(state, 'v_probe')
    assert not deferred.admitted(state, 'v_probe')


@pytest.mark.parametrize('mode', ['two_bearings', 'clear_ready'])
def test_complete_mission_still_clears_all_sources(mode):
    scene = next(scene for scene in runner.experiment.make_scenes(361421100, 'unit', [10]) if scene['problem'] == 'q4' and scene['profile'] == 'random')
    row, _trace = runner.evaluate(scene, mode)
    assert row['success'], row['failure']
    assert row['cleared_count'] == 10
    assert row['deferred_channels']


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match='Unknown observation-gate mode'):
        policy_fixture('unsupported')
