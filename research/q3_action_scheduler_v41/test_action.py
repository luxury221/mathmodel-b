from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import run_action as runner
from action_policy import ActionSchedulerPolicy
from geometry import seven_network
from offline_benchmark import OfflineRuleWorld, Source
from shapely.geometry import Point


def policy_fixture(mode='stepwise_action'):
    world = OfflineRuleWorld([Source(1, (850.0, 100.0), 1200.0, None)], 431840351)
    policy = ActionSchedulerPolicy(world.port(), 'q3', seven_network(), 'E_joint', 'grid7', mode=mode)
    policy.measure(1, np.zeros(2))
    return policy, world


def test_action_preview_preserves_real_evidence_and_clock():
    policy, world = policy_fixture()
    state = policy.states[1]
    before = (state.region.wkb, state.revision, state.status, list(world.trace), policy.virtual_seconds)
    action = policy.next_local_action(state, 0)
    assert action.kind == 'probe'
    assert before == (state.region.wkb, state.revision, state.status, world.trace, policy.virtual_seconds)
    assert policy.probe_steps[1] == 0 and policy.active_target is None


def test_preview_matches_first_frozen_localize_action():
    policy, world = policy_fixture('atomic_action')
    action = policy.next_local_action(policy.states[1], 0)
    trace_length = len(world.trace)
    policy.localize(policy.states[1])
    assert world.trace[trace_length]['position'] == action.position.tolist()
    assert world.trace[trace_length]['action'] == 'measure'
    assert policy.states[1].status == 'CLEARED'


def test_step_executes_only_one_active_measurement_then_releases_target():
    policy, world = policy_fixture()
    action = policy.next_local_action(policy.states[1], 0)
    measure_count = world.stats['measure_count']
    clear_count = world.stats['successful_clear_count']
    policy.execute_one(policy.states[1], action)
    assert world.stats['measure_count'] == measure_count + 1
    assert world.stats['successful_clear_count'] == clear_count
    assert policy.probe_steps[1] == 1 and len(policy.probe_history[1]) == 1
    assert policy.active_target is None


def test_clear_preview_uses_actual_nearest_viable_center():
    policy, _world = policy_fixture()
    state = SimpleNamespace(region=Point(0, 0).buffer(100))
    plan = SimpleNamespace(centers=np.array([[80.0, 0.0], [5.0, 0.0]]))
    action = policy.plan_action(state, plan)
    assert np.array_equal(action.position, [5.0, 0.0])


def test_probe_limit_uses_a_certified_fallback():
    policy, world = policy_fixture()
    policy.probe_steps[1] = 8
    action = policy.next_local_action(policy.states[1], 8)
    assert action.kind == 'clear' and action.fallback and action.plan.certified
    policy.execute_one(policy.states[1], action)
    assert policy.states[1].status == 'CLEARED' and not world.remaining
    assert policy.stats['certified_fallbacks'] == 1 and policy.probe_steps[1] == 8
    runner.baseline.audit_trace(world.trace, policy.virtual_seconds)


def test_failed_measurement_resets_active_target_without_fabricating_step(monkeypatch):
    policy, world = policy_fixture()
    action = policy.next_local_action(policy.states[1], 0)
    trace = list(world.trace)
    def fail_measure(*arguments, **keywords):
        raise RuntimeError('injected measurement failure')
    monkeypatch.setattr(policy, 'measure', fail_measure)
    with pytest.raises(RuntimeError, match='measurement failure'):
        policy.execute_one(policy.states[1], action)
    assert world.trace == trace and policy.probe_steps[1] == 0
    assert policy.active_target is None


def test_locked_executor_is_action_exact_to_frozen_baseline():
    scene = next(scene for scene in runner.experiment.make_scenes(441941367, 'unit', [10])
                 if scene['problem'] == 'q3' and scene['profile'] == 'random')
    previous, trace = runner.evaluate(scene, 'previous')
    identity, same = runner.evaluate(scene, 'identity')
    assert previous['success'] and identity['success']
    assert trace == same
    assert identity['policy_stats']['scheduler_probe_steps'] > 0
    assert identity['effective_target_radius'] == 2000


@pytest.mark.parametrize('mode', ['atomic_action', 'stepwise_action'])
def test_complete_mission_clears_all_sources_and_predictions_match(mode):
    scene = next(scene for scene in runner.experiment.make_scenes(441941367, 'unit', [10])
                 if scene['problem'] == 'q3' and scene['profile'] == 'random')
    row, _trace = runner.evaluate(scene, mode)
    assert row['success'], row['failure']
    assert row['cleared_count'] == 10 and row['prediction_checks'] > 0
    assert row['effective_target_radius'] == 2000


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match='Unknown Q3 action-scheduler mode'):
        policy_fixture('unsupported')
