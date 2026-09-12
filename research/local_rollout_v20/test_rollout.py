import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_rollout import LocalRolloutPolicy, evaluate, experiment
from geometry import seven_network
from local_model import HypothesisPort, conditional_samples, forecast
from offline_benchmark import OfflineRuleWorld, Source
from plan_geometry import hull_vertices


def make_policy():
    world = OfflineRuleWorld([Source(1, (1200, 100), 1500, None)], 331411201)
    policy = LocalRolloutPolicy(world.port(), 'q3', seven_network(), 'E_joint', 'grid7')
    policy.measure(1, [0, 0])
    return world, policy


def test_hypothetical_port_has_no_actual_world_and_fixed_error():
    port = HypothesisPort(1, [1200, 0], 1500, 1)
    assert port.measure([0, 0], 1) == port.measure([0, 0], 1)
    assert port.measure([-400, 0], 1)['result'] == 'no_signal'
    assert port.clear([1200, 0], 1)['result'] == 'success'
    with pytest.raises(ValueError):
        port.measure([0, 0], 2)


def test_forecast_does_not_change_actual_observations_or_clock():
    world, policy = make_policy()
    state = policy.states[1]
    point = policy.choose_probe(state)
    before = (state.region.wkb, state.revision, len(world.trace), world.virtual_seconds,
              policy.position.copy(), len(state.positives), len(state.negatives))
    samples = conditional_samples(policy.hypotheses(state))
    result = forecast(policy, state, samples[:2], point=point)
    assert result['sample_count'] == 6
    assert not result['failures']
    assert state.region.wkb == before[0] and state.revision == before[1]
    assert len(world.trace) == before[2] and world.virtual_seconds == before[3]
    assert np.array_equal(policy.position, before[4])
    assert (len(state.positives), len(state.negatives)) == before[5:]


def test_new_candidates_have_full_range_certificate_and_spacing():
    _world, policy = make_policy()
    state = policy.states[1]
    original = policy.choose_probe(state)
    candidates = policy.candidates(state, original)
    assert np.array_equal(candidates[0], original)
    assert len(candidates) <= 8
    for point in candidates[1:]:
        assert np.linalg.norm(hull_vertices(state.region) - point, axis=1).max() <= 999.9
        assert all(np.linalg.norm(point - prior) >= 25 for prior in state.measured_positions)


@pytest.mark.parametrize('mode', ['rollout', 'cost_clear', 'clear_first'])
def test_full_task_clearance_and_clock(mode):
    scene = next(scene for scene in experiment.make_scenes(331411300, 'unit_test', (10,))
                 if scene['problem'] == 'q3' and scene['profile'] == 'random')
    row, _trace = evaluate(scene, mode)
    assert row['success'], row['failure']
    assert row['cleared_count'] == 10
