from __future__ import annotations

import copy

import numpy as np
import pytest
import run_rotation as runner
from geometry import seven_network
from offline_benchmark import OfflineRuleWorld, Source
from rotation_policy import SymmetryLayoutPolicy, rotated_layout


def test_identity_is_action_exact():
    scene = next(scene for scene in runner.experiment.make_scenes(361423100, 'unit', [10]) if scene['problem'] == 'q3' and scene['profile'] == 'random')
    previous, trace = runner.evaluate(scene, 'previous')
    identity, same = runner.evaluate(scene, 'identity')
    assert previous['success'] and identity['success']
    assert trace == same


def test_zero_rotation_is_exact_and_other_phases_preserve_radii():
    points = seven_network()
    assert np.array_equal(rotated_layout(points, 0), points)
    for phase in range(5, 60, 5):
        proposal = rotated_layout(points, phase)
        assert np.array_equal(proposal[0], points[0])
        assert np.allclose(np.linalg.norm(proposal, axis=1), np.linalg.norm(points, axis=1), atol=1e-9)


def test_planned_scans_are_not_actual_coverage_evidence():
    world = OfflineRuleWorld([Source(1, (800, 100), 1300, None), Source(2, (500, 700), 1300, None)], 361423001)
    policy = SymmetryLayoutPolicy(world.port(), 'q3', seven_network(), 'E_joint', 'grid7')
    policy.scan_unknown(np.zeros(2), force=True)
    assert len(policy.coverage.observations) == 1
    assert all(action['position'] == [0.0, 0.0] for action in world.trace)
    assert policy.rotation_events
    event = policy.rotation_events[0]
    assert event['selected_proxy_meters'] <= event['original_proxy_meters'] + 1e-8
    trial = copy.deepcopy(policy.coverage)
    for station in policy.route[1:]:
        trial.observe_absence(station)
    assert trial.empty and not policy.coverage.empty


def test_full_mission_clears_all_sources():
    scene = next(scene for scene in runner.experiment.make_scenes(361423100, 'unit', [10]) if scene['problem'] == 'q3' and scene['profile'] == 'random')
    row, _trace = runner.evaluate(scene, 'joint_phase')
    assert row['success'], row['failure']
    assert row['cleared_count'] == 10


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match='Unknown symmetry-layout mode'):
        SymmetryLayoutPolicy(mode='unsupported')
