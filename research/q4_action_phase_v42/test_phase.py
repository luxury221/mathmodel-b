from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import run_phase as runner
from geometry import dual_ring_network
from offline_benchmark import OfflineRuleWorld, Source
from phase_policy import ActionPhasePolicy, PHASES, first_clear_position, rotate_stations
from shapely.geometry import Point


def fixture(mode='action_phase', sources=None):
    if sources is None:
        sources = [Source(1, (850.0, 100.0), 1200.0, None)]
    world = OfflineRuleWorld(sources, 471842503)
    policy = ActionPhasePolicy(world.port(), 'q4', dual_ring_network(), 'F_route', 'dual21', mode=mode)
    return policy, world


def test_rotation_preserves_origin_distances_and_zero_exactness():
    points = dual_ring_network()
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    assert np.array_equal(rotate_stations(points, 0), points)
    for phase in PHASES:
        proposal = rotate_stations(points, phase)
        assert np.array_equal(proposal[0], points[0])
        assert np.allclose(np.linalg.norm(proposal[:, None] - proposal[None, :], axis=2), distances, atol=1e-9)
    assert np.array_equal(points, dual_ring_network())


def test_identity_is_full_mission_action_exact():
    scene = next(scene for scene in runner.experiment.make_scenes(481942519, 'unit', [10])
                 if scene['problem'] == 'q4' and scene['profile'] == 'random')
    original, trace = runner.evaluate(scene, 'previous')
    identity, same = runner.evaluate(scene, 'identity')
    assert original['success'] and identity['success']
    assert trace == same


def test_planning_keeps_actual_evidence_and_rebuilds_station_matrix():
    policy, world = fixture()
    policy.common_clear(np.zeros(2), all_unresolved=True)
    policy.scan_unknown(np.zeros(2), force=True)
    assert len(policy.coverage.observations) == 1
    assert len(policy.coverage.clear_observations) == 1
    assert all(action['position'] == [0.0, 0.0] for action in world.trace)
    assert len(policy.phase_events) == 1
    event = policy.phase_events[0]
    assert event['selected_proxy_meters'] <= event['original_proxy_meters'] + 1e-8
    assert np.array_equal(policy.patrol.matrix, np.array([policy.search_planner.visible(point) for point in policy.route]))
    trial = copy.deepcopy(policy.coverage)
    for point in policy.route[1:]:
        trial.observe_absence(point)
    assert trial.empty and not policy.coverage.empty
    assert policy.remaining_stations == list(range(1, 21))
    assert all(count == 0 for count in policy.active_counts.values())


def test_action_preview_does_not_measure_or_change_source_region():
    policy, world = fixture('identity')
    policy.common_clear(np.zeros(2), all_unresolved=True)
    policy.scan_unknown(np.zeros(2), force=True)
    before = (list(world.trace), policy.coverage.region.wkb, policy.states[1].region.wkb, policy.states[1].revision)
    targets = policy.action_targets()
    assert targets and targets[0]['channel'] == 1
    assert before == (world.trace, policy.coverage.region.wkb, policy.states[1].region.wkb, policy.states[1].revision)
    kind, destination, _extra = policy.next_target_action(policy.states[1])
    assert kind == targets[0]['kind']
    assert destination.tolist() == targets[0]['position']


def test_nearest_viable_clear_center_is_the_physical_entry():
    state = SimpleNamespace(region=Point(100, 0).buffer(5))
    plan = SimpleNamespace(centers=np.array([[1000.0, 0], [115.0, 0], [85.0, 0]]))
    assert np.array_equal(first_clear_position(state, plan, np.zeros(2)), [85.0, 0])
    with pytest.raises(ValueError, match='no viable'):
        first_clear_position(state, SimpleNamespace(centers=np.array([[1000.0, 0]])), np.zeros(2))


def test_no_detected_targets_keeps_original_layout():
    policy, _world = fixture(sources=[Source(1, (1750.0, 0.0), 1000.0, 0.0)])
    original = policy.route.copy()
    policy.common_clear(np.zeros(2), all_unresolved=True)
    policy.scan_unknown(np.zeros(2), force=True)
    assert not policy.phase_events and np.array_equal(policy.route, original)


def test_phase_selection_is_once_only():
    policy, _world = fixture()
    policy.common_clear(np.zeros(2), all_unresolved=True)
    policy.scan_unknown(np.zeros(2), force=True)
    original = policy.route.copy()
    policy.scan_unknown(policy.route[1], force=True)
    assert len(policy.phase_events) == 1 and np.array_equal(policy.route, original)


def test_off_origin_first_scan_cannot_rotate_past_observations():
    policy, _world = fixture()
    with pytest.raises(ValueError, match='first real origin scan'):
        policy.scan_unknown(np.array([100.0, 0.0]), force=True)


def test_unregistered_modes_are_rejected():
    with pytest.raises(ValueError, match='Unregistered'):
        fixture(mode='unchecked')
