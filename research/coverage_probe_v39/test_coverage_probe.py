from __future__ import annotations

import math

import numpy as np
import pytest
import run_coverage as runner
from coverage_planner import ProbeCommitment
from coverage_policy import CoverageProbePolicy
from geometry import dual_ring_network
from offline_benchmark import OfflineRuleWorld, Source
from plan_geometry import disk_polygon, initial_outer_region, positive_update
from probe_constraints import certify_probe, missing_heading, solve_point
from shaped_geometry import certified_shaped_probe


def region_fixture():
    region = positive_update(initial_outer_region(), [0, 0], 0).difference(disk_polygon([0, 0], 700))
    original = certified_shaped_probe(region, [0, 0], 0, 0.7, 1.0)
    assert original is not None
    return region, original


def policy_fixture(mode='coverage_nudge'):
    world = OfflineRuleWorld([Source(1, (1500.0, 0.0), 1500.0, 270.0)], 401739251)
    policy = CoverageProbePolicy(world.port(), 'q4', dual_ring_network(), 'F_route', 'dual21', mode=mode)
    return policy, world


def prepared_commitment():
    policy, world = policy_fixture()
    policy.common_clear(policy.position, all_unresolved=True)
    policy.scan_unknown(policy.position, force=True)
    policy.measure(1, np.array([500.0, -500.0]), active=True)
    state = policy.states[1]
    anchor, bearing = state.positives[0]
    original = certified_shaped_probe(state.region, anchor, bearing, 0.7, 1.0)
    assert original is not None
    certificate = certify_probe(state.region, anchor, policy.route[2], original.candidates[1], bearing)
    assert certificate is not None
    kept = tuple(index for index in policy.remaining_stations if index != 2)
    proposal = ProbeCommitment(1, 2, kept, certificate, original.candidates[0], 0, 0, 0)
    return policy, world, proposal


def test_asymmetric_probe_has_full_vertex_proof_and_heading_recovery():
    region, original = region_fixture()
    certificate = certify_probe(region, [0, 0], [560, 140], original.candidates[1], 0)
    assert certificate is not None
    headings = np.deg2rad(np.arange(0, 360, 0.25))
    directions = np.column_stack((np.cos(headings), np.sin(headings)))
    failed_first = 0
    for distance in (700.1, 1000.0, 1499.0):
        for error in (-1.005, 0.0, 1.005):
            angle = math.radians(error)
            source = distance * np.array([math.cos(angle), math.sin(angle)])
            active_headings = directions[directions @ (-source) >= 0]
            responses = active_headings @ (certificate.candidates - source).T >= 0
            assert np.all(responses.any(axis=1))
            failed_first += int((~responses[:, 0]).sum())
            assert np.all(np.linalg.norm(certificate.candidates - source, axis=1) < distance)
    assert failed_first > 0


@pytest.mark.parametrize('point', [[500, -200], [900, 200], [1500, 1500]])
def test_invalid_probe_is_rejected(point):
    region, original = region_fixture()
    assert certify_probe(region, [0, 0], point, original.candidates[1], 0) is None


def test_convex_generation_enforces_new_coverage_constraint_without_mutation():
    region, original = region_fixture()
    before = region.wkb
    positions = np.array([[-450.0, 0.0]])
    headings = np.array([[1.0, 0.0]])
    certificate, status = solve_point(region, [0, 0], original.candidates[0], original.candidates[1], 0, positions, headings)
    assert status == 'optimal' and certificate is not None
    assert np.linalg.norm(certificate.candidates[0] - original.candidates[0]) > 1
    assert np.linalg.norm(certificate.candidates[0] - positions[0]) <= 999.9
    assert (certificate.candidates[0] - positions[0]) @ headings[0] >= 0
    assert region.wkb == before


def test_impossible_coverage_constraint_is_rejected():
    region, original = region_fixture()
    certificate, _status = solve_point(region, [0, 0], original.candidates[0], original.candidates[1], 0,
                                        [[-2000.0, 0.0]], [[1.0, 0.0]])
    assert certificate is None


def test_missing_heading_is_a_real_halfplane_gap():
    heading = missing_heading([0, 0], [[100, 0], [100, 100]])
    assert heading is not None and np.all(np.array([[100, 0], [100, 100]]) @ heading < 0)
    assert missing_heading([0, 0], [[100, 0], [0, 100], [-100, 0], [0, -100]]) is None


def test_real_scan_precedes_deletion_and_failed_first_retains_mirror():
    policy, world, proposal = prepared_commitment()
    before = policy.remaining_stations.copy()
    policy.execute_transit(proposal.certificate.candidates[0], proposal)
    event = policy.coverage_commitments[-1]
    assert event['outcome'] == 'deleted_after_real_scan'
    assert event['remaining_after_scan'] == before
    assert 2 not in policy.remaining_stations
    assert event['first_failed'] and 1 in policy.pending_probes
    mirror = policy.pending_probes[1].copy()
    policy.execute_stop(('v_mirror', 1, mirror, None))
    assert 1 not in policy.pending_probes
    assert any(action['channel'] == 1 and action['position'] == mirror.tolist()
               and action['response']['result'] in ('direction', 'near') for action in world.trace)
    runner.baseline.run_scan_benchmark.audit_trace(world.trace, policy.virtual_seconds)


def test_failed_scan_keeps_station_and_mirror(monkeypatch):
    policy, _world, proposal = prepared_commitment()
    def failing_scan(*arguments, **keywords):
        raise RuntimeError('injected scan failure')
    monkeypatch.setattr(policy, 'scan_unknown', failing_scan)
    with pytest.raises(RuntimeError, match='scan failure'):
        policy.execute_transit(proposal.certificate.candidates[0], proposal)
    assert 2 in policy.remaining_stations and 1 in policy.pending_probes
    assert not policy.pair_active
    assert policy.coverage_commitments[-1]['outcome'] == 'failed_no_deletion'


def test_identity_is_action_exact():
    scene = next(scene for scene in runner.experiment.make_scenes(411839273, 'unit', [10])
                 if scene['problem'] == 'q4' and scene['profile'] == 'random')
    previous, trace = runner.evaluate(scene, 'previous')
    identity, same = runner.evaluate(scene, 'identity')
    assert previous['success'] and identity['success'] and trace == same


@pytest.mark.parametrize('mode', ['station_bind', 'coverage_nudge'])
def test_complete_mission_clears_all_sources(mode):
    scene = next(scene for scene in runner.experiment.make_scenes(411839273, 'unit', [10])
                 if scene['problem'] == 'q4' and scene['profile'] == 'random')
    row, _trace = runner.evaluate(scene, mode)
    assert row['success'], row['failure']
    assert row['cleared_count'] == 10


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match='Unknown coverage-probe mode'):
        policy_fixture('unsupported')
