from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import run_paired as runner
from geometry import dual_ring_network
from offline_benchmark import OfflineRuleWorld, Source
from paired_planner import PairCommitment
from paired_policy import PairedScanPolicy


def policy_fixture(mode='paired'):
    sources = [Source(1, (1000.0, 0.0), 1300.0, None),
               Source(2, (float(1000 / np.sqrt(2)), float(1000 / np.sqrt(2))), 1300.0, None)]
    world = OfflineRuleWorld(sources, 361425001)
    return PairedScanPolicy(world.port(), 'q4', dual_ring_network(), 'F_route', 'dual21', mode=mode), world


def prepared_pair():
    policy, world = policy_fixture()
    policy.common_clear(policy.position, all_unresolved=True)
    policy.scan_unknown(policy.position, force=True)
    for channel in (1, 2):
        policy.measure(channel, policy.route[channel])
        assert policy.states[channel].status == 'NEAR'
    stops = tuple(('near', channel, policy.route[channel].copy(), None) for channel in (1, 2))
    proposal = PairCommitment(stops, (1, 2), tuple(range(3, 21)), 0.0, 0.0)
    return policy, world, proposal


def test_identity_is_action_exact():
    scene = next(scene for scene in runner.experiment.make_scenes(381738127, 'unit', [10])
                 if scene['problem'] == 'q4' and scene['profile'] == 'random')
    previous, trace = runner.evaluate(scene, 'previous')
    identity, same = runner.evaluate(scene, 'identity')
    assert previous['success'] and identity['success']
    assert trace == same


def test_planned_coverage_does_not_become_actual_evidence():
    policy, world, proposal = prepared_pair()
    observations = [point.copy() for point in policy.coverage.observations]
    area = policy.coverage.area
    trace = list(world.trace)
    assert policy.pair_planner.complete(proposal.stops, proposal.kept).empty
    assert not policy.pair_planner.complete(proposal.stops[:1], proposal.kept).empty
    assert not policy.pair_planner.complete(proposal.stops[1:], proposal.kept).empty
    assert len(policy.coverage.observations) == len(observations)
    assert all(np.array_equal(first, second) for first, second in zip(observations, policy.coverage.observations))
    assert policy.coverage.area == area and world.trace == trace
    assert policy.remaining_stations == list(range(1, 21))


def test_stations_delete_only_after_both_real_scans(monkeypatch):
    policy, world, proposal = prepared_pair()
    def forbidden_single_replacement():
        raise AssertionError('Single replacement must be suspended within the pair')
    monkeypatch.setattr(policy.patrol, 'replace_from_current_position', forbidden_single_replacement)
    policy.execute_transit(proposal.stops[0][2], proposal)
    event = policy.commitment_events[-1]
    assert event['outcome'] == 'deleted_after_real_scans'
    assert len(event['checkpoints']) == 2
    assert all(point['actual_scan_recorded'] and point['remaining'] == list(range(1, 21)) for point in event['checkpoints'])
    assert policy.remaining_stations == list(range(3, 21))
    assert not policy.pair_active and policy.stats['pair_station_deletions'] == 2
    assert not world.remaining
    runner.run_scan_benchmark.audit_trace(world.trace, policy.virtual_seconds)


def test_joint_planner_selects_only_a_continuously_complete_pair(monkeypatch):
    policy, world, proposal = prepared_pair()
    stops = [*proposal.stops, *(('fixed', index, policy.route[index], None) for index in policy.remaining_stations)]
    monkeypatch.setattr(policy.pair_planner, 'proposed_price', lambda *arguments: -100000.0)
    trace = list(world.trace)
    selected = policy.pair_planner.choose(stops)
    assert selected is not None and selected.removed == (1, 2)
    assert policy.pair_planner.complete(selected.stops, selected.kept).empty
    assert world.trace == trace and policy.remaining_stations == list(range(1, 21))


def test_finite_witness_pass_never_substitutes_for_continuous_certificate(monkeypatch):
    policy, _world, proposal = prepared_pair()
    stops = [*proposal.stops, *(('fixed', index, policy.route[index], None) for index in policy.remaining_stations)]
    monkeypatch.setattr(policy.pair_planner, 'proposed_price', lambda *arguments: -100000.0)
    monkeypatch.setattr(policy.pair_planner, 'complete', lambda *arguments: SimpleNamespace(empty=False))
    assert policy.pair_planner.choose(stops) is None
    assert policy.pair_planner.decisions[-1]['witness_joint_passes'] > 0
    assert policy.pair_planner.decisions[-1]['continuous_checks'] > 0
    assert policy.remaining_stations == list(range(1, 21))


def test_failed_second_scan_retains_fallback(monkeypatch):
    policy, _world, proposal = prepared_pair()
    original_scan = policy.scan_unknown
    calls = []
    def failing_scan(point, force=False):
        calls.append(point.copy())
        if len(calls) == 2:
            raise RuntimeError('injected second scan failure')
        return original_scan(point, force=force)
    monkeypatch.setattr(policy, 'scan_unknown', failing_scan)
    with pytest.raises(RuntimeError, match='second scan'):
        policy.execute_transit(proposal.stops[0][2], proposal)
    assert policy.remaining_stations == list(range(1, 21))
    assert policy.commitment_events[-1]['outcome'] == 'failed_no_deletion'
    assert not policy.pair_active


def test_cleared_target_still_requires_committed_scan():
    policy, world, proposal = prepared_pair()
    assert policy.clear(1, proposal.stops[0][2])
    trace_start = len(world.trace)
    policy.execute_transit(proposal.stops[0][2], proposal)
    assert policy.commitment_events[-1]['outcome'] == 'deleted_after_real_scans'
    assert any(action['action'] == 'measure' and action['position'] == proposal.stops[0][2].tolist()
               for action in world.trace[trace_start:])


def test_missing_mirror_does_not_consume_recovery_state():
    policy, world = policy_fixture()
    policy.measure(1, np.zeros(2))
    policy.pending_probes[1] = np.array([900.0, 80.0])
    trace = list(world.trace)
    policy.execute_committed_action(('v_mirror', 1, np.array([900.0, -80.0]), None))
    assert np.array_equal(policy.pending_probes[1], [900.0, 80.0])
    assert world.trace == trace


def test_single_physical_action_and_spacing_filter():
    policy, _world = policy_fixture()
    policy.scan_unknown(policy.position, force=True)
    multi = SimpleNamespace(certified=True, centers=np.array([[800.0, 0.0], [820.0, 0.0]]))
    stops = [('clear', 1, np.array([800.0, 0.0]), multi),
             ('probe', 2, np.array([1.0, 0.0]), None),
             ('probe', 3, np.array([850.0, 0.0]), None)]
    assert [stop[1] for stop in policy.pair_planner.candidates(stops)] == [3]


def test_price_includes_forced_prefix_and_both_scan_fees():
    policy, _world = policy_fixture()
    policy.remaining_stations[:] = [1, 2, 3]
    pair = (('probe', 1, np.array([100.0, 0.0]), None), ('probe', 2, np.array([100.0, 100.0]), None))
    stops = [*pair, ('fixed', 1, np.array([50.0, 50.0]), None),
             ('fixed', 2, np.array([1000.0, 0.0]), None), ('fixed', 3, np.array([1000.0, 0.0]), None)]
    expected = 200 / 5 + np.hypot(900, 100) / 5 + 6 * 7 * 4
    assert policy.pair_planner.proposed_price(pair, (1,), stops, 7) == pytest.approx(expected)


def test_complete_mission_clears_all_sources():
    scene = next(scene for scene in runner.experiment.make_scenes(381738127, 'unit', [10])
                 if scene['problem'] == 'q4' and scene['profile'] == 'random')
    row, _trace = runner.evaluate(scene, 'paired')
    assert row['success'], row['failure']
    assert row['cleared_count'] == 10


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match='Unknown paired-scan mode'):
        policy_fixture('unsupported')
