from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_scan_benchmark import ScanEconomyPolicy
from geometry import seven_network
from offline_benchmark import OfflineRuleWorld


def make_policy(mode):
    world = OfflineRuleWorld([], 254511101)
    policy = ScanEconomyPolicy(world.port(), 'q3', seven_network(), 'E_joint', 'grid7', mode=mode)
    return world, policy


def test_suppressed_scan_does_not_add_evidence_or_movement():
    world, policy = make_policy('station_only')
    region = policy.coverage.region.wkb
    policy.scan_unknown([1200, 200], force=False)
    assert world.trace == []
    assert world.virtual_seconds == 0
    assert policy.coverage.region.wkb == region
    assert policy.coverage.observations == []


def test_proposed_replacement_uses_copy_and_requires_actual_scan():
    world, policy = make_policy('certified_reuse')
    policy.scan_unknown([0, 0], force=True)
    policy.scan_unknown(policy.route[1], force=True)
    assert not policy.pilot_pending
    destination = policy.planned_positions()[0]
    before = (policy.coverage.region.wkb, len(world.trace), world.virtual_seconds)
    proposal = policy.replacement(destination)
    assert proposal is not None
    assert (policy.coverage.region.wkb, len(world.trace), world.virtual_seconds) == before
    policy.scan_unknown(destination, force=False)
    assert len(world.trace) > before[1]
    assert tuple(proposal[0]) in policy.skipped_positions
    after = (len(world.trace), world.virtual_seconds, policy.coverage.region.wkb)
    policy.scan_unknown(proposal[0], force=True)
    assert (len(world.trace), world.virtual_seconds, policy.coverage.region.wkb) == after


def test_close_by_measurement_jitter_is_not_used_for_replacement():
    _world, policy = make_policy('certified_reuse')
    policy.scan_unknown([0, 0], force=True)
    policy.scan_unknown(policy.route[1], force=True)
    assert policy.replacement(policy.position + [0.001, 0]) is None


def test_empty_world_finishes_with_actual_coverage_in_both_modes():
    for mode in ('station_only', 'certified_reuse'):
        world, policy = make_policy(mode)
        result = policy.run()
        assert policy.coverage.empty
        assert result['virtual_seconds'] == world.virtual_seconds
        assert set(result['declared_absent']) == set(range(1, 21))
