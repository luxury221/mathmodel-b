from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT / 'research/joint_search_v5')]
import experiment_v5
from radial_policy import RadialPolicy
from geometry import seven_network
from offline_benchmark import OfflineRuleWorld, Source
import numpy as np
import pytest


def fixture_policy(sources):
    world = OfflineRuleWorld(sources, 110711200)
    return RadialPolicy(world.port(), 'q3', seven_network(), 'E_joint', 'grid7', mode='pilot1600')


def test_pilot_waits_for_a_real_planned_station():
    angles = np.deg2rad([-20, 0, 20, 100, 120, 140, 160, 180, 200, 220])
    sources = [Source(index + 1, tuple(1790 * np.array([np.cos(angle), np.sin(angle)])), 1000, None)
               for index, angle in enumerate(angles)]
    policy = fixture_policy(sources)
    policy.scan_unknown([0, 0], force=True)
    assert policy.pilot_pending
    assert not policy.stats['radial_patrol_enabled']
    station = policy.route[1].copy()
    policy.scan_unknown(station, force=True)
    assert policy.stats['radial_patrol_enabled']
    assert np.array_equal(policy.route[1], station)
    certificate = copy.deepcopy(policy.coverage)
    for point in policy.route:
        certificate.observe_absence(point)
    assert certificate.empty


def test_compact_observed_bearings_preserve_the_old_route():
    sources = [Source(index + 1, (1790, float(index) - 5), 1000, None) for index in range(10)]
    policy = fixture_policy(sources)
    original = policy.route.copy()
    policy.scan_unknown([0, 0], force=True)
    policy.scan_unknown(policy.route[1], force=True)
    assert not policy.pilot_pending
    assert not policy.stats['radial_patrol_enabled']
    assert np.array_equal(policy.route, original)


def test_radial_termination_requires_executed_coverage():
    policy = fixture_policy([])
    with pytest.raises(ValueError, match='actual coverage'):
        policy.finish_discovery('full_route')
