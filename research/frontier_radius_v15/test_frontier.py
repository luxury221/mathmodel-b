from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_frontier_benchmark import FrontierPolicy
from geometry import seven_network
from offline_benchmark import OfflineRuleWorld


@pytest.mark.parametrize('mode', ['pilot1700', 'pilot1725', 'pilot1725_station_only'])
def test_frontier_expansion_preserves_complete_continuous_patrol(mode):
    world = OfflineRuleWorld([], 256511101)
    policy = FrontierPolicy(world.port(), 'q3', seven_network(), 'E_joint', 'grid7', mode=mode)
    policy.scan_unknown([0, 0], force=True)
    policy.scan_unknown(policy.route[1], force=True)
    assert policy.stats['radial_patrol_enabled']
    trial = copy.deepcopy(policy.coverage)
    for point in policy.route:
        trial.observe_absence(point)
    assert trial.empty
    assert policy.stats['radial_patrol_radius'] == (1700 if mode == 'pilot1700' else 1725)


def test_frontier_empty_world_requires_actual_coverage_and_time():
    world = OfflineRuleWorld([], 256511102)
    policy = FrontierPolicy(world.port(), 'q3', seven_network(), 'E_joint', 'grid7')
    result = policy.run()
    assert policy.coverage.empty
    assert result['virtual_seconds'] == world.virtual_seconds
    assert set(result['declared_absent']) == set(range(1, 21))
