from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_benchmark import JointCoverPolicy
from geometry import seven_network, dual_ring_network
from offline_benchmark import OfflineRuleWorld
import numpy as np


def test_future_plans_do_not_change_actual_evidence_or_clock():
    for problem in ('q3', 'q4'):
        world = OfflineRuleWorld([], 121311200)
        policy = JointCoverPolicy(world.port(), problem, seven_network() if problem == 'q3' else dual_ring_network(),
                                  'E_joint' if problem == 'q3' else 'F_route', 'grid7' if problem == 'q3' else 'dual21')
        policy.coverage.observe_clear_absence([0, 0])
        policy.coverage.observe_absence([0, 0])
        before = policy.coverage.region.wkb
        plan = policy.joint_planner.choose([])
        assert policy.joint_planner.complete(plan.scans).empty
        assert policy.coverage.region.wkb == before
        assert len(policy.coverage.observations) == 1
        assert world.virtual_seconds == 0
        assert world.trace == []
        assert plan.predicted_seconds <= plan.baseline_seconds + 1e-8


def test_known_action_remains_in_joint_itinerary():
    policy = JointCoverPolicy(OfflineRuleWorld([], 121311201).port(), 'q3', seven_network(), 'E_joint', 'grid7')
    policy.coverage.observe_clear_absence([0, 0])
    policy.coverage.observe_absence([0, 0])
    destination = np.array([1500.0, 300.0])
    plan = policy.joint_planner.choose([('probe', 1, destination, None)])
    assert any(np.array_equal(point, destination) for point in plan.itinerary)
    assert policy.joint_planner.complete(plan.scans).empty
