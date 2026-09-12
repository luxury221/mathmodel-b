from __future__ import annotations

import unittest

import numpy as np
from candidate import CandidatePolicy, continuation_cost, continuation_plan
from geometry import dual_ring_network, seven_network
from offline_benchmark import OfflineRuleWorld, Source
from plan_geometry import ClearPlan, conservative_cover_check, fallback_plan
from plan_policy import PlanPolicy
from shapely.geometry import Polygon


class CandidateTests(unittest.TestCase):
    def test_original_station_routes_unchanged(self):
        for problem, stations, variant, network in (
            ('q3', seven_network(), 'E_joint', 'grid7'),
            ('q4', dual_ring_network(), 'F_route', 'dual21'),
        ):
            world = OfflineRuleWorld([], 1)
            baseline = PlanPolicy(world.port(), problem, stations, variant, network)
            for mode in ('baseline', 'prune', 'joint', 'combined'):
                candidate = CandidatePolicy(world.port(), problem, stations, variant, network, mode)
                np.testing.assert_array_equal(candidate.route, baseline.route)

    def test_continuation_bound_covers_every_success_prefix(self):
        centers = np.array([[50.0, 25.0], [70.0, 10.0], [90.0, 0.0]])
        start, next_station = np.zeros(2), np.array([200.0, 0.0])
        bound = continuation_cost(centers, start, next_station)
        for prefix_length in range(1, 4):
            prefix = centers[:prefix_length]
            path = np.linalg.norm(np.diff(np.vstack((start, prefix)), axis=0), axis=1).sum()
            extra = (path + np.linalg.norm(prefix[-1] - next_station) - np.linalg.norm(start - next_station)) / 5
            self.assertLessEqual(extra + 3 * (prefix_length - 1) + 5, bound + 1e-9)

    def test_plan_reordering_preserves_certificate(self):
        region = Polygon([(-30, -3), (30, -3), (30, 3), (-30, 3)])
        plan = ClearPlan(np.array([[-15.0, 0.0], [15.0, 0.0]]), 100, True, 'test')
        result = continuation_plan(plan, np.array([50.0, 0.0]), np.array([-60.0, 0.0]))
        self.assertTrue(conservative_cover_check(region, result.centers))
        self.assertLessEqual(continuation_cost(result.centers, np.array([50.0, 0.0]), np.array([-60.0, 0.0])),
                             continuation_cost(plan.centers, np.array([50.0, 0.0]), np.array([-60.0, 0.0])))

    def test_safe_empty_circle_is_skipped(self):
        world = OfflineRuleWorld([Source(1, (0.0, 0.0), 1000, None)], 1)
        policy = CandidatePolicy(world.port(), 'q3', seven_network(), 'E_joint', 'grid7', 'prune')
        state = policy.states[1]
        state.region = Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])
        state.status = 'DETECTED'
        plan = ClearPlan(np.array([[-500.0, 0.0], [0.0, 0.0]]), 500, True,
                         'rectangle_partition_circumradius_bound')
        policy.execute_clear_plan(state, plan)
        self.assertEqual(policy.stats['pruned_centers'], 1)
        self.assertEqual(len(world.trace), 1)
        self.assertEqual(state.status, 'CLEARED')

    def test_fallback_keeps_finite_clearance_coverage(self):
        region = Polygon([(-80, -9), (80, -9), (80, 9), (-80, 9)])
        for position in ((-80.0, -9.0), (-30.0, 8.0), (20.0, 0.0), (80.0, 9.0)):
            world = OfflineRuleWorld([Source(1, position, 1000, None)], 2)
            policy = CandidatePolicy(world.port(), 'q3', seven_network(), 'E_joint', 'grid7', 'prune')
            state = policy.states[1]
            state.region = region
            state.status = 'DETECTED'
            plan = fallback_plan(region, 0, policy.position)
            self.assertTrue(conservative_cover_check(region, plan.centers))
            policy.execute_clear_plan(state, plan)
            self.assertFalse(world.remaining)
            self.assertLessEqual(len(world.trace), len(plan.centers))

    def test_directional_negative_does_not_delete_position(self):
        world = OfflineRuleWorld([Source(1, (0.0, 0.0), 1000, 180.0)], 3)
        policy = CandidatePolicy(world.port(), 'q4', dual_ring_network(), 'F_route', 'dual21')
        before = policy.states[1].region
        response = policy.measure(1, np.array([100.0, 0.0]))
        self.assertEqual(response['result'], 'no_signal')
        self.assertTrue(before.equals(policy.states[1].region))
        self.assertEqual(len(policy.states[1].negatives), 1)

    def test_repeated_measurement_uses_cache(self):
        world = OfflineRuleWorld([Source(1, (500.0, 0.0), 1000, None)], 4)
        policy = CandidatePolicy(world.port(), 'q3', seven_network(), 'E_joint', 'grid7')
        first = policy.measure(1, np.zeros(2))
        second = policy.measure(1, np.zeros(2))
        self.assertEqual(first, second)
        self.assertEqual(len(world.trace), 1)


if __name__ == '__main__':
    unittest.main()
