import unittest
from unittest.mock import patch

import numpy as np
from candidate_v2 import MODES, Q4OptimizationPolicy, shorter_station_order
from geometry import dual_ring_network, route_length
from offline_benchmark import OfflineRuleWorld, Source
from plan_geometry import geometry_contains


class CandidateV2Tests(unittest.TestCase):
    def make_policy(self, mode='scan'):
        world = OfflineRuleWorld([Source(1, (800.0, 100.0), 1000.0, 0.0)], 89110001)
        policy = Q4OptimizationPolicy(world.port(), 'q4', dual_ring_network(), 'F_route', 'dual21', mode)
        return world, policy

    def test_only_q4_and_frozen_primary_network(self):
        world, _ = self.make_policy()
        for problem, network, variant in (('q3', 'dual21', 'F_route'), ('q4', 'grid25', 'F_route'), ('q4', 'dual21', 'E_joint')):
            with self.assertRaises(ValueError):
                Q4OptimizationPolicy(world.port(), problem, dual_ring_network(), variant, network)

    def test_every_mode_keeps_all_original_station_coordinates_exactly(self):
        _, original = self.make_policy()
        expected = sorted(tuple(float(value).hex() for value in point) for point in original.route)
        for mode in MODES:
            _, candidate = self.make_policy(mode)
            actual = sorted(tuple(float(value).hex() for value in point) for point in candidate.route)
            self.assertEqual(actual, expected)
            self.assertEqual(len(candidate.route), 21)
            np.testing.assert_array_equal(candidate.route[0], (0, 0))

    def test_shorter_route_never_exceeds_original_static_length(self):
        _, original = self.make_policy()
        _, revised = self.make_policy('tour_scan')
        self.assertLessEqual(route_length(revised.route), route_length(original.route) + 1e-8)

    def test_shorter_route_is_deterministic_and_complete(self):
        _, policy = self.make_policy()
        coordinates = tuple(tuple(float(value) for value in point) for point in policy.route)
        first, second = shorter_station_order(coordinates), shorter_station_order(coordinates)
        self.assertEqual(first, second)
        self.assertEqual(sorted(first), list(range(21)))

    def test_directional_no_signal_does_not_delete_source_positions(self):
        _, policy = self.make_policy()
        original_area = policy.states[1].region.area
        policy.measure(1, np.array([0.0, 0.0]))
        self.assertEqual(policy.states[1].region.area, original_area)
        self.assertTrue(geometry_contains(policy.states[1].region, (800, 100)))

    def test_cached_measurement_has_no_second_physical_action(self):
        world, policy = self.make_policy()
        policy.measure(1, np.array([900.0, 100.0]))
        count = len(world.trace)
        policy.measure(1, np.array([900.0, 100.0]))
        self.assertEqual(len(world.trace), count)

    def test_reuse_is_zero_movement_and_does_not_repeat_at_same_position(self):
        world, policy = self.make_policy()
        policy.measure(1, np.array([900.0, 100.0]))
        policy.measure(2, np.array([900.0, 120.0]))
        position = policy.position.copy()
        policy.reuse_current_position()
        self.assertGreater(policy.stats['reuse_measures'], 0)
        np.testing.assert_array_equal(policy.position, position)
        count = len(world.trace)
        policy.reuse_current_position()
        self.assertEqual(len(world.trace), count)
        self.assertTrue(geometry_contains(policy.states[1].region, (800, 100)))

    def test_probe_budget_and_quota_are_bounded(self):
        from types import SimpleNamespace

        import candidate_v2
        world, policy = self.make_policy('probe_scan')
        policy.measure(1, np.array([900.0, 100.0]))
        state = policy.states[1]
        plan = SimpleNamespace(certificate_kind='rectangle_partition_circumradius_bound', centers=np.zeros((8, 2)),
                               worst_seconds=100.0, certified=True)
        measurement = SimpleNamespace(immediate_seconds=21.0)
        with patch.object(candidate_v2, 'minimax_measurement', return_value=measurement), \
                patch.object(candidate_v2.CandidatePolicy, 'execute_clear_plan') as execute:
            count = len(world.trace)
            policy.execute_clear_plan(state, plan)
            self.assertEqual(len(world.trace), count)
            self.assertEqual(policy.probe_counts[1], 0)
            execute.assert_called_once()
        policy.probe_counts[1] = 1
        with patch.object(candidate_v2, 'minimax_measurement') as minimax, \
                patch.object(candidate_v2.CandidatePolicy, 'execute_clear_plan'):
            policy.execute_clear_plan(state, plan)
            minimax.assert_not_called()


if __name__ == '__main__':
    unittest.main()
