from __future__ import annotations

import math
import unittest

import numpy as np
from geometry import initial_region, update_bearing_polygon
from plan_geometry import (
    clear_failure_update,
    conservative_cover_check,
    geometry_contains,
    minimax_measurement,
    positive_update,
    reception_proxy,
    small_clear_plan,
    source_proxy_hypotheses,
)
from shapely.geometry import Polygon


class PlanGeometryTests(unittest.TestCase):
    def test_convex_constraint_with_near_duplicate_closing_vertices(self):
        position = (1322.289680818844, 1322.2896808188436)
        vertices = update_bearing_polygon(initial_region(), position, 164.13)
        updated = positive_update(Polygon(initial_region()), position, 164.13)
        self.assertTrue(updated.is_valid)
        self.assertFalse(updated.is_empty)
        for vertex in vertices:
            self.assertTrue(geometry_contains(updated, vertex))

    def test_two_circle_continuous_cover(self):
        region = Polygon([(-30, -3), (30, -3), (30, 3), (-30, 3)])
        plan = small_clear_plan(region, np.array([-50.0, 0.0]))
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.centers), 2)
        self.assertTrue(conservative_cover_check(region, plan.centers))

    def test_four_circle_square_cover(self):
        region = Polygon([(-20, -20), (20, -20), (20, 20), (-20, 20)])
        plan = small_clear_plan(region, np.array([0.0, 0.0]))
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.centers), 4)
        self.assertTrue(conservative_cover_check(region, plan.centers))

    def test_sample_vertices_are_not_a_cover_certificate(self):
        vertices = np.array([[0.0, 0.0], [60.0, 0.0], [30.0, 30 * math.sqrt(3)]])
        self.assertFalse(conservative_cover_check(Polygon(vertices), vertices))

    def test_clear_failure_removes_only_safe_inner_region(self):
        region = Polygon([(-40, -5), (40, -5), (40, 5), (-40, 5)])
        reduced = clear_failure_update(region, np.array([0.0, 0.0]))
        self.assertLess(reduced.area, region.area)
        self.assertTrue(geometry_contains(reduced, (30, 0)))
        self.assertFalse(geometry_contains(reduced, (0, 0)))

    def test_candidate_is_inside_full_conservative_reception_domain(self):
        region = Polygon(update_bearing_polygon(initial_region(), (0, 0), 0.0))
        plan = minimax_measurement(region, np.zeros(2), np.zeros(2), [np.zeros(2)])
        self.assertIsNotNone(plan)
        self.assertLessEqual(plan.max_receiving_distance, 999.900001)
        self.assertGreater(np.linalg.norm(plan.position), 1e-5)

    def test_mec_shortcut(self):
        region = Polygon([(-10, -10), (10, -10), (10, 10), (-10, 10)])
        plan = small_clear_plan(region, np.array([0.0, 0.0]))
        self.assertEqual(plan.certificate_kind, 'mec')
        self.assertEqual(plan.worst_seconds, 5.0)

    def test_negative_direction_does_not_delete_source_position(self):
        region = Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])
        positives = [(np.array([-100.0, 0.0]), 0.0)]
        negatives = [np.array([100.0, 0.0])]
        bank = source_proxy_hypotheses(region, positives, negatives, True)
        self.assertIsNotNone(bank)
        self.assertGreater(reception_proxy(bank, (-500, 0)), 0.9)
        self.assertLess(reception_proxy(bank, (500, 0)), 0.1)
        self.assertTrue(geometry_contains(region, (0, 0)))

    def test_joint_radius_heading_hypotheses_obey_all_observations(self):
        region = Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)])
        positives = [(np.array([-1100.0, 0.0]), 0.0)]
        negatives = [np.array([900.0, 0.0]), np.array([-1400.0, 0.0])]
        bank = source_proxy_hypotheses(region, positives, negatives, True)
        self.assertIsNotNone(bank)
        positions, headings, radii, weights = bank
        for observation_position, _bearing in positives:
            vector = observation_position - positions
            visible = (np.linalg.norm(headings, axis=1) == 0) | (np.sum(vector * headings, axis=1) >= 0)
            self.assertTrue(np.all(visible & (np.linalg.norm(vector, axis=1) <= radii)))
        for observation_position in negatives:
            vector = observation_position - positions
            visible = (np.linalg.norm(headings, axis=1) == 0) | (np.sum(vector * headings, axis=1) >= 0)
            self.assertFalse(np.any(visible & (np.linalg.norm(vector, axis=1) <= radii)))
        self.assertAlmostEqual(float(weights.sum()), 1.0)


if __name__ == '__main__':
    unittest.main()
