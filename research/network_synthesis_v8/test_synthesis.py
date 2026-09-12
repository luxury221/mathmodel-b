from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthesis import certify, counterexamples, initial_witnesses, weakest_heading, visibility_violation
from geometry import dual_ring_network
import numpy as np
from shapely.geometry import Polygon
from plan_geometry import disk_polygon, initial_outer_region
from geometry import ring


def test_original_network_remains_continuously_certified():
    assert certify(dual_ring_network()).empty


def test_weak_heading_is_really_unseen():
    receivers = np.array([[500, 0], [600, 200], [600, -200]])
    position = np.array([0, 0])
    heading = weakest_heading(position, receivers)
    assert heading is not None
    assert np.all((receivers - position) @ heading < 0)


def test_surrounding_receivers_have_no_blind_heading():
    assert weakest_heading(np.zeros(2), np.array([[500, 0], [-250, 400], [-250, -400]])) is None


def test_counterexamples_do_not_depend_on_source_cases():
    region = Polygon([[-5, -5], [5, -5], [5, 5], [-5, 5]])
    receivers = np.array([[500, 0], [600, 200], [600, -200]])
    examples = counterexamples(region, receivers)
    assert examples
    for violation, position, heading in examples:
        assert violation > 0
        assert visibility_violation(receivers, position[None, :], heading[None, :]).min() > 0


def test_witnesses_are_in_public_geometry_and_unit_headings():
    positions, headings = initial_witnesses()
    assert len(positions) == len(headings)
    assert np.linalg.norm(positions, axis=1).max() < 1801
    assert np.linalg.norm(positions, axis=1).min() > 19.99
    assert np.allclose(np.linalg.norm(headings, axis=1), 1)


def test_seven_and_eight_point_omni_rings_cover_without_origin_radio():
    for count, radius in ((7, 999.0), (8, 940.0)):
        region = initial_outer_region()
        for point in ring(radius, count):
            region = region.difference(disk_polygon(point, 999.9))
        assert region.is_empty
