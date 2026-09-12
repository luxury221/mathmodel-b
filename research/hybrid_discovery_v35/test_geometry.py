from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import Point, box

from geometry_hybrid import (certify_hybrid, dual_ring_network, mixed_price,
                             patch_remainder, residual_after_radio, witness_patches)
from plan_geometry import disk_polygon


def test_original_network_needs_no_additional_optical_patch():
    result = certify_hybrid(dual_ring_network())
    assert result['complete'] and result['optical_centers'] == []
    assert result['radio_remaining_area'] == 0


def test_small_but_nonempty_residual_is_actually_patched():
    region = box(123, 67, 123.00001, 67.00001)
    assert not region.is_empty
    centers = patch_remainder(region)
    assert centers is not None and len(centers) == 1
    assert region.difference(disk_polygon(centers[0], 19.99)).is_empty


def test_multiple_disconnected_fragments_all_remain_accounted_for():
    region = shapely.union_all([Point(100, 0).buffer(4), Point(-100, 0).buffer(3)])
    centers = patch_remainder(region)
    assert centers is not None and len(centers) == 2
    assert region.difference(shapely.union_all([disk_polygon(center, 19.99) for center in centers])).is_empty
    assert patch_remainder(region, cap=1) is None


def test_large_hole_is_rejected_without_area_truncation():
    region = Point(300, 300).buffer(150)
    assert patch_remainder(region) is None
    assert not region.is_empty


def test_collinear_radio_points_do_not_certify_a_nearby_off_axis_source():
    coverage = residual_after_radio(np.array([[0, 0], [-900, 0], [900, 0]]))
    assert coverage.region.covers(Point(100, 25))


def test_finite_witness_patch_cap_and_zero_case():
    assert witness_patches(np.empty((0, 2))).shape == (0, 2)
    assert witness_patches(np.array([[100, 0], [-100, 0]]), cap=1) is None
    assert len(witness_patches(np.array([[100, 0], [101, 0]]), cap=1)) == 1


def test_proxy_price_counts_optical_actions_and_travel():
    stations = np.array([[0.0, 0.0], [100.0, 0.0]])
    price = mixed_price(stations, np.array([[200.0, 0.0]]))
    assert price['route_length_meters'] == 200
    assert price['proxy_cost_meters'] == 650
