from __future__ import annotations

import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'research/annular_topology_v25'), str(ROOT / 'research/joint_search_v5'),
                str(ROOT / 'research/offline_validation')]
import numpy as np
import shapely
from coverage_certificate import TriangleCoverage
from geometry import dual_ring_network, open_route, route_length
from plan_geometry import disk_polygon, small_clear_plan
from topology import configurations, make_layout, public_witnesses, uncovered_witnesses


def layouts():
    seen = set()
    for index, parameters in enumerate(configurations()):
        stations = make_layout(**parameters)
        key = tuple(np.round(stations.ravel(), 8))
        if key in seen:
            continue
        seen.add(key)
        yield 'legacy_' + str(index), parameters, stations
    for radius in range(1000, 1051, 5):
        for margin in (5, 15, 30, 50):
            for phase in (0.0, 180 / 91):
                parameters = {'inner_count': 7, 'outer_count': 13, 'inner_radius': radius,
                              'outer_radius': 1800 / math.cos(math.pi / 13) + margin, 'phase': phase}
                stations = make_layout(**parameters)
                key = tuple(np.round(stations.ravel(), 8))
                if key in seen:
                    continue
                seen.add(key)
                yield 'offset_' + str(len(seen)), parameters, stations
    baseline = dual_ring_network()
    for index in range(1, len(baseline)):
        stations = np.delete(baseline, index, axis=0)
        yield 'delete_' + str(index), {'deleted_baseline_index': index}, stations


def witness_patches(points, cap=16):
    remaining = np.asarray(points, dtype=float).reshape((-1, 2))
    centers = []
    while len(remaining):
        if len(centers) >= cap:
            return None
        counts = np.sum(np.linalg.norm(remaining[:, None] - remaining[None, :], axis=2) <= 19.95, axis=1)
        center = remaining[int(np.argmax(counts))]
        centers.append(center.copy())
        remaining = remaining[np.linalg.norm(remaining - center, axis=1) > 19.95]
    return np.asarray(centers).reshape((-1, 2))


def mixed_price(stations, optical_centers):
    points = np.vstack((np.asarray(stations)[1:], np.asarray(optical_centers).reshape((-1, 2))))
    itinerary = open_route(points)
    length = route_length(itinerary)
    return {'route_length_meters': length,
            'proxy_cost_meters': length + 300 * (len(stations) - 1) + 150 * len(optical_centers)}


def residual_after_radio(stations):
    coverage = TriangleCoverage()
    coverage.observe_clear_absence(np.zeros(2))
    for point in stations:
        coverage.observe_absence(point)
    return coverage


def patch_remainder(region, cap=16):
    if region.is_empty:
        return np.empty((0, 2))
    parts = list(region.geoms) if hasattr(region, 'geoms') else [region]
    centers = []
    for part in sorted(parts, key=lambda component: component.area, reverse=True):
        if part.is_empty:
            continue
        plan = small_clear_plan(part, np.zeros(2), 4)
        if plan is None:
            return None
        centers.extend(point.copy() for point in plan.centers)
        if len(centers) > 64:
            return None
    disks = [disk_polygon(center, 19.99, 128) for center in centers]
    if not region.difference(shapely.union_all(disks)).is_empty:
        raise ValueError('Optical proposal does not cover the complete residual')
    for index in reversed(range(len(centers))):
        retained = disks[:index] + disks[index + 1:]
        if retained and region.difference(shapely.union_all(retained)).is_empty:
            centers.pop(index)
            disks.pop(index)
    if len(centers) > cap:
        return None
    return np.asarray(centers).reshape((-1, 2))


def certify_hybrid(stations, cap=16):
    coverage = residual_after_radio(stations)
    original_area = float(coverage.area)
    original_geometry = coverage.region.wkt
    patches = patch_remainder(coverage.region, cap)
    if patches is None:
        return {'complete': False, 'radio_remaining_area': original_area,
                'radio_remaining_geometry': original_geometry, 'optical_centers': None,
                'reason': 'bounded_optical_proposal_failed'}
    for center in patches:
        coverage.observe_clear_absence(center)
    if not coverage.empty:
        raise ValueError('Hybrid completion left a nonempty continuous remainder')
    return {'complete': True, 'radio_remaining_area': original_area,
            'radio_remaining_geometry': original_geometry, 'optical_centers': patches.tolist(),
            'remaining_area': float(coverage.area), 'actual_optical_execution_required': True,
            **mixed_price(stations, patches)}
