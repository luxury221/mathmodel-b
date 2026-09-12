from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'research/joint_search_v5'), str(ROOT / 'research/offline_validation')]
import numpy as np
from coverage_certificate import TriangleCoverage
from geometry import dual_ring_network, open_route, ring, route_length
from plan_geometry import hull_vertices, initial_outer_region


def make_layout(inner_count, outer_count, inner_radius, outer_radius, phase):
    return np.vstack((np.zeros((1, 2)), ring(inner_radius, inner_count), ring(outer_radius, outer_count, phase)))


def public_witnesses():
    points = [hull_vertices(initial_outer_region())]
    points.extend(ring(radius, 120, 1.0) for radius in (20.01, 100, 250, 500, 750, 900, 1000, 1100, 1250, 1500, 1650, 1750, 1799.9))
    return np.vstack(points)


def uncovered_witnesses(stations, witnesses):
    vectors = np.asarray(stations)[None, :] - np.asarray(witnesses)[:, None]
    visible = np.sum(vectors * vectors, axis=2) <= 999.9**2
    counts = visible.sum(axis=1)
    angles = np.mod(np.arctan2(vectors[:, :, 1], vectors[:, :, 0]), 2 * math.pi)
    ordered = np.sort(np.where(visible, angles, np.inf), axis=1)
    valid_pairs = np.arange(ordered.shape[1] - 1)[None, :] < counts[:, None] - 1
    safe = np.where(np.isfinite(ordered), ordered, 0.0)
    gaps = np.where(valid_pairs, np.diff(safe, axis=1), 0.0)
    last = safe[np.arange(len(counts)), np.maximum(0, counts - 1)]
    wrap = safe[:, 0] + 2 * math.pi - last
    largest = np.maximum(gaps.max(axis=1), wrap)
    return (counts < 3) | (largest >= math.pi - 1e-7)


def certify_layout(stations):
    coverage = TriangleCoverage()
    coverage.observe_clear_absence(np.zeros(2))
    for station in stations:
        coverage.observe_absence(station)
    return {'complete': coverage.empty, 'remaining_area': float(coverage.area),
            'remaining_geometry': coverage.region.wkt if not coverage.empty else None,
            'actual_execution_required': True}


def geometry_price(stations):
    itinerary = open_route(stations[1:])
    length = route_length(itinerary)
    return {'route_length_meters': length, 'proxy_cost_meters': length + 300 * (len(stations) - 1)}


def configurations():
    for inner_count in range(5, 11):
        for outer_count in range(10, 17):
            if 1 + inner_count + outer_count > 21:
                continue
            period = 360 / math.lcm(inner_count, outer_count)
            for inner_radius in range(750, 1201, 50):
                for margin in (2, 8, 25, 50, 90):
                    outer_radius = 1800 / math.cos(math.pi / outer_count) + margin
                    for phase_fraction in (0, 1 / 6, 1 / 3, 1 / 2):
                        yield {'inner_count': inner_count, 'outer_count': outer_count,
                               'inner_radius': inner_radius, 'outer_radius': outer_radius,
                               'phase': phase_fraction * period}
