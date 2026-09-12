from __future__ import annotations

import math

import numpy as np
from coverage_certificate import TriangleCoverage
from geometry import dual_ring_network, open_route, route_length


def radio_certificate(stations):
    coverage = TriangleCoverage()
    for station in stations:
        coverage.observe_absence(station)
    return {'complete_without_optical': coverage.empty, 'remaining_area': float(coverage.area),
            'remaining_geometry': None if coverage.empty else coverage.region.wkt}


def select_network():
    candidates = []
    for inner in (975.0, 985.0, 990.0, 995.0, 997.5, 999.0):
        for outer in (1800 / math.cos(math.pi / 12) + 2, 1870.0):
            for phase in (0.0, 15.0):
                stations = dual_ring_network(inner, outer)
                angle = math.radians(phase - 15.0)
                rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
                stations[9:] = stations[9:] @ rotation.T
                certificate = radio_certificate(stations)
                length = route_length(open_route(stations[1:]))
                candidates.append({'inner_radius': inner, 'outer_radius': outer, 'phase': phase,
                                   'stations': stations.tolist(), 'route_length_meters': length,
                                   'proxy_cost_meters': length + 6000, **certificate})
    eligible = [row for row in candidates if row['complete_without_optical']]
    selected = min(eligible, key=lambda row: row['proxy_cost_meters']) if eligible else None
    return {'selected': selected, 'candidates': candidates,
            'public_geometry_only': True, 'origin_optical_assumed': False}
