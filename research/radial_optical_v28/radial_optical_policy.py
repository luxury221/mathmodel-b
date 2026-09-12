from __future__ import annotations

import numpy as np
from anchored_policy import original_ordered_route
from complete_network import radio_certificate
from optical_policy import SelectiveOpticalPolicy
from patrol_planner import CertifiedPatrol


MODES = ('network_only', 'sentinel', 'radio_first')


def select_network():
    candidates = []
    for inner in (999.0, 997.5, 995.0):
        for outer in (1869.0, 1868.0, 1867.0, 1866.0, 1865.5):
            route = original_ordered_route()
            route[1:9] *= inner / 1000
            route[9:] *= outer / 1870
            candidates.append({'inner_radius': inner, 'outer_radius': outer, 'stations': route.tolist(),
                               'squared_radial_change': (1000 - inner)**2 + (1870 - outer)**2,
                               **radio_certificate(route)})
    eligible = [row for row in candidates if row['complete_without_optical']]
    selected = min(eligible, key=lambda row: row['squared_radial_change']) if eligible else None
    return {'selected': selected, 'candidates': candidates, 'public_geometry_only': True}


class RadialOpticalPolicy(SelectiveOpticalPolicy):
    def __init__(self, port, problem, stations, variant, network, mode='sentinel'):
        if mode not in (*MODES, 'identity'):
            raise ValueError('Unknown radial optical mode')
        super().__init__(port, problem, stations, variant, 'dual21', mode='network_only' if mode == 'identity' else mode)
        if mode != 'identity':
            planned = np.asarray(stations)
            if planned.shape != self.route.shape or not np.array_equal(planned[0], self.route[0]):
                raise ValueError('Radial network changed origin or node count')
            for selected_slice in (slice(1, 9), slice(9, 21)):
                ratio = np.linalg.norm(planned[selected_slice][0]) / np.linalg.norm(self.route[selected_slice][0])
                if not np.allclose(planned[selected_slice], self.route[selected_slice] * ratio, rtol=0, atol=1e-9):
                    raise ValueError('Radial network changed angles or node order')
            self.route = planned.copy()
        self.stations = self.route.copy()
        self.patrol = CertifiedPatrol(self)
        self.stats.update({'inner_radius': float(np.linalg.norm(self.route[1])),
                           'outer_radius': float(np.linalg.norm(self.route[9])), 'preserved_patrol_order': True})

    def finish_discovery(self, reason):
        if reason == 'full_route':
            if not self.coverage.empty:
                raise ValueError('Route exhaustion cannot replace actual complete coverage')
            reason = 'coverage'
        return super().finish_discovery(reason)
