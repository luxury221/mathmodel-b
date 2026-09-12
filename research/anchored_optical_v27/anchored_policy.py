from __future__ import annotations

import numpy as np
from complete_network import radio_certificate
from geometry import ring
from optical_policy import SelectiveOpticalPolicy
from patrol_planner import CertifiedPatrol


MODES = ('network_only', 'sentinel', 'radio_first')


def original_ordered_route():
    return np.vstack(([0, 0], ring(1000, 8), ring(1870, 12, 315)))


def select_preserved_network():
    attempts = []
    for radius in (999.0, 997.5, 995.0, 990.0, 985.0, 975.0):
        route = original_ordered_route()
        route[1:9] *= radius / 1000
        certificate = radio_certificate(route)
        attempts.append({'inner_radius': radius, **certificate})
        if certificate['complete_without_optical']:
            return {'inner_radius': radius, 'stations': route.tolist(), 'attempts': attempts,
                    'preserves_outer_floating_coordinates': True, 'preserves_order': True}
    raise ValueError('No registered inner contraction has a radio-only continuous certificate')


class AnchoredOpticalPolicy(SelectiveOpticalPolicy):
    def __init__(self, port, problem, stations, variant, network, mode='sentinel'):
        if mode not in (*MODES, 'identity'):
            raise ValueError('Unknown anchored optical mode')
        super().__init__(port, problem, stations, variant, 'dual21', mode='network_only' if mode == 'identity' else mode)
        actual_route = self.route.copy()
        if mode != 'identity':
            planned = np.asarray(stations)
            if len(planned) != len(actual_route) or not np.array_equal(planned[[0, *range(9, 21)]], actual_route[[0, *range(9, 21)]]):
                raise ValueError('The proposed route changed the origin or frozen outer coordinates')
            actual_route[1:9] = planned[1:9]
            if not radio_certificate(actual_route)['complete_without_optical']:
                raise ValueError('Ordered replacement route lost radio-only coverage')
        self.route = actual_route
        self.stations = actual_route.copy()
        self.patrol = CertifiedPatrol(self)
        self.stats['inner_radius'] = float(np.linalg.norm(self.route[1]))
        self.stats['preserved_patrol_order'] = True

    def finish_discovery(self, reason):
        if reason == 'full_route':
            if not self.coverage.empty:
                raise ValueError('Route exhaustion cannot replace an actual complete coverage certificate')
            reason = 'coverage'
        return super().finish_discovery(reason)
