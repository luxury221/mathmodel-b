from __future__ import annotations

import copy
import numpy as np
from policy import JointSearchPolicy


class RadialPolicy(JointSearchPolicy):
    def __init__(self, *args, mode='radial1450'):
        super().__init__(*args, mode='adaptive')
        if self.problem != 'q3':
            raise ValueError('Radial ablation is only for omnidirectional Q3')
        self.pilot_enabled = mode.startswith('pilot')
        self.radial_radius = {'radial1300': 1300.0, 'radial1450': 1450.0, 'radial1600': 1600.0,
                              'pilot1450': 1450.0, 'pilot1600': 1600.0}[mode]
        self.origin_assessed = False
        self.pilot_pending = False
        self.stats.update({'radial_patrol_enabled': False, 'radial_patrol_radius': 1125.0})

    def scan_unknown(self, position, force=False):
        response = super().scan_unknown(position, force)
        if self.origin_assessed:
            if self.pilot_pending and any(np.linalg.norm(position - station) < 1e-8 for station in self.route[1:]):
                self.pilot_pending = False
                if not self.discovery_done:
                    bearings = [bearing for state in self.states.values() for point, bearing in state.positives
                                if np.array_equal(point, position)]
                    ordered = np.sort(np.mod(bearings, 360))
                    span = 360 - float(np.max(np.diff(np.r_[ordered, ordered[0] + 360]))) if len(ordered) else 360
                    self.stats['pilot_bearing_count'] = len(bearings)
                    self.stats['pilot_bearing_span'] = span
                    if not (len(bearings) >= 3 and span <= 20):
                        self.expand_patrol()
            return response
        self.origin_assessed = True
        if np.linalg.norm(position) > 1e-8:
            raise ValueError('Initial patrol decision requires actual origin observations')
        if self.known_count() > 2 or self.discovery_done:
            return response
        if self.pilot_enabled:
            self.pilot_pending = True
        else:
            self.expand_patrol()
        return response

    def expand_patrol(self):
        from plan_geometry import disk_polygon
        proposed = self.route.copy()
        for index, station in enumerate(proposed[1:], 1):
            observed = any(np.linalg.norm(station - point) < 1e-8 for point in self.scan_positions)
            irrelevant = self.coverage.region.intersection(disk_polygon(station, 999.9)).is_empty
            if not observed and not irrelevant:
                proposed[index] *= self.radial_radius / np.linalg.norm(station)
        certificate = copy.deepcopy(self.coverage)
        for station in proposed[1:]:
            certificate.observe_absence(station)
        if not certificate.empty:
            raise ValueError('Proposed outer patrol lacks continuous coverage')
        self.route = proposed
        self.stats['radial_patrol_enabled'] = True
        self.stats['radial_patrol_radius'] = self.radial_radius

    def finish_discovery(self, reason):
        if reason == 'full_route':
            if not self.coverage.empty:
                raise ValueError('Radial route ended without actual coverage')
            reason = 'coverage'
        return super().finish_discovery(reason)
