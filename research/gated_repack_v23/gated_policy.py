from __future__ import annotations

import numpy as np
from adaptive_policy import AdaptiveSourceCoverPolicy
from cover_planner import AdaptiveCoverPlanner
from geometry import open_route, route_length
from plan_geometry import disk_polygon


MODES = ('identity', 'repack', 'repack_scan')


class GatedCoverPlanner(AdaptiveCoverPlanner):
    def price(self, points):
        targets = [point for _state, point in self.policy.eligible_targets(points)]
        positions = [*targets, *points]
        itinerary = open_route(np.asarray(positions), self.policy.position) if positions else np.empty((0, 2))
        return float(route_length(itinerary, self.policy.position) / 5
                     + 6 * len(self.policy.unknown_channels()) * len(points))


class GatedRepackPolicy(AdaptiveSourceCoverPolicy):
    def __init__(self, *args, mode='repack'):
        if mode not in MODES:
            raise ValueError('Unknown gated-repack mode')
        super().__init__(*args, mode='source_repack')
        self.gated_mode = mode
        self.cover_planner = GatedCoverPlanner(self)

    def eligible_targets(self, remaining):
        targets = []
        for state in self.states.values():
            if state.status not in ('DETECTED', 'NEAR'):
                continue
            center, radius = (state.near_position, 0.0) if state.status == 'NEAR' else self.center_radius(state)
            if radius <= self.options.get('target_radius', 130) or not remaining or self.discovery_done:
                targets.append((state, center))
        return targets

    def run(self):
        self.scan_unknown(self.position, force=True)
        self.pending_scans = [point.copy() for point in self.route[1:]]
        self.stats['outer_first'] = False
        for _iteration in range(150):
            unresolved = [state for state in self.states.values() if state.status in ('DETECTED', 'NEAR')]
            if self.discovery_done and not unresolved:
                self.stats.update({'adaptive_relocations': self.cover_planner.relocations,
                                   'adaptive_prunes': self.cover_planner.prunes,
                                   'adaptive_certificates': self.cover_planner.certificates_checked})
                return self.result()
            if not self.pending_scans and not self.discovery_done:
                self.finish_discovery('full_route')
            if not self.discovery_done:
                relevant = [point for point in self.pending_scans
                            if not self.coverage.region.intersection(disk_polygon(point, 999.9)).is_empty]
                self.stats['coverage_skips'] += len(self.pending_scans) - len(relevant)
                self.pending_scans = relevant
                if not relevant:
                    if not self.coverage.empty:
                        raise ValueError('Relevant stations removed before complete coverage')
                    self.finish_discovery('coverage')
                elif self.gated_mode != 'identity':
                    self.refine_pending()
            stops = [('target', state.channel, point) for state, point in self.eligible_targets(self.pending_scans)]
            if not self.discovery_done:
                stops.extend(('station', index, point) for index, point in enumerate(self.pending_scans))
            if not stops:
                raise ValueError('No continuation before certified completion')
            positions = np.asarray([stop[2] for stop in stops])
            itinerary = open_route(positions, self.position)
            selected = int(np.argmin(np.linalg.norm(positions - itinerary[0], axis=1)))
            kind, identifier, point = stops[selected]
            if kind == 'target':
                self.stats['joint_targets'] += 1
                self.localize(self.states[identifier])
                if self.gated_mode == 'repack_scan':
                    self.refine_pending()
                    self.try_current_scan()
            else:
                was_pilot = self.pilot_pending
                self.scan_unknown(point, force=True)
                self.stats['visited_stations'] += 1
                self.stats['adaptive_executed_nonbaseline_scans'] += int(
                    not any(np.linalg.norm(point - original) < 1e-5 for original in self.route))
                self.reuse_directions()
                if was_pilot and not self.pilot_pending:
                    self.pending_scans = [station.copy() for station in self.route[1:]
                                          if not any(np.array_equal(station, observed) for observed in self.coverage.observations)]
                else:
                    self.pending_scans.pop(identifier)
                self.cover_events.append({'kind': 'planned_scan', 'point': point.tolist(), 'clock': self.virtual_seconds,
                                         'remaining_count': len(self.pending_scans),
                                         'actual_observations': len(self.coverage.observations)})
        raise ValueError('Gated-repack action limit exceeded')
