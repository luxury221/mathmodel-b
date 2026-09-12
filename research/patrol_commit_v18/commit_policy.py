from __future__ import annotations

import numpy as np
from geometry import open_route
from motion_policy import JointMotionPolicy


class CommittedPatrolPolicy(JointMotionPolicy):
    def __init__(self, *args, mode='committed'):
        if mode != 'committed':
            raise ValueError('Unknown committed-patrol mode')
        super().__init__(*args, mode='terminal')
        stations = self.route[1:]
        ordered = open_route(stations, self.position)
        self.patrol_order = [1 + int(np.argmin(np.linalg.norm(stations - point, axis=1))) for point in ordered]
        self.due_channels = set()
        self.commit_events = []
        self.stats.update({'committed_deferred': 0, 'committed_due_actions': 0})

    def closest_segment(self, center, station_ids):
        path = np.vstack((self.position, self.route[station_ids]))
        vectors = path[1:] - path[:-1]
        squared = np.sum(vectors * vectors, axis=1)
        fractions = np.clip(np.sum((center - path[:-1]) * vectors, axis=1) / np.maximum(squared, 1e-10), 0, 1)
        projections = path[:-1] + fractions[:, None] * vectors
        return int(np.argmin(np.linalg.norm(projections - center, axis=1)))

    def scheduled_stops(self, stops):
        if self.discovery_done:
            return stops, set()
        fixed = {identifier: stop for stop in stops for kind, identifier, _point, _extra in [stop] if kind == 'fixed'}
        if not fixed:
            return stops, set()
        station_ids = [identifier for identifier in self.patrol_order if identifier in fixed]
        if set(station_ids) != set(fixed):
            raise ValueError('Committed patrol lost a planned station')
        targets = [stop for stop in stops if stop[0] not in ('fixed', 'optical')]
        overdue = [stop for stop in targets if stop[1] in self.due_channels]
        if overdue:
            self.stats['committed_due_actions'] += 1
            return overdue, set()
        current = []
        for stop in targets:
            state = self.states[stop[1]]
            center = state.near_position if state.status == 'NEAR' else self.center_radius(state)[0]
            if self.closest_segment(center, station_ids) == 0:
                current.append(stop)
            else:
                self.stats['committed_deferred'] += 1
        return [fixed[station_ids[0]], *current], {stop[1] for stop in current}

    def run(self):
        self.common_clear(self.position, all_unresolved=True)
        self.scan_unknown(self.position, force=True)
        for _iteration in range(400):
            self.check_discovery()
            if self.discovery_done and not any(state.status in ('DETECTED', 'NEAR') for state in self.states.values()):
                return self.result()
            stops = self.build_stops()
            if not stops:
                continue
            available, current_channels = self.scheduled_stops(stops)
            destinations = np.asarray([stop[2] for stop in available])
            itinerary = open_route(destinations, self.position)
            selected = int(np.argmin(np.linalg.norm(destinations - itinerary[0], axis=1)))
            stop = available[selected]
            if stop[0] == 'fixed':
                self.due_channels.update(current_channels)
            self.commit_events.append({'position': self.position.tolist(), 'clock': self.virtual_seconds,
                                       'action': stop[0], 'identifier': stop[1],
                                       'eligible_count': len(available), 'all_count': len(stops),
                                       'due_channels': sorted(self.due_channels)})
            self.execute_stop(stop)
            self.due_channels = {channel for channel in self.due_channels
                                 if self.states[channel].status in ('DETECTED', 'NEAR')}
        raise ValueError('Committed-patrol action limit exceeded')
