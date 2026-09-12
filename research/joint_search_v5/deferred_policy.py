from __future__ import annotations

import numpy as np
from geometry import open_route
from plan_geometry import reception_proxy, small_clear_plan
from policy_v5 import ContinuousPolicy
from shapely.geometry import Point


DEFERRED_MODES = ('deferred_v', 'deferred_center', 'deferred_outer')


class DeferredPolicy(ContinuousPolicy):
    def __init__(self, *args, mode='deferred_v'):
        self.deferred_mode = mode
        super().__init__(*args, mode='certified_fixed')
        self.stats.update({'deferred_targets': 0, 'future_bearing_opportunities': 0})

    @property
    def v_enabled(self):
        return self.deferred_mode != 'deferred_center'

    def future_bearing_opportunity(self, state, indices):
        center, radius = self.center_radius(state)
        if radius <= 60 or not indices:
            return False
        anchor = state.positives[-1][0]
        initial_vector = center - anchor
        bank = self.hypotheses(state) if self.problem == 'q4' else None
        for index in indices:
            station = self.route[index]
            if any(np.linalg.norm(station - previous) < 2 for previous in state.measured_positions):
                continue
            if state.region.distance(Point(station)) > 900:
                continue
            next_vector = center - station
            sine = abs(initial_vector[0] * next_vector[1] - initial_vector[1] * next_vector[0])
            sine /= max(1e-9, np.linalg.norm(initial_vector) * np.linalg.norm(next_vector))
            if sine < 0.35:
                continue
            if bank is not None and reception_proxy(bank, station) < 0.45:
                continue
            return True
        return False

    def run(self):
        self.common_clear(self.position, all_unresolved=True)
        self.scan_unknown(self.position, force=True)
        outer_first = self.deferred_mode == 'deferred_outer' and self.problem == 'q4' and self.known_count() <= 2
        self.stats['outer_first'] = outer_first
        for _iteration in range(180):
            self.check_discovery()
            unresolved = [state for state in self.states.values() if state.status in ('DETECTED', 'NEAR')]
            if self.discovery_done and not unresolved:
                return self.result()
            if not self.discovery_done:
                targets = [self.center_radius(state)[0] for state in unresolved if state.status == 'DETECTED']
                self.patrol.reduce(targets)
            available = [] if self.discovery_done else self.remaining_stations.copy()
            if outer_first and any(index >= 9 for index in available):
                available = [index for index in available if index >= 9]
            stops = [('fixed', index, self.route[index], None) for index in available]
            for state in unresolved:
                if state.status == 'NEAR':
                    stops.append(('near', state.channel, state.near_position, None))
                    continue
                plan = small_clear_plan(state.region, self.position, 4)
                if plan is not None:
                    stops.append(('clear', state.channel, plan.centers[0], plan))
                elif self.future_bearing_opportunity(state, available):
                    self.stats['deferred_targets'] += 1
                    self.stats['future_bearing_opportunities'] += 1
                else:
                    stops.append(('target', state.channel, self.center_radius(state)[0], None))
            if not stops:
                if self.discovery_done:
                    raise ValueError('Deferred policy lost its unresolved targets')
                self.fallback_discovery()
                continue
            destinations = np.asarray([stop[2] for stop in stops])
            itinerary = open_route(destinations, self.position)
            selected = int(np.argmin(np.linalg.norm(destinations - itinerary[0], axis=1)))
            action, identifier, destination, plan = stops[selected]
            if action == 'fixed':
                self.scan_unknown(destination, force=True)
                self.stats['visited_stations'] += 1
                self.remaining_stations.remove(identifier)
                self.reuse_directions()
            elif action == 'near':
                self.clear_near(self.states[identifier])
                self.reuse_stop()
            elif action == 'clear':
                self.execute_clear_plan(self.states[identifier], plan)
            else:
                self.stats['joint_targets'] += 1
                if self.v_enabled:
                    self.resolve_with_v(self.states[identifier])
                else:
                    self.localize(self.states[identifier])
        raise ValueError('Deferred search exhausted')
