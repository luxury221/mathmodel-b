from __future__ import annotations

import numpy as np
from motion_policy import JointMotionPolicy
from shaped_policy import ShapedProbePolicy


MODES = ('identity', 'two_bearings', 'clear_ready')


class ObservationGatePolicy(ShapedProbePolicy):
    def __init__(self, *args, mode='two_bearings'):
        if mode not in MODES:
            raise ValueError('Unknown observation-gate mode')
        super().__init__(*args, mode='shaped_cost')
        self.gate_mode = mode
        self.deferred_channels = set()
        self.stats['observation_gate_deferrals'] = 0

    def admitted(self, state, kind):
        if self.gate_mode == 'identity' or self.discovery_done:
            return True
        if kind in ('near', 'clear', 'anchor_clear', 'v_mirror'):
            return True
        return self.gate_mode == 'two_bearings' and len(state.positives) >= 2

    def build_stops(self):
        stops = []
        for state in self.states.values():
            if state.status not in ('DETECTED', 'NEAR'):
                continue
            kind, destination, extra = self.next_target_action(state)
            if self.admitted(state, kind):
                stops.append((kind, state.channel, destination, extra))
            else:
                self.deferred_channels.add(state.channel)
                self.stats['observation_gate_deferrals'] += 1
        if not self.discovery_done:
            if self.stats['visited_stations'] >= 40:
                self.fallback_discovery()
                return []
            self.patrol.reduce([stop[2] for stop in stops])
            optical = self.optical_components()
            if optical:
                stops.extend(('optical', index, point, None) for index, point in enumerate(optical))
            elif self.remaining_stations:
                stops.extend(('fixed', index, self.route[index], None) for index in self.remaining_stations)
            else:
                self.fallback_discovery()
                return []
        return stops

    def route_positions(self, stops):
        return np.asarray([stop[2] for stop in stops])

    def transit_proposal(self, destination, stops):
        return None

    execute_stop = JointMotionPolicy.execute_stop
    run = JointMotionPolicy.run
