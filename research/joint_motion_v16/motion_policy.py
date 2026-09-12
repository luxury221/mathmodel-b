from __future__ import annotations

import math

import numpy as np
from geometry import open_route
from plan_geometry import small_clear_plan
from shaped_policy import ShapedProbePolicy


MODES = ('terminal', 'transit', 'combined')


class JointMotionPolicy(ShapedProbePolicy):
    def __init__(self, *args, mode='combined'):
        if mode not in MODES:
            raise ValueError('Unknown joint-motion mode')
        super().__init__(*args, mode='shaped_cost')
        self.motion_mode = mode
        self.transit_counts = {channel: 0 for channel in self.states}
        self.motion_events = []
        self.stats.update({'transit_stops': 0, 'transit_measures': 0,
                           'transit_directions': 0, 'transit_clearable': 0})

    def build_stops(self):
        unresolved = [state for state in self.states.values() if state.status in ('DETECTED', 'NEAR')]
        stops = [(kind, state.channel, destination, extra)
                 for state in unresolved for kind, destination, extra in [self.next_target_action(state)]]
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
        positions = []
        for kind, identifier, destination, _extra in stops:
            if self.motion_mode in ('terminal', 'combined') and kind in ('v_probe', 'v_mirror', 'probe', 'anchor_clear'):
                positions.append(self.center_radius(self.states[identifier])[0])
            else:
                positions.append(destination)
        return np.asarray(positions)

    def transit_proposal(self, destination, stops):
        if self.motion_mode == 'terminal':
            return None
        segment = np.asarray(destination) - self.position
        length_squared = float(segment @ segment)
        if length_squared < 120 ** 2:
            return None
        information = []
        fractions = [0.25, 0.5, 0.75, 1.0]
        for kind, identifier, planned, _extra in stops:
            if kind not in ('v_probe', 'v_mirror', 'probe', 'anchor_clear'):
                continue
            state = self.states[identifier]
            if state.status != 'DETECTED' or self.transit_counts[identifier] >= 3:
                continue
            center, radius = self.center_radius(state)
            bank = self.hypotheses(state)
            if bank is None or radius <= 50:
                continue
            extra_distance = max(0.0, float(np.linalg.norm(planned - self.position)
                                           + np.linalg.norm(planned - center)
                                           - np.linalg.norm(center - self.position)))
            value = 6 + extra_distance / 5 + min(90, radius * 0.25)
            information.append((state, radius, bank, value))
            fractions.append(float(np.clip((center - self.position) @ segment / length_squared, 0.0, 1.0)))
        choices = []
        candidates = []
        for fraction in sorted(fractions):
            point = self.position + fraction * segment
            if np.linalg.norm(point - self.position) < 60:
                continue
            if any(np.linalg.norm(point - prior) < 25 for prior in candidates):
                continue
            candidates.append(point)
            gains = []
            for state, radius, bank, value in information:
                if any(np.linalg.norm(point - prior) < 50 for prior in state.measured_positions):
                    continue
                positions, headings, ranges, weights = bank
                vectors = point - positions
                distances = np.linalg.norm(vectors, axis=1)
                receive = (np.linalg.norm(headings, axis=1) == 0) | (np.sum(vectors * headings, axis=1) >= 0)
                receive &= distances <= ranges
                residual = np.full(len(positions), radius)
                for anchor, _bearing in state.positives:
                    previous = anchor - positions
                    crosses = np.abs(previous[:, 0] * vectors[:, 1] - previous[:, 1] * vectors[:, 0])
                    sine = crosses / np.maximum(1e-8, np.linalg.norm(previous, axis=1) * distances)
                    estimate = math.tan(math.radians(1.005)) * distances / np.maximum(sine, 0.015)
                    residual = np.minimum(residual, estimate)
                improvement = np.clip(1 - residual / radius, 0, 1)
                gain = float(weights @ (receive * improvement)) * value - 6
                if gain >= 10 and float(weights[receive].sum()) >= 0.6:
                    gains.append((gain, state.channel))
            gains = sorted(gains, reverse=True)[:6]
            if gains:
                choices.append((sum(gain for gain, _channel in gains), -fraction, point, gains))
        if not choices:
            return None
        _gain, _fraction, point, gains = max(choices, key=lambda item: item[:2])
        return point, gains

    def execute_transit(self, destination, proposal):
        point, gains = proposal
        origin = self.position.copy()
        event = {'origin': origin.tolist(), 'planned_destination': np.asarray(destination).tolist(),
                 'actual_stop': point.tolist(), 'channels': [], 'clock_before': self.virtual_seconds}
        for gain, channel in gains:
            state = self.states[channel]
            response = self.measure(channel, point, active=True)
            self.transit_counts[channel] += 1
            self.stats['transit_measures'] += 1
            self.stats['transit_directions'] += int(response['result'] in ('near', 'direction'))
            clearable = state.status == 'NEAR' or small_clear_plan(state.region, self.position, 4) is not None
            self.stats['transit_clearable'] += int(clearable)
            event['channels'].append({'channel': channel, 'predicted_gain': gain,
                                      'result': response['result'], 'clearable': clearable})
        self.stats['transit_stops'] += 1
        event['clock_after'] = self.virtual_seconds
        self.motion_events.append(event)
        self.reuse_stop()

    def execute_stop(self, stop):
        kind, channel, destination, extra = stop
        if kind == 'fixed':
            self.scan_unknown(destination, force=True)
            self.stats['visited_stations'] += 1
            self.reuse_directions()
            self.remaining_stations.remove(channel)
        elif kind == 'optical':
            self.stats['optical_completion_stops'] += 1
            self.common_clear(destination)
            self.reuse_directions()
        elif kind == 'near':
            self.clear_near(self.states[channel])
            self.reuse_stop()
        elif kind == 'clear':
            self.execute_clear_plan(self.states[channel], extra)
        elif kind == 'anchor_clear':
            self.stats['anchor_optical_probes'] += 1
            self.clear(channel, destination)
            self.reuse_stop()
        elif kind in ('probe', 'v_probe', 'v_mirror'):
            if kind == 'v_probe':
                mirror, certificate = extra
                self.probe_certificates.append(certificate)
            response = self.measure(channel, destination, active=True)
            self.active_counts[channel] += 1
            if kind == 'v_probe':
                self.stats['v_probes'] += 1
                if response['result'] == 'no_signal':
                    self.pending_probes[channel] = mirror
            elif kind == 'v_mirror':
                self.stats['v_mirrors'] += 1
                self.pending_probes.pop(channel)
                if response['result'] == 'no_signal':
                    raise ValueError('Both guaranteed V-probe members failed')
            else:
                self.stats['one_step_probes'] += 1
            self.reuse_stop()
        else:
            raise ValueError('Unknown joint-motion action')

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
            positions = self.route_positions(stops)
            itinerary = open_route(positions, self.position)
            selected = int(np.argmin(np.linalg.norm(positions - itinerary[0], axis=1)))
            stop = stops[selected]
            proposal = self.transit_proposal(stop[2], stops)
            if proposal is not None:
                self.execute_transit(stop[2], proposal)
            else:
                self.execute_stop(stop)
        raise ValueError('Joint-motion action limit exceeded')
