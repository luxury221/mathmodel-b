from __future__ import annotations

import math

import numpy as np
from plan_geometry import CLEAR_SUPPORT_RADIUS, reception_proxy
from scan_policy import ScanEconomyPolicy
from shaped_policy import ShapedProbePolicy
from shapely.geometry import Point


MODES = ('identity', 'extended', 'guarded')


def forecast_precision(state, point, bank, radius):
    if bank is None:
        return 0.0, radius
    positions, headings, radii, weights = bank
    vectors = np.asarray(point) - positions
    distances = np.linalg.norm(vectors, axis=1)
    received = (distances <= radii) & ((np.linalg.norm(headings, axis=1) == 0) | (np.sum(vectors * headings, axis=1) >= 0))
    probability = float(weights[received].sum())
    if probability <= 1e-12:
        return probability, radius
    residual = np.full(len(positions), radius, dtype=float)
    for anchor, _bearing in state.positives:
        previous = anchor - positions
        crosses = np.abs(previous[:, 0] * vectors[:, 1] - previous[:, 1] * vectors[:, 0])
        sine = crosses / np.maximum(1e-8, np.linalg.norm(previous, axis=1) * distances)
        estimated = math.tan(math.radians(1.005)) * distances / np.maximum(sine, 0.015)
        residual = np.minimum(residual, estimated)
    return probability, float(weights[received] @ residual[received]) / probability


class RangeOpportunityMixin:
    reference_mode = None

    def __init__(self, port, problem, stations, variant, network, mode='guarded'):
        if mode not in MODES:
            raise ValueError('Unknown range-opportunity mode')
        super().__init__(port, problem, stations, variant, network, mode=self.reference_mode)
        self.opportunity_mode = mode
        self.extended_events = []
        self.stats.update({'extended_candidates': 0, 'extended_guard_rejections': 0,
                           'extended_actions': 0, 'extended_directions': 0, 'extended_no_signal': 0})

    def reuse_directions(self):
        if self.opportunity_mode == 'identity':
            return super().reuse_directions()
        candidates = []
        extensions = {}
        for state in self.states.values():
            if state.status != 'DETECTED' or state.channel == self.active_target:
                continue
            if any(np.linalg.norm(self.position - previous) < 1e-5 for previous in state.measured_positions):
                continue
            center, radius = self.center_radius(state)
            lower_distance = state.region.distance(Point(self.position))
            if radius <= CLEAR_SUPPORT_RADIUS or lower_distance > 1500:
                continue
            previous_vector = center - state.positives[-1][0]
            new_vector = center - self.position
            sine = abs(previous_vector[0] * new_vector[1] - previous_vector[1] * new_vector[0]) / max(1e-9, np.linalg.norm(previous_vector) * np.linalg.norm(new_vector))
            if sine < 0.2 and np.linalg.norm(new_vector) > 100:
                continue
            bank = self.hypotheses(state) if self.problem == 'q4' or lower_distance > 1000 else None
            if self.problem == 'q4' and reception_proxy(bank, self.position) < 0.25:
                continue
            if lower_distance > 1000:
                probability, residual = forecast_precision(state, self.position, bank, radius)
                self.stats['extended_candidates'] += 1
                if self.opportunity_mode == 'guarded' and (probability < 0.6 or residual > 0.65 * radius):
                    self.stats['extended_guard_rejections'] += 1
                    continue
                extensions[state.channel] = {'channel': state.channel, 'position': self.position.tolist(),
                                             'lower_distance': lower_distance, 'radius_before': radius,
                                             'receive_probability': probability, 'predicted_radius': residual,
                                             'clock_before': self.virtual_seconds}
            candidates.append((radius * max(sine, 0.2), state.channel))
        for _score, channel in sorted(candidates, reverse=True)[:6]:
            clock = self.virtual_seconds
            response = self.measure(channel, self.position, opportunistic=True)
            if channel in extensions and self.virtual_seconds > clock:
                self.stats['extended_actions'] += 1
                self.stats['extended_directions'] += int(response['result'] == 'direction')
                self.stats['extended_no_signal'] += int(response['result'] == 'no_signal')
                state = self.states[channel]
                event = extensions[channel]
                event.update({'response': response['result'], 'clock_after': self.virtual_seconds,
                              'radius_after': self.center_radius(state)[1] if state.status == 'DETECTED' else 0.0})
                self.extended_events.append(event)


class RangeOpportunityQ3(RangeOpportunityMixin, ScanEconomyPolicy):
    reference_mode = 'station_only'


class RangeOpportunityQ4(RangeOpportunityMixin, ShapedProbePolicy):
    reference_mode = 'shaped_cost'
