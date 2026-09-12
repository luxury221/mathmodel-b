from __future__ import annotations

import math

import numpy as np
from shaped_policy import ShapedProbePolicy


MODES = ('identity', 'single', 'cost_guarded')


def probe_forecast(policy, state, point, original):
    bank = policy.hypotheses(state)
    if bank is None:
        return None
    _kind, original_point, extra = original
    _mirror, certificate = extra
    first = int(np.argmin(np.linalg.norm(certificate.candidates - original_point, axis=1)))
    reference_cost = policy.score(state, certificate, first, bank)
    positions, headings, radii, weights = bank
    vectors = np.asarray(point) - positions
    distances = np.linalg.norm(vectors, axis=1)
    received = (distances <= radii) & ((np.linalg.norm(headings, axis=1) == 0) | (np.sum(vectors * headings, axis=1) >= 0))
    probability = float(weights[received].sum())
    radius = policy.center_radius(state)[1]
    residual = np.full(len(positions), radius, dtype=float)
    for anchor, _bearing in state.positives:
        previous = anchor - positions
        crosses = np.abs(previous[:, 0] * vectors[:, 1] - previous[:, 1] * vectors[:, 0])
        sine = crosses / np.maximum(1e-8, np.linalg.norm(previous, axis=1) * distances)
        estimated = math.tan(math.radians(1.005)) * distances / np.maximum(sine, 0.015)
        residual = np.minimum(residual, estimated)
    reference_remaining = max(0.0, reference_cost - float(np.linalg.norm(original_point - policy.position)) / 5)
    cost = float(np.linalg.norm(point - policy.position)) / 5 + 6
    cost += float(weights[received] @ (distances[received] / 5 + 2 * np.maximum(0, residual[received] - 19.95)))
    cost += (1 - probability) * (float(np.linalg.norm(point - original_point)) / 5 + reference_remaining)
    return {'receive_probability': probability, 'predicted_cost': cost, 'reference_cost': reference_cost,
            'radius_before': radius, 'certified_lower_range': certificate.lower_range,
            'certified_forward': certificate.forward}


class BoundedProbePolicy(ShapedProbePolicy):
    def __init__(self, *args, mode='cost_guarded'):
        if mode not in MODES:
            raise ValueError('Unknown bounded probe mode')
        super().__init__(*args, mode='shaped_cost')
        self.bounded_mode = mode
        self.bounded_used = set()
        self.bounded_pending = {}
        self.bounded_events = []
        self.stats.update({'bounded_proposals': 0, 'bounded_actions': 0, 'bounded_directions': 0, 'bounded_no_signal': 0})

    def next_target_action(self, state):
        original = super().next_target_action(state)
        if self.bounded_mode == 'identity' or state.channel in self.bounded_used or original[0] != 'v_probe':
            return original
        self.bounded_pending.pop(state.channel, None)
        if self.center_radius(state)[1] <= 100:
            return original
        point = self.choose_probe(state)
        if point is None or any(np.linalg.norm(point - previous) < 25 for previous in state.measured_positions):
            return original
        certificate = original[2][1]
        bearing = next(bearing for anchor, bearing in state.positives if np.array_equal(anchor, certificate.anchor))
        direction = np.array([math.cos(math.radians(bearing)), math.sin(math.radians(bearing))])
        forward = float((point - certificate.anchor) @ direction)
        if forward < certificate.forward + 50:
            return original
        forecast = probe_forecast(self, state, point, original)
        if forecast is None or forecast['receive_probability'] < 0.5:
            return original
        if self.bounded_mode == 'cost_guarded' and (forecast['receive_probability'] < 0.65 or forecast['predicted_cost'] > forecast['reference_cost'] - 6):
            return original
        self.stats['bounded_proposals'] += 1
        self.bounded_pending[state.channel] = (point.copy(), {**forecast, 'channel': state.channel,
                                                             'point': point.tolist(), 'forward': forward,
                                                             'original_point': original[1].tolist()})
        return 'probe', point, None

    def measure(self, channel, position, active=False, opportunistic=False):
        proposal = self.bounded_pending.get(channel)
        executed = active and channel not in self.bounded_used and proposal is not None and np.array_equal(position, proposal[0])
        before = self.virtual_seconds
        response = super().measure(channel, position, active, opportunistic)
        if executed and self.virtual_seconds > before:
            self.bounded_used.add(channel)
            self.bounded_pending.pop(channel, None)
            self.stats['bounded_actions'] += 1
            self.stats['bounded_directions'] += int(response['result'] == 'direction')
            self.stats['bounded_no_signal'] += int(response['result'] == 'no_signal')
            state = self.states[channel]
            event = proposal[1]
            event.update({'response': response['result'], 'clock_before': before, 'clock_after': self.virtual_seconds,
                          'radius_after': self.center_radius(state)[1] if state.status == 'DETECTED' else 0.0})
            self.bounded_events.append(event)
        return response
