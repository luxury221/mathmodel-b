from __future__ import annotations

import copy
import math

import numpy as np
from scan_policy import ScanEconomyPolicy


class HypothesisPort:
    def __init__(self, channel, position, radius, error):
        self.channel = channel
        self.position = np.asarray(position).copy()
        self.radius = float(radius)
        self.error = float(error)
        self.cleared = False

    def measure(self, point, channel):
        if channel != self.channel:
            raise ValueError('Local forecast requested another channel')
        vector = self.position - np.asarray(point)
        distance = float(np.linalg.norm(vector))
        if self.cleared or distance > self.radius:
            return {'result': 'no_signal'}
        if distance <= 5:
            return {'result': 'near'}
        angle = math.degrees(math.atan2(vector[1], vector[0]))
        return {'result': 'direction', 'bearing_deg': round((angle + self.error) % 360, 2) % 360}

    def clear(self, point, channel):
        if channel != self.channel:
            raise ValueError('Local forecast requested another channel')
        success = not self.cleared and np.linalg.norm(self.position - point) <= 20
        self.cleared |= bool(success)
        return {'result': 'success' if success else 'no_target_in_range'}


class LocalizationSandbox(ScanEconomyPolicy):
    def reuse_stop(self):
        return None


def conditional_samples(bank):
    if bank is None:
        return []
    positions, _headings, radii, weights = bank
    cumulative = np.cumsum(weights)
    indices = np.searchsorted(cumulative, (np.arange(7) + 0.5) / 7, side='left')
    indices = np.minimum(indices, len(positions) - 1)
    return [(positions[index].copy(), float(radii[index])) for index in indices]


def simulate_completion(policy, state, sample, error, point=None, clear_plan=None):
    position, radius = sample
    port = HypothesisPort(state.channel, position, radius, error)
    sandbox = LocalizationSandbox(port, 'q3', policy.stations, 'E_joint', 'grid7', mode='station_only')
    sandbox.states = {state.channel: copy.deepcopy(state)}
    sandbox.position = policy.position.copy()
    sandbox.receiver_channel = policy.receiver_channel
    sandbox.cache = {key: copy.deepcopy(value) for key, value in policy.cache.items() if key[0] == state.channel}
    selected = sandbox.states[state.channel]
    if clear_plan is not None:
        sandbox.active_target = state.channel
        sandbox.execute_clear_plan(selected, clear_plan)
    else:
        sandbox.measure(state.channel, np.asarray(point), active=True)
        sandbox.localize(selected)
    if selected.status != 'CLEARED' or not port.cleared:
        raise ValueError('Conditional local completion failed')
    return sandbox.virtual_seconds


def forecast(policy, state, samples, point=None, clear_plan=None):
    costs = []
    failures = []
    for sample_index, sample in enumerate(samples):
        for error in (-1.0, 0.0, 1.0):
            try:
                costs.append(simulate_completion(policy, state, sample, error, point, clear_plan))
            except Exception as exception:
                costs.append(1000000.0)
                failures.append({'sample_index': sample_index, 'error': error,
                                 'failure': type(exception).__name__ + ': ' + str(exception)})
    if not costs:
        raise ValueError('Forecast requires conditional samples')
    mean = float(np.mean(costs))
    worst = float(max(costs))
    return {'score': 0.85 * mean + 0.15 * worst, 'mean': mean, 'worst': worst,
            'sample_count': len(costs), 'failures': failures}
