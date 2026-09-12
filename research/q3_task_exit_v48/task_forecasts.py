from __future__ import annotations

import copy
import math
import time

import numpy as np
from local_model import HypothesisPort
from plan_geometry import source_proxy_hypotheses
from scan_policy import ScanEconomyPolicy


class RecordedHypothesisPort(HypothesisPort):
    def __init__(self, channel, position, radius, error):
        super().__init__(channel, position, radius, error)
        self.trace = []

    def measure(self, point, channel):
        response = super().measure(point, channel)
        self.trace.append({'action': 'measure', 'position': np.asarray(point).tolist(), 'channel': channel,
                           'response': response})
        return response

    def clear(self, point, channel):
        response = super().clear(point, channel)
        self.trace.append({'action': 'clear', 'position': np.asarray(point).tolist(), 'channel': channel,
                           'response': response})
        return response


class ForecastSandbox(ScanEconomyPolicy):
    def reuse_stop(self):
        return None


def legal_samples(state):
    bank = source_proxy_hypotheses(state.region, state.positives, state.negatives, False, sample_count=40)
    if bank is None:
        raise ValueError('No conditional source hypotheses')
    positions, headings, radii, weights = bank
    valid = (np.linalg.norm(positions, axis=1) <= 1800) & (radii >= 1000) & (radii <= 1500)
    valid &= np.linalg.norm(headings, axis=1) == 0
    for anchor, bearing in state.positives:
        vectors = positions - anchor
        distances = np.linalg.norm(vectors, axis=1)
        angles = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))
        errors = (angles - bearing + 180) % 360 - 180
        valid &= (distances > 5) & (distances <= radii) & (np.abs(errors) <= 1.005 + 1e-9)
    for anchor in state.negatives:
        valid &= np.linalg.norm(positions - anchor, axis=1) > radii
    if not valid.any():
        raise ValueError('No hypotheses satisfying the public source domain and actual observations')
    positions, radii, weights = positions[valid], radii[valid], weights[valid]
    weights = weights / weights.sum()
    indices = np.searchsorted(np.cumsum(weights), (np.arange(7) + 0.5) / 7, side='left')
    indices = np.minimum(indices, len(positions) - 1)
    return [(positions[index].copy(), float(radii[index])) for index in indices], {
        'bank_count': len(bank[0]), 'legal_bank_count': len(positions), 'quadrature_samples': len(indices)}


def simulate_original_task(policy, state, sample, error, expected_action):
    source_position, radius = sample
    port = RecordedHypothesisPort(state.channel, source_position, radius, error)
    sandbox = ForecastSandbox(port, 'q3', policy.stations, 'E_joint', 'grid7', mode='station_only')
    sandbox.states = {state.channel: copy.deepcopy(state)}
    sandbox.position = policy.position.copy()
    sandbox.receiver_channel = policy.receiver_channel
    sandbox.options = copy.deepcopy(policy.options)
    sandbox.cache = {key: copy.deepcopy(value) for key, value in policy.cache.items() if key[0] == state.channel}
    sandbox.probe_history[state.channel] = copy.deepcopy(policy.probe_history[state.channel])
    sandbox.localize(sandbox.states[state.channel])
    if sandbox.states[state.channel].status != 'CLEARED' or not port.cleared or not port.trace:
        raise ValueError('Original localize failed to finish a conditional task')
    expected_kind = 'measure' if expected_action.kind == 'probe' else 'clear'
    first = port.trace[0]
    if first['action'] != expected_kind or first['position'] != expected_action.position.tolist():
        raise ValueError('Conditional task first action differs from the real-action preview')
    if sandbox.stats['active_measures'] > 8:
        raise ValueError('Conditional task exceeded original active measurement budget')
    return {
        'source_position_hypothesis': np.asarray(source_position).tolist(), 'radius_hypothesis': radius,
        'future_error': error, 'exit': sandbox.position.tolist(), 'seconds': sandbox.virtual_seconds,
        'active_measures': sandbox.stats['active_measures'], 'physical_actions': len(port.trace),
        'first_action': first, 'cleared': True,
    }


def forecast_task(policy, state, first_action):
    started = time.perf_counter()
    if state.status == 'NEAR' or (first_action.kind == 'clear' and len(first_action.plan.centers) == 1):
        return {'ok': True, 'exact_exit': True, 'exits': [first_action.position.tolist()], 'weights': [1.0],
                'outcomes': [], 'mean_seconds': float(np.linalg.norm(first_action.position - policy.position)) / 5 + 5,
                'wall_seconds': time.perf_counter() - started}
    try:
        samples, metadata = legal_samples(state)
    except ValueError as error:
        return {'ok': False, 'failures': [{'failure': str(error)}], 'outcomes': [],
                'wall_seconds': time.perf_counter() - started}
    outcomes, failures = [], []
    for sample_index, sample in enumerate(samples):
        for error in (-1.0, 0.0, 1.0):
            try:
                outcome = simulate_original_task(policy, state, sample, error, first_action)
                outcome['sample_index'] = sample_index
                outcomes.append(outcome)
            except Exception as exception:
                failures.append({'sample_index': sample_index, 'future_error': error,
                                 'failure': type(exception).__name__ + ': ' + str(exception)})
    result = {'ok': not failures, 'exact_exit': False, 'outcomes': outcomes, 'failures': failures,
              'sampling': metadata, 'wall_seconds': time.perf_counter() - started}
    if not failures:
        result.update({'exits': [outcome['exit'] for outcome in outcomes], 'weights': [1 / len(outcomes)] * len(outcomes),
                       'mean_seconds': float(np.mean([outcome['seconds'] for outcome in outcomes]))})
    return result
