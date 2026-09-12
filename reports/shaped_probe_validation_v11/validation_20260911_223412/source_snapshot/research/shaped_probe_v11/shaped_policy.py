from __future__ import annotations

import math

import numpy as np
from plan_geometry import reception_proxy
from policy_v5 import ContinuousPolicy
from shaped_geometry import certified_shaped_probe


MODES = ('narrow70', 'narrow85', 'shaped_cost')


class ShapedProbePolicy(ContinuousPolicy):
    def __init__(self, port, problem, stations, variant, network, mode='shaped_cost'):
        if problem != 'q4' or mode not in MODES:
            raise ValueError('Shaped V study only supports Q4 and registered modes')
        super().__init__(port, problem, stations, variant, network, mode='certified_fixed')
        self.shaped_mode = mode
        self.shaped_events = []
        self.stats.update({'shaped_proposals': 0, 'shaped_selections': 0})

    def score(self, state, probe, first, bank):
        destination = probe.candidates[first]
        mirror = probe.candidates[1 - first]
        first_distance = float(np.linalg.norm(destination - self.position))
        mirror_distance = float(np.linalg.norm(destination - mirror))
        if self.shaped_mode != 'shaped_cost' or bank is None:
            return first_distance + mirror_distance * (1 - reception_proxy(bank, destination))
        positions, headings, radii, weights = bank
        vectors = destination - positions
        receive = (np.linalg.norm(headings, axis=1) == 0) | (np.sum(vectors * headings, axis=1) >= 0)
        receive &= np.linalg.norm(vectors, axis=1) <= radii
        endpoints = np.where(receive[:, None], destination, mirror)
        next_vectors = endpoints - positions
        next_ranges = np.linalg.norm(next_vectors, axis=1)
        residual = np.full(len(positions), self.center_radius(state)[1])
        for anchor, _bearing in state.positives:
            old_vectors = anchor - positions
            old_ranges = np.linalg.norm(old_vectors, axis=1)
            crosses = np.abs(old_vectors[:, 0] * next_vectors[:, 1] - old_vectors[:, 1] * next_vectors[:, 0])
            sine = crosses / np.maximum(1e-8, old_ranges * next_ranges)
            estimated = math.tan(math.radians(1.005)) * next_ranges / np.maximum(sine, 0.015)
            residual = np.minimum(residual, estimated)
        probability_mirror = float(weights[~receive].sum())
        travel = (first_distance + probability_mirror * mirror_distance + float(weights @ next_ranges)) / 5
        return travel + 6 * (1 + probability_mirror) + 2 * float(weights @ np.maximum(0, residual - 19.95))

    def next_target_action(self, state):
        original = super().next_target_action(state)
        kind, _destination, extra = original
        if kind != 'v_probe':
            return original
        _mirror, original_probe = extra
        anchor = original_probe.anchor
        bearing = next(bearing for position, bearing in state.positives if np.array_equal(position, anchor))
        parameters = [(0.7 if self.shaped_mode == 'narrow70' else 0.85, 0.35)]
        if self.shaped_mode == 'shaped_cost':
            parameters = [(fraction, aspect) for fraction in (0.7, 0.85, 0.95) for aspect in (0.2, 0.5, 1.0)]
        proposals = []
        for fraction, aspect in parameters:
            probe = certified_shaped_probe(state.region, anchor, bearing, fraction, aspect)
            if probe is not None:
                proposals.append((probe, fraction, aspect))
        if self.shaped_mode == 'shaped_cost' or not proposals:
            proposals.append((original_probe, 0.7, 1.0))
        bank = self.hypotheses(state)
        choices = [(self.score(state, probe, first, bank), first, probe, fraction, aspect)
                   for probe, fraction, aspect in proposals for first in (0, 1)]
        score, first, probe, fraction, aspect = min(choices, key=lambda item: item[0])
        self.stats['shaped_proposals'] += len(proposals)
        self.stats['shaped_selections'] += int(probe is not original_probe)
        self.shaped_events.append({'channel': state.channel, 'revision': state.revision,
                                   'actual_position': self.position.tolist(), 'fraction': fraction,
                                   'aspect': aspect, 'score': score, 'mode': self.shaped_mode})
        return 'v_probe', probe.candidates[first], (probe.candidates[1 - first], probe)
