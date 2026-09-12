from __future__ import annotations

import itertools
import math
from collections import OrderedDict

import numpy as np


def subset_coefficients(values):
    coefficients = np.array([1.0])
    for value in values:
        updated = np.zeros(len(coefficients) + 1)
        updated[:-1] += coefficients
        updated[1:] += value * coefficients
        coefficients = updated
    return coefficients


def population_posterior(survival, known_logs, known_count, absent_count, prior,
                         minimum_count=10, maximum_count=16, total_channels=20):
    survival = np.asarray(survival, dtype=float)
    family_count, unknown_count = survival.shape
    if known_count + absent_count + unknown_count != total_channels:
        raise ValueError('Channel counts disagree')
    lower = max(minimum_count, known_count)
    upper = min(maximum_count, known_count + unknown_count)
    if lower > upper or np.any(survival < 0) or np.any(survival > 1 + 1e-10):
        raise ValueError('Inconsistent reference population model')
    bounded = np.clip(survival, 1e-12, 1.0)
    coefficients = [subset_coefficients(row) for row in bounded]
    log_weights = np.full((family_count, maximum_count + 1), -np.inf)
    for family in range(family_count):
        if prior[family] <= 0:
            continue
        for count in range(lower, upper + 1):
            log_weights[family, count] = (math.log(prior[family]) + known_logs[family]
                                         + math.log(coefficients[family][count - known_count])
                                         - math.log(math.comb(total_channels, count)))
    largest = float(np.max(log_weights))
    if not np.isfinite(largest):
        raise ValueError('No finite reference posterior')
    joint = np.exp(log_weights - largest)
    joint /= joint.sum()
    presence = np.zeros_like(survival)
    for family in range(family_count):
        for channel_index in range(unknown_count):
            others = subset_coefficients(np.delete(bounded[family], channel_index))
            for count in range(max(lower, known_count + 1), upper + 1):
                remaining = count - known_count
                conditional = bounded[family, channel_index] * others[remaining - 1] / coefficients[family][remaining]
                presence[family, channel_index] += joint[family, count] * conditional
    return joint, presence


def public_particles(problem):
    radial_nodes, radial_weights = np.polynomial.legendre.leggauss(2)
    receiver_nodes, receiver_weights = np.polynomial.legendre.leggauss(3)
    radii = 1250 + 250 * receiver_nodes
    receiver_weights /= 2
    positions, position_weights = [], []
    bands = (0, 20, 100, 300, 600, 900, 1200, 1500, 1800)
    for band_index, (lower, upper) in enumerate(zip(bands[:-1], bands[1:])):
        for node, weight in zip(radial_nodes, radial_weights):
            radius = math.sqrt(lower**2 + (node + 1) / 2 * (upper**2 - lower**2))
            for angular_index in range(72):
                angle = (angular_index + 0.5 * (band_index % 2)) * 2 * math.pi / 72
                positions.append((radius * math.cos(angle), radius * math.sin(angle)))
                area_weight = (upper**2 - lower**2) / 1800**2 * weight / 2 / 72
                boundary_weight = (upper**2 - lower**2) / (1800**2 - 1500**2) * weight / 2 / 72 if lower >= 1500 else 0.0
                position_weights.append((area_weight, boundary_weight))
    headings = np.zeros((1, 2))
    heading_weights = np.ones(1)
    if problem == 'q4':
        angles = np.arange(24) * 2 * math.pi / 24
        headings = np.vstack((headings, np.column_stack((np.cos(angles), np.sin(angles)))))
        heading_weights = np.r_[0.5, np.full(24, 0.5 / 24)]
    elif problem != 'q3':
        raise ValueError('Unknown problem')
    positions = np.asarray(positions)
    group = len(headings) * len(radii)
    expanded_positions = np.repeat(positions, group, axis=0)
    expanded_headings = np.tile(np.repeat(headings, len(radii), axis=0), (len(positions), 1))
    expanded_radii = np.tile(radii, len(positions) * len(headings))
    factors = np.repeat(heading_weights, len(radii)) * np.tile(receiver_weights, len(headings))
    weights = np.repeat(np.asarray(position_weights).T, group, axis=1) * np.tile(factors, len(positions))
    if not np.allclose(weights.sum(axis=1), 1.0):
        raise ValueError('Public quadrature weights are not normalized')
    return expanded_positions, expanded_headings, expanded_radii, weights


class PopulationModel:
    def __init__(self, problem, mode):
        self.positions, self.headings, self.radii, self.weights = public_particles(problem)
        self.omni = np.linalg.norm(self.headings, axis=1) == 0
        self.alive = {channel: np.ones(len(self.positions), dtype=bool) for channel in range(1, 21)}
        self.known_channels = set()
        self.known_logs = np.zeros(2)
        self.prior = np.array([1.0, 0.0]) if mode == 'area' else np.array([0.5, 0.5])
        self.cache = OrderedDict()
        self.posterior_cache = None
        self.revision = 0
        self.radio_observations = 0
        self.events = []

    def visible(self, position, optical=False):
        key = (optical, *map(float, position))
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        vectors = np.asarray(position) - self.positions
        squared = np.sum(vectors * vectors, axis=1)
        if optical:
            visible = squared <= 20**2
        else:
            visible = (squared <= self.radii**2) & (self.omni | (np.sum(vectors * self.headings, axis=1) >= 0))
        self.cache[key] = visible
        if len(self.cache) > 48:
            self.cache.popitem(last=False)
        return visible

    def observe(self, channel, position, hit, optical=False):
        if channel in self.known_channels or channel not in self.alive:
            raise ValueError('Only real pre-discovery observations enter the model')
        visible = self.visible(position, optical)
        alive = self.alive[channel]
        if hit:
            masses = np.sum(self.weights[:, alive & visible], axis=1)
            self.known_logs += np.log(np.maximum(masses, 1e-12))
            self.known_channels.add(channel)
            del self.alive[channel]
            self.events.append({'channel': channel, 'position': np.asarray(position).tolist(),
                                'optical': optical, 'first_hit_masses': masses.tolist()})
        else:
            alive &= ~visible
        self.revision += 1
        self.radio_observations += int(not optical)
        self.posterior_cache = None

    def forecast(self, unknown_channels, known_count):
        unknown_channels = tuple(unknown_channels)
        key = (self.revision, unknown_channels, known_count)
        if self.posterior_cache is not None and self.posterior_cache[0] == key:
            return self.posterior_cache[1]
        if len(self.known_channels) != known_count:
            raise ValueError('Reference model missed a real first detection')
        survival = np.column_stack([np.sum(self.weights[:, self.alive[channel]], axis=1) for channel in unknown_channels])
        absent_count = 20 - known_count - len(unknown_channels)
        joint, presence = population_posterior(survival, self.known_logs, known_count, absent_count, self.prior)
        masses = np.zeros(len(self.positions))
        for channel_index, channel in enumerate(unknown_channels):
            alive = self.alive[channel]
            for family in range(2):
                if survival[family, channel_index] > 1e-15:
                    masses += presence[family, channel_index] / survival[family, channel_index] * self.weights[family] * alive
        expected_sources = float(presence.sum())
        mapped_sources = float(masses.sum())
        result = {'masses': masses, 'expected_sources': expected_sources,
                  'unmapped_source_mass': max(0.0, expected_sources - mapped_sources),
                  'constant_unknown': max(0.0, len(unknown_channels) - mapped_sources),
                  'family_probabilities': joint.sum(axis=1).tolist(),
                  'survival_probabilities': survival.mean(axis=1).tolist()}
        self.posterior_cache = (key, result)
        return result


def cumulative_subset_mass(masks, weights, bit_count):
    if not 0 <= bit_count <= 20:
        raise ValueError('Subset mass is bounded to twenty planned stations')
    masses = np.bincount(np.asarray(masks, dtype=np.int64), weights=weights, minlength=1 << bit_count)
    if len(masses) != 1 << bit_count:
        raise ValueError('Visibility mask outside the planned station set')
    for bit in range(bit_count):
        half = 1 << bit
        blocks = masses.reshape(-1, half * 2)
        blocks[:, half:] += blocks[:, :half]
    return masses
