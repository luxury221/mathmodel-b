from __future__ import annotations

import numpy as np
from geometry import open_route
from population import cumulative_subset_mass


class RouteObjective:
    def __init__(self, points, start, scan_indices, subset_mass, constant_unknown, radio_enabled):
        self.points = np.asarray(points)
        self.start_index = len(points)
        all_points = np.vstack((self.points, start))
        self.distances = np.linalg.norm(all_points[:, None] - all_points[None, :], axis=2)
        self.bits = np.zeros(len(points), dtype=np.int64)
        for bit, node in enumerate(scan_indices):
            self.bits[node] = 1 << bit
        self.all_bits = (1 << len(scan_indices)) - 1
        self.subset_mass = np.asarray(subset_mass)
        self.constant_unknown = constant_unknown
        self.radio_enabled = radio_enabled

    def scores(self, orders):
        orders = np.atleast_2d(orders)
        previous = np.column_stack((np.full(len(orders), self.start_index), orders[:, :-1]))
        travel = np.sum(self.distances[previous, orders], axis=1) / 5
        if not self.radio_enabled:
            return travel
        bits = self.bits[orders]
        prefixes = np.bitwise_or.accumulate(bits, axis=1)
        before = np.column_stack((np.zeros(len(orders), dtype=np.int64), prefixes[:, :-1]))
        survivors = self.constant_unknown + self.subset_mass[self.all_bits ^ before]
        radio = 6 * np.sum((bits != 0) * survivors, axis=1)
        return travel + radio


def nearest_order(distances, node_count, first):
    remaining = set(range(node_count)) - {first}
    order = [first]
    while remaining:
        selected = min(remaining, key=lambda node: (distances[order[-1], node], node))
        order.append(selected)
        remaining.remove(selected)
    return np.asarray(order, dtype=np.int64)


def improve_route(objective, reference, scan_indices):
    node_count = len(reference)
    candidates = [np.asarray(reference, dtype=np.int64)]
    candidates.extend(nearest_order(objective.distances, node_count, first) for first in scan_indices)
    scores = objective.scores(candidates)
    selected = int(np.argmin(scores))
    current, best = candidates[selected], float(scores[selected])
    for _iteration in range(8):
        neighbors = [current]
        for left in range(node_count - 1):
            for right in range(left + 1, node_count):
                trial = current.copy()
                trial[left:right + 1] = trial[left:right + 1][::-1]
                neighbors.append(trial)
        for source in range(node_count):
            reduced = np.delete(current, source)
            for destination in range(node_count):
                if destination != source:
                    neighbors.append(np.insert(reduced, destination, current[source]))
        scores = objective.scores(neighbors)
        index = int(np.argmin(scores))
        if scores[index] >= best - 1e-8:
            break
        current, best = neighbors[index], float(scores[index])
    if sorted(current.tolist()) != list(range(node_count)):
        raise ValueError('Route search lost a legal node')
    if best > float(objective.scores(reference)[0]) + 1e-7:
        raise ValueError('Proxy route became worse than its reference')
    return current, best


class DiscoveryRouter:
    def __init__(self, policy, model, mode):
        self.policy = policy
        self.model = model
        self.mode = mode
        self.subset_cache = None
        self.events = []

    def __call__(self, points, start=(0.0, 0.0)):
        points = np.asarray(points)
        reference_points = open_route(points, start)
        policy = self.policy
        policy.stats['latency_dispatch_calls'] += 1
        if self.mode == 'identity' or policy.discovery_done or len(points) < 2 or not policy.unknown_channels():
            return reference_points
        pair_distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
        np.fill_diagonal(pair_distances, np.inf)
        if np.min(pair_distances) < 1e-8:
            policy.stats['latency_colocated_fallbacks'] += 1
            return reference_points
        scan_indices = [index for index, point in enumerate(points)
                        if any(np.array_equal(point, station) for station in policy.route)
                        and not any(np.array_equal(point, observed) for observed in policy.coverage.observations)]
        if not 2 <= len(scan_indices) <= 20:
            return reference_points
        forecast = self.model.forecast(policy.unknown_channels(), policy.known_count())
        key = (self.model.revision, tuple(tuple(points[index]) for index in scan_indices), tuple(policy.unknown_channels()))
        if self.subset_cache is not None and self.subset_cache[0] == key:
            subset_mass = self.subset_cache[1]
        else:
            masks = np.zeros(len(self.model.positions), dtype=np.int64)
            for bit, index in enumerate(scan_indices):
                masks |= self.model.visible(points[index]).astype(np.int64) << bit
            subset_mass = cumulative_subset_mass(masks, forecast['masses'], len(scan_indices))
            self.subset_cache = (key, subset_mass)
        objective = RouteObjective(points, start, scan_indices, subset_mass, forecast['constant_unknown'], self.mode != 'distance')
        reference = np.array([int(np.argmin(np.linalg.norm(points - point, axis=1))) for point in reference_points])
        selected, value = improve_route(objective, reference, scan_indices)
        changed = selected[0] != reference[0]
        policy.stats['latency_route_calls'] += 1
        policy.stats['latency_first_changes'] += int(changed)
        self.events.append({'mode': self.mode, 'clock': policy.virtual_seconds, 'actual_position': np.asarray(start).tolist(),
                            'unknown_count': len(policy.unknown_channels()), 'known_count': policy.known_count(),
                            'scan_nodes': len(scan_indices), 'total_nodes': len(points),
                            'expected_sources': forecast['expected_sources'],
                            'unmapped_source_mass': forecast['unmapped_source_mass'],
                            'family_probabilities': forecast['family_probabilities'],
                            'reference_proxy_seconds': float(objective.scores(reference)[0]), 'selected_proxy_seconds': value,
                            'reference_first': reference_points[0].tolist(), 'selected_first': points[selected[0]].tolist(),
                            'first_changed': bool(changed), 'model_revision': self.model.revision})
        return points[selected].copy()
