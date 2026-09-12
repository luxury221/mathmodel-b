from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research' / 'policy_optimization_v3'))

import numpy as np
from candidate import CandidatePolicy
from geometry import minimum_circle, open_route
from plan_geometry import (
    CLEAR_SUPPORT_RADIUS,
    fallback_plan,
    hull_vertices,
    minimax_measurement,
    reception_proxy,
    small_clear_plan,
)
from shapely.geometry import Point

MODES = ('scan', 'tour_scan', 'outer_scan', 'probe_scan')
MAX_REUSE_MEASURES = 8
NEAR_REGION_METERS = 80.0
MIN_RECEPTION_PROXY = 0.2
MIN_CROSSING_SINE = 0.25
MAX_PROBES_PER_CHANNEL = 1
MAX_PROBE_SECONDS = 90.0
MAX_PROBE_CLEAR_FRACTION = 0.2


@lru_cache(maxsize=8)
def shorter_station_order(coordinates):
    points = np.asarray(coordinates)
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)

    def length(order):
        return float(sum(distances[first, second] for first, second in zip((0, *order[:-1]), order, strict=True)))

    starts = [list(range(1, len(points)))]
    for first in range(1, len(points)):
        order = [first]
        remaining = set(range(1, len(points))) - {first}
        while remaining:
            chosen = min(remaining, key=lambda index: (distances[order[-1], index], index))
            order.append(chosen)
            remaining.remove(chosen)
        starts.append(order)
    best = starts[0].copy()
    best_length = length(best)
    for order in starts:
        for _ in range(60):
            changed = False
            for left in range(len(order) - 1):
                previous = 0 if left == 0 else order[left - 1]
                for right in range(left + 1, len(order)):
                    before, after = distances[previous, order[left]], distances[previous, order[right]]
                    if right + 1 < len(order):
                        before += distances[order[right], order[right + 1]]
                        after += distances[order[left], order[right + 1]]
                    if after < before - 1e-8:
                        order[left:right + 1] = reversed(order[left:right + 1])
                        changed = True
            if not changed:
                break
        cost = length(order)
        if cost < best_length - 1e-8:
            best, best_length = order.copy(), cost
    return (0, *best)


class Q4OptimizationPolicy(CandidatePolicy):
    def __init__(self, port, problem, stations, variant, network, mode='scan'):
        if problem != 'q4' or network != 'dual21' or variant != 'F_route' or mode not in MODES:
            raise ValueError('This independent candidate only supports frozen q4 F_route/dual21')
        super().__init__(port, problem, stations, variant, network, mode='joint')
        self.optimization_mode = mode
        self.probe_counts = {channel: 0 for channel in self.states}
        self.stats.update({'reuse_measures': 0, 'reuse_positive': 0, 'reuse_near': 0,
                           'near_region_measures': 0, 'bounded_probes': 0, 'bounded_probe_positive': 0})
        if mode == 'tour_scan':
            order = shorter_station_order(tuple(tuple(float(value) for value in point) for point in self.route))
            self.route = self.route[list(order)].copy()
        elif mode == 'outer_scan':
            self.route = np.vstack((self.route[:1], self.route[9:], open_route(self.route[1:9], self.route[-1])))

    def reuse_current_position(self):
        candidates = []
        for state in self.states.values():
            if state.status != 'DETECTED' or not state.positives:
                continue
            key = (state.channel, float(self.position[0]).hex(), float(self.position[1]).hex())
            if key in self.cache:
                continue
            center, radius = minimum_circle(hull_vertices(state.region))
            if radius <= CLEAR_SUPPORT_RADIUS:
                continue
            near_region = state.region.distance(Point(self.position)) <= NEAR_REGION_METERS
            first, second = center - state.positives[-1][0], center - self.position
            product = float(np.linalg.norm(first) * np.linalg.norm(second))
            sine = abs(first[0] * second[1] - first[1] * second[0]) / max(product, 1e-9)
            if near_region:
                score = 2 * radius
            elif sine >= MIN_CROSSING_SINE:
                probability = reception_proxy(self.hypotheses(state), self.position)
                if probability < MIN_RECEPTION_PROXY:
                    continue
                score = probability * sine * radius
            else:
                continue
            candidates.append((float(score), state.channel, near_region))
        position = self.position.copy()
        for _score, channel, near_region in sorted(candidates, key=lambda item: (-item[0], item[1]))[:MAX_REUSE_MEASURES]:
            response = self.measure(channel, position, opportunistic=True)
            self.stats['reuse_measures'] += 1
            self.stats['reuse_positive'] += int(response['result'] == 'direction')
            self.stats['reuse_near'] += int(response['result'] == 'near')
            self.stats['near_region_measures'] += int(near_region)
            if not np.array_equal(self.position, position):
                raise ValueError('A zero-movement reuse measurement changed position')

    def opportunistic_actions(self, next_station):
        self.reuse_current_position()
        super().opportunistic_actions(next_station)

    def execute_clear_plan(self, state, plan, opportunistic=False):
        eligible = (self.optimization_mode == 'probe_scan' and not opportunistic
                    and plan.certificate_kind == 'rectangle_partition_circumradius_bound'
                    and len(plan.centers) >= 8 and self.probe_counts[state.channel] < MAX_PROBES_PER_CHANNEL)
        if eligible:
            measurement = minimax_measurement(state.region, state.positives[-1][0], self.position, state.measured_positions)
            budget = min(MAX_PROBE_SECONDS, MAX_PROBE_CLEAR_FRACTION * plan.worst_seconds)
            if measurement is not None and measurement.immediate_seconds <= budget:
                probability = reception_proxy(self.hypotheses(state), measurement.position)
                if probability >= MIN_RECEPTION_PROXY:
                    if measurement.max_receiving_distance > 999.900001:
                        raise ValueError('Probe violates the existing minimum-radius guard')
                    self.probe_counts[state.channel] += 1
                    self.stats['bounded_probes'] += 1
                    before = self.virtual_seconds
                    response = self.measure(state.channel, measurement.position, active=True)
                    if self.virtual_seconds - before > budget + 1e-6:
                        raise ValueError('Probe exceeded its predeclared immediate-cost bound')
                    self.stats['bounded_probe_positive'] += int(response['result'] in ('direction', 'near'))
                    if state.status == 'NEAR':
                        self.clear_near(state)
                        return
                    plan = small_clear_plan(state.region, self.position, 4)
                    if plan is None:
                        plan = fallback_plan(state.region, state.positives[0][1], self.position)
        super().execute_clear_plan(state, plan, opportunistic)
