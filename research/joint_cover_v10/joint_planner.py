from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import shapely
from geometry import open_route, route_length
from search_planner import WitnessPlanner


@dataclass
class JointPlan:
    scans: list
    itinerary: np.ndarray
    predicted_seconds: float
    baseline_seconds: float
    mode: str


def unique_points(points):
    return list({tuple(np.asarray(point, dtype=float)): np.asarray(point, dtype=float) for point in points}.values())


def insertion_cost(point, itinerary, start):
    if not len(itinerary):
        return float(np.linalg.norm(point - start)), 0
    starts = np.vstack((start, itinerary[:-1]))
    costs = np.linalg.norm(starts - point, axis=1) + np.linalg.norm(itinerary - point, axis=1)
    costs -= np.linalg.norm(itinerary - starts, axis=1)
    costs = np.r_[costs, np.linalg.norm(itinerary[-1] - point)]
    index = int(np.argmin(costs))
    return max(0.0, float(costs[index])), index


class JointCoverPlanner:
    def __init__(self, policy):
        self.policy = policy
        self.witnesses = WitnessPlanner(policy.problem)
        self.decisions = []

    def complete(self, scans):
        coverage = copy.deepcopy(self.policy.coverage)
        for point in scans:
            coverage.observe_absence(point)
        return coverage

    def price(self, scans, targets):
        points = unique_points([*scans, *targets])
        itinerary = open_route(np.asarray(points), self.policy.position) if points else np.empty((0, 2))
        seconds = route_length(itinerary, self.policy.position) / 5 + 6 * len(self.policy.unknown_channels()) * len(scans)
        return seconds, itinerary

    def remaining_baseline(self):
        return [point.copy() for point in self.policy.route
                if not any(np.array_equal(point, previous) for previous in self.policy.coverage.observations)]

    def candidates(self, baseline, targets):
        points = [*baseline, *targets, self.policy.position]
        for point in [*targets, *self.policy.clear_positions[-8:]]:
            points.append(point)
            radius = np.linalg.norm(point)
            if radius > 1400:
                points.append(point * (min(radius, 1700) / radius))
                if self.policy.problem == 'q4':
                    points.append(point * (1880 / radius))
        return unique_points(points)

    def alive(self):
        witnesses = self.witnesses
        witnesses.update(self.policy.coverage.observations)
        for point in self.policy.coverage.clear_observations:
            witnesses.alive &= np.linalg.norm(witnesses.positions - point, axis=1) > 19.99
        witnesses.alive &= shapely.covers(self.policy.coverage.region, shapely.points(witnesses.positions))
        return witnesses.alive.copy()

    def extend_to_certificate(self, scans, baseline, targets):
        trial = self.complete(scans)
        remaining = [point for point in baseline if not any(np.array_equal(point, chosen) for chosen in scans)]
        while not trial.empty and remaining:
            choices = []
            price, itinerary = self.price(scans, targets)
            for index, point in enumerate(remaining):
                candidate = copy.deepcopy(trial)
                candidate.observe_absence(point)
                gain = trial.area - candidate.area
                extra, _position = insertion_cost(point, itinerary, self.policy.position)
                score = gain / (30 * len(self.policy.unknown_channels()) + extra + 1)
                choices.append((score, -extra, index, candidate))
            _score, _extra, index, trial = max(choices, key=lambda item: item[:2])
            scans.append(remaining.pop(index))
        if not trial.empty:
            return None
        return scans

    def choose(self, actions):
        policy = self.policy
        targets = [action[2] for action in actions]
        if policy.discovery_done or not policy.unknown_channels():
            price, itinerary = self.price([], targets)
            return JointPlan([], itinerary, price, price, 'known_only')
        baseline = self.remaining_baseline()
        if not self.complete(baseline).empty:
            raise ValueError('Original remaining network lacks a continuous completion certificate')
        baseline_price, baseline_itinerary = self.price(baseline, targets)
        best = JointPlan(baseline, baseline_itinerary, baseline_price, baseline_price, 'baseline')
        candidates = self.candidates(baseline, targets)
        matrix = np.array([self.witnesses.visible(point) for point in candidates])
        alive = self.alive()
        for fee_multiplier in (0.6, 1.0, 1.7):
            outstanding = alive.copy()
            scans = []
            available = list(range(len(candidates)))
            itinerary = open_route(np.asarray(unique_points(targets)), policy.position) if targets else np.empty((0, 2))
            for _iteration in range(len(baseline) + 8):
                if not outstanding.any() or not available:
                    break
                choices = []
                for index in available:
                    gain = int(np.sum(matrix[index] & outstanding))
                    if not gain:
                        continue
                    extra, insertion = insertion_cost(candidates[index], itinerary, policy.position)
                    score = gain / (fee_multiplier * 30 * len(policy.unknown_channels()) + extra + 1)
                    choices.append((score, -extra, index, insertion))
                if not choices:
                    break
                _score, _extra, index, insertion = max(choices, key=lambda item: item[:2])
                point = candidates[index]
                scans.append(point)
                if not any(np.array_equal(point, previous) for previous in itinerary):
                    itinerary = np.insert(itinerary, insertion, point, axis=0)
                outstanding &= ~matrix[index]
                available.remove(index)
            scans = self.extend_to_certificate(scans, baseline, targets)
            if scans is None:
                continue
            price, itinerary = self.price(scans, targets)
            if price < best.predicted_seconds - 0.1:
                best = JointPlan(scans, itinerary, price, baseline_price, f'joint_{fee_multiplier}')
        self.decisions.append({'actual_observations': len(policy.coverage.observations),
                               'position': policy.position.tolist(), 'known_actions': len(actions),
                               'baseline_scans': len(baseline), 'selected_scans': len(best.scans),
                               'predicted_seconds': best.predicted_seconds, 'baseline_seconds': baseline_price,
                               'mode': best.mode, 'scans': [point.tolist() for point in best.scans]})
        policy.stats['joint_cover_replans'] += 1
        policy.stats['joint_cover_changed'] += int(best.mode != 'baseline')
        return best
