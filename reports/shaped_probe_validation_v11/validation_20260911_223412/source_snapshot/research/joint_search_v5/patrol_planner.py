from __future__ import annotations

import copy

import numpy as np
import shapely


class CertifiedPatrol:
    def __init__(self, policy):
        self.policy = policy
        self.matrix = np.array([policy.search_planner.visible(point) for point in policy.route])
        self.revision = None
        self.proofs = []

    def refresh_witnesses(self):
        policy = self.policy
        planner = policy.search_planner
        planner.update(policy.coverage.observations)
        for point in policy.coverage.clear_observations:
            planner.alive &= np.linalg.norm(planner.positions - point, axis=1) > 19.99
        planner.alive &= shapely.covers(policy.coverage.region, shapely.points(planner.positions))

    def complete_with(self, indices, extra=None):
        trial = copy.deepcopy(self.policy.coverage)
        if extra is not None:
            trial.observe_absence(extra)
        for index in indices:
            trial.observe_absence(self.policy.route[index])
        return trial

    def reduce(self, targets):
        policy = self.policy
        revision = (len(policy.coverage.observations), len(policy.coverage.clear_observations), tuple(policy.remaining_stations))
        if revision == self.revision:
            return
        self.refresh_witnesses()
        remaining = policy.remaining_stations
        planner = policy.search_planner
        planner.candidates = policy.route[remaining]
        planner.matrix = self.matrix[remaining]
        proposed = planner.plan(policy.position, targets) if remaining else []
        selected = [remaining[int(np.argmin(np.linalg.norm(policy.route[remaining] - point, axis=1)))] for point in proposed]
        trial = self.complete_with(selected)
        available = [index for index in remaining if index not in selected]
        while not trial.empty and available:
            candidates = []
            for index in available:
                candidate = copy.deepcopy(trial)
                candidate.observe_absence(policy.route[index])
                cost = 400 + np.linalg.norm(policy.position - policy.route[index])
                score = (trial.area - candidate.area) / cost
                candidates.append((score, index, candidate))
            _score, index, trial = max(candidates, key=lambda item: item[0])
            selected.append(index)
            available.remove(index)
        if not trial.empty:
            raise ValueError('Remaining patrol has no complete continuous certificate')
        policy.stats['certified_station_prunes'] += len(remaining) - len(selected)
        remaining[:] = selected
        self.revision = (len(policy.coverage.observations), len(policy.coverage.clear_observations), tuple(remaining))
        self.proofs.append({'observed': len(policy.coverage.observations), 'planned_station_ids': selected.copy()})

    def replace_from_current_position(self):
        policy = self.policy
        if policy.discovery_done or not policy.unknown_channels() or not policy.remaining_stations:
            return False
        point = policy.position.copy()
        if any(np.linalg.norm(point - previous) < 1e-5 for previous in policy.coverage.observations):
            return False
        self.refresh_witnesses()
        remaining = policy.remaining_stations
        planner = policy.search_planner
        visible = planner.visible(point)
        coverage_count = np.sum(self.matrix[remaining], axis=0)
        ordered = sorted(remaining, key=lambda index: np.linalg.norm(point - policy.route[index]), reverse=True)
        for index in ordered:
            unique = planner.alive & self.matrix[index] & (coverage_count == 1)
            if np.any(unique & ~visible):
                continue
            kept = [other for other in remaining if other != index]
            trial = self.complete_with(kept, extra=point)
            if not trial.empty:
                continue
            policy.scan_unknown(point, force=True)
            remaining.remove(index)
            policy.stats['station_replacements'] += 1
            self.revision = None
            return True
        return False
