from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass

import numpy as np
from geometry import open_route


@dataclass(frozen=True)
class PairCommitment:
    stops: tuple
    removed: tuple
    kept: tuple
    baseline_seconds: float
    proposed_seconds: float


def route_seconds(points, start):
    unique = []
    for point in points:
        if not any(np.array_equal(point, previous) for previous in unique):
            unique.append(np.asarray(point))
    if not unique:
        return 0.0
    itinerary = open_route(np.asarray(unique), start)
    return float(np.linalg.norm(np.diff(np.vstack((start, itinerary)), axis=0), axis=1).sum() / 5)


class PairedScanPlanner:
    def __init__(self, policy):
        self.policy = policy
        self.decisions = []

    def candidates(self, stops):
        policy = self.policy
        choices = []
        for stop in stops:
            kind, channel, point, extra = stop
            if kind not in ('near', 'clear', 'anchor_clear', 'probe', 'v_probe', 'v_mirror'):
                continue
            if kind == 'clear' and (not extra.certified or len(extra.centers) != 1):
                continue
            if any(np.linalg.norm(point - previous) < 50 for previous in policy.coverage.observations):
                continue
            choices.append(stop)
        choices.sort(key=lambda stop: (float(np.linalg.norm(stop[2] - policy.position)), stop[1]))
        choices = choices[:8]
        if all(np.linalg.norm(policy.position - previous) >= 50 for previous in policy.coverage.observations):
            choices.append(('scan_here', None, policy.position.copy(), None))
        return choices

    def complete(self, stops, kept):
        trial = copy.deepcopy(self.policy.coverage)
        for stop in stops:
            trial.observe_absence(stop[2])
        for index in kept:
            trial.observe_absence(self.policy.route[index])
        return trial

    def proposed_price(self, pair, removed, stops, unknown_count):
        policy = self.policy
        consumed = {(stop[0], stop[1]) for stop in pair}
        tail = [stop[2] for stop in stops
                if (stop[0], stop[1]) not in consumed
                and not (stop[0] == 'fixed' and stop[1] in removed)]
        prefix = float((np.linalg.norm(pair[0][2] - policy.position)
                        + np.linalg.norm(pair[1][2] - pair[0][2])) / 5)
        scan_count = 2 + len(policy.remaining_stations) - len(removed)
        return prefix + route_seconds(tail, pair[1][2]) + 6 * unknown_count * scan_count

    def choose(self, stops):
        policy = self.policy
        if policy.discovery_done or not policy.unknown_channels() or not policy.remaining_stations:
            return None
        if any(stop[0] == 'optical' for stop in stops):
            return None
        candidates = self.candidates(stops)
        if len(candidates) < 2:
            return None
        policy.patrol.refresh_witnesses()
        witnesses = policy.search_planner
        remaining = tuple(policy.remaining_stations)
        coverage_count = policy.patrol.matrix[list(remaining)].sum(axis=0)
        missing_sets = []
        for size in (1, 2):
            for removed in itertools.combinations(remaining, size):
                removed_count = policy.patrol.matrix[list(removed)].sum(axis=0)
                missing = np.flatnonzero(witnesses.alive & (coverage_count == removed_count))
                if len(missing):
                    missing_sets.append((removed, missing))
        visibility = [witnesses.visible(stop[2]) for stop in candidates]
        unknown_count = len(policy.unknown_channels())
        baseline = route_seconds([stop[2] for stop in stops], policy.position) + 6 * unknown_count * len(remaining)
        options = []
        event = {'seconds': policy.virtual_seconds, 'candidate_points': len(candidates),
                 'pairs': 0, 'witness_joint_passes': 0, 'price_passes': 0,
                 'continuous_checks': 0, 'continuous_complete': 0, 'selected': False}
        for first, second in itertools.combinations(range(len(candidates)), 2):
            if candidates[first][1] == candidates[second][1]:
                continue
            if np.linalg.norm(candidates[first][2] - candidates[second][2]) < 50:
                continue
            event['pairs'] += 1
            for removed, missing in missing_sets:
                first_hits = visibility[first][missing]
                second_hits = visibility[second][missing]
                if first_hits.all() or second_hits.all() or not np.all(first_hits | second_hits):
                    continue
                event['witness_joint_passes'] += 1
                orders = [(candidates[first], candidates[second]), (candidates[second], candidates[first])]
                ranked = [(self.proposed_price(order, removed, stops, unknown_count), order) for order in orders]
                price, pair = min(ranked, key=lambda item: item[0])
                if baseline - price < 1.0:
                    continue
                event['price_passes'] += 1
                kept = tuple(index for index in remaining if index not in removed)
                options.append(PairCommitment(pair, removed, kept, baseline, price))
        for proposal in sorted(options, key=lambda option: option.proposed_seconds)[:12]:
            event['continuous_checks'] += 1
            if not self.complete(proposal.stops, proposal.kept).empty:
                continue
            event['continuous_complete'] += 1
            event['selected'] = True
            self.decisions.append(event)
            return proposal
        self.decisions.append(event)
        return None
