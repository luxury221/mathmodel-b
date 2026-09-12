from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from paired_planner import route_seconds
from probe_constraints import certify_probe, missing_heading, solve_point


@dataclass
class ProbeCommitment:
    channel: int
    station: int
    kept: tuple
    certificate: object
    original_destination: np.ndarray
    baseline_seconds: float
    proposed_seconds: float
    continuation_penalty: float


class CoverageProbePlanner:
    def __init__(self, policy):
        self.policy = policy
        self.decisions = []

    def complete(self, point, kept):
        trial = copy.deepcopy(self.policy.coverage)
        if point is not None:
            trial.observe_absence(point)
        for index in kept:
            trial.observe_absence(self.policy.route[index])
        return trial

    def fresh(self, point, channel):
        policy = self.policy
        previous = [*policy.states[channel].measured_positions, *policy.coverage.observations]
        return all(np.linalg.norm(point - prior) >= 50 for prior in previous)

    def synthesize(self, state, original, first, station, missing, event):
        policy = self.policy
        template = original.candidates[first]
        mirror = original.candidates[1 - first]
        anchor = original.anchor
        bearing = next(bearing for point, bearing in state.positives if np.array_equal(point, anchor))
        kept = tuple(index for index in policy.remaining_stations if index != station)
        if policy.coverage_mode == 'station_bind':
            certificate = certify_probe(state.region, anchor, policy.route[station], mirror, bearing)
            if certificate is None or not self.fresh(certificate.candidates[0], state.channel):
                return None
            event['probe_certified'] += 1
            event['continuous_checks'] += 1
            if self.complete(certificate.candidates[0], kept).empty:
                event['continuous_passes'] += 1
                return certificate
            return None
        positions = policy.search_planner.positions[missing].copy()
        headings = policy.search_planner.headings[missing].copy()
        for round_index in range(4):
            event['solver_calls'] += 1
            certificate, status = solve_point(state.region, anchor, template, mirror, bearing, positions, headings)
            event['solver_statuses'][status] = event['solver_statuses'].get(status, 0) + 1
            if certificate is None or not self.fresh(certificate.candidates[0], state.channel):
                return None
            event['probe_certified'] += 1
            event['continuous_checks'] += 1
            point = certificate.candidates[0]
            trial = self.complete(point, kept)
            if trial.empty:
                event['continuous_passes'] += 1
                return certificate
            if round_index == 3:
                return None
            receivers = [*policy.coverage.observations, point, *(policy.route[index] for index in kept)]
            parts = list(trial.region.geoms) if hasattr(trial.region, 'geoms') else [trial.region]
            added = 0
            for part in sorted(parts, key=lambda part: part.area, reverse=True):
                location = np.asarray(part.representative_point().coords[0])
                heading = missing_heading(location, receivers)
                if heading is None:
                    continue
                if len(positions) and np.any((np.linalg.norm(positions - location, axis=1) < 1e-4)
                                            & (headings @ heading > 1 - 1e-10)):
                    continue
                positions = np.vstack((positions, location))
                headings = np.vstack((headings, heading))
                added += 1
                if added == 4:
                    break
            event['repair_constraints'] += added
            if not added:
                return None
        return None

    def choose(self, stops):
        policy = self.policy
        if policy.discovery_done or not policy.unknown_channels() or not policy.remaining_stations:
            return None
        if any(stop[0] == 'optical' for stop in stops):
            return None
        actions = [stop for stop in stops if stop[0] == 'v_probe']
        actions.sort(key=lambda stop: (float(np.linalg.norm(stop[2] - policy.position)), stop[1]))
        if not actions:
            return None
        policy.patrol.refresh_witnesses()
        coverage_count = policy.patrol.matrix[policy.remaining_stations].sum(axis=0)
        missing = {index: np.flatnonzero(policy.search_planner.alive & policy.patrol.matrix[index] & (coverage_count == 1))
                   for index in policy.remaining_stations}
        unknown_count = len(policy.unknown_channels())
        baseline = route_seconds([stop[2] for stop in stops], policy.position) + 6 * unknown_count * len(policy.remaining_stations)
        event = {'seconds': policy.virtual_seconds, 'candidates': 0, 'solver_calls': 0, 'solver_statuses': {},
                 'probe_certified': 0, 'continuous_checks': 0, 'continuous_passes': 0,
                 'repair_constraints': 0, 'price_passes': 0, 'selected': False}
        options = []
        for _kind, channel, destination, extra in actions[:4]:
            state = policy.states[channel]
            _mirror, original = extra
            bank = policy.hypotheses(state)
            original_first = int(np.argmin(np.linalg.norm(original.candidates - destination, axis=1)))
            score_scale = 5 if bank is None else 1
            continuation_old = policy.score(state, original, original_first, bank) / score_scale - float(np.linalg.norm(destination - policy.position)) / 5
            tail = [stop for stop in stops if not (stop[0] == 'v_probe' and stop[1] == channel)]
            for first in (original_first, 1 - original_first):
                stations = sorted(policy.remaining_stations, key=lambda index: float(np.linalg.norm(policy.route[index] - original.candidates[first])))[:2]
                for station in stations:
                    event['candidates'] += 1
                    certificate = self.synthesize(state, original, first, station, missing[station], event)
                    if certificate is None:
                        continue
                    point = certificate.candidates[0]
                    kept = tuple(index for index in policy.remaining_stations if index != station)
                    destinations = [stop[2] for stop in tail if not (stop[0] == 'fixed' and stop[1] == station)]
                    travel = float(np.linalg.norm(point - policy.position)) / 5
                    continuation_new = policy.score(state, certificate, 0, bank) / score_scale - travel
                    penalty = max(0.0, continuation_new - continuation_old)
                    price = travel + route_seconds(destinations, point) + 6 * unknown_count * len(policy.remaining_stations) + penalty
                    if baseline - price < 1:
                        continue
                    event['price_passes'] += 1
                    options.append(ProbeCommitment(channel, station, kept, certificate, destination.copy(), baseline, price, penalty))
        self.decisions.append(event)
        if not options:
            return None
        event['selected'] = True
        return min(options, key=lambda option: option.proposed_seconds)
