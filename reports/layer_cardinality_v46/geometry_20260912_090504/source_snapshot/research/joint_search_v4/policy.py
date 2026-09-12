from __future__ import annotations

import math

import numpy as np
from coverage import DirectionalBelief, DirectionalCoverage, OmniCoverage
from geometry import minimum_circle, open_route
from plan_geometry import (
    CLEAR_SUPPORT_RADIUS, disk_polygon, fallback_plan, hull_vertices, reception_proxy,
    small_clear_plan, source_proxy_hypotheses,
)
from plan_policy import PlanPolicy
from shapely.geometry import Point
from search_planner import WitnessPlanner


MODES = {
    'active': {'joint': False, 'detour': 0.0, 'scan_spacing': 450.0},
    'joint600': {'joint': True, 'detour': 600.0, 'scan_spacing': 450.0},
    'joint1200': {'joint': True, 'detour': 1200.0, 'scan_spacing': 450.0},
    'joint1800': {'joint': True, 'detour': 1800.0, 'scan_spacing': 600.0},
    'belief': {'joint': False, 'detour': 0.0, 'scan_spacing': 450.0, 'belief': True},
    'tour': {'joint': True, 'detour': 0.0, 'scan_spacing': 600.0, 'belief': True, 'tour': True},
    'tour_all': {'joint': True, 'detour': 0.0, 'scan_spacing': 600.0, 'belief': True, 'tour': True, 'target_radius': 2000},
    'short_probe': {'joint': True, 'detour': 0.0, 'scan_spacing': 650.0, 'belief': True, 'tour': True, 'target_radius': 2000, 'short_probe': True},
    'adaptive': {'joint': True, 'detour': 0.0, 'scan_spacing': 650.0, 'belief': True, 'tour': True, 'target_radius': 2000, 'adaptive': True},
    'adaptive_short': {'joint': True, 'detour': 0.0, 'scan_spacing': 650.0, 'belief': True, 'tour': True, 'target_radius': 2000, 'adaptive': True, 'short_probe': True},
    'dynamic': {'joint': True, 'detour': 0.0, 'scan_spacing': 400.0, 'belief': True, 'dynamic': True},
}


class JointSearchPolicy(PlanPolicy):
    def __init__(self, port, problem, stations, variant, network, mode='joint1200'):
        super().__init__(port, problem, stations, variant, network)
        self.options = MODES[mode]
        self.coverage = DirectionalCoverage() if problem == 'q4' else OmniCoverage()
        self.scan_positions = []
        self.coverage_events = []
        self.discovery_done = False
        self.active_target = None
        self.probe_history = {channel: [] for channel in self.states}
        self.direction_beliefs = {}
        self.clear_positions = []
        self.cluster_centers = []
        self.cluster_busy = False
        self.stats.update({'joint_targets': 0, 'scan_stops': 0, 'coverage_absent': 0,
                           'coverage_skips': 0, 'close_probes': 0, 'close_no_signal': 0,
                           'safe_pruned': 0, 'certified_fallbacks': 0})

    def clear(self, channel, position):
        success = super().clear(channel, position)
        if success:
            self.clear_positions.append(np.asarray(position).copy())
        return success

    def measure(self, channel, position, active=False, opportunistic=False):
        response = super().measure(channel, position, active, opportunistic)
        state = self.states[channel]
        if self.problem == 'q4' and self.options.get('belief') and state.status == 'DETECTED':
            belief = self.direction_beliefs.setdefault(channel, DirectionalBelief())
            state.region = belief.update(state.region, state.positives, state.negatives)
            self.snapshot(state)
        return response

    def center_radius(self, state):
        return minimum_circle(hull_vertices(state.region))

    def unknown_channels(self):
        return [state.channel for state in self.states.values() if state.status == 'UNKNOWN']

    def finish_discovery(self, reason):
        if reason not in ('coverage', 'full_route', 'maximum_count'):
            raise ValueError('No discovery stopping certificate')
        if reason == 'coverage' and not self.coverage.empty:
            raise ValueError('Nonempty directional hypothesis domain')
        if reason == 'maximum_count' and self.known_count() != 16:
            raise ValueError('Maximum source count not observed')
        for state in self.states.values():
            if state.status == 'UNKNOWN':
                state.status = 'ABSENT'
                self.stats['coverage_absent'] += int(reason == 'coverage')
        self.discovery_done = True
        self.stats['discovery_stop_reason'] = reason
        self.stats['discovery_end_seconds'] = self.virtual_seconds

    def scan_unknown(self, position, force=False):
        channels = self.unknown_channels()
        if not channels or self.discovery_done:
            return
        if not force:
            if self.scan_positions and min(np.linalg.norm(position - point) for point in self.scan_positions) < self.options['scan_spacing']:
                return
            if self.coverage.gain(position) < 15000.0:
                return
        for channel in sorted(channels, reverse=bool(len(self.scan_positions) % 2)):
            self.measure(channel, position)
            if self.known_count() == 16:
                self.finish_discovery('maximum_count')
                return
        self.scan_positions.append(np.asarray(position).copy())
        self.coverage.observe_absence(position)
        self.stats['scan_stops'] += 1
        self.coverage_events.append((tuple(self.unknown_channels()), self.coverage.observations[-1].copy()))
        if self.coverage.empty:
            self.finish_discovery('coverage')

    def reuse_directions(self):
        candidates = []
        for state in self.states.values():
            if state.status != 'DETECTED' or state.channel == self.active_target:
                continue
            if any(np.linalg.norm(self.position - previous) < 1e-5 for previous in state.measured_positions):
                continue
            center, radius = self.center_radius(state)
            if radius <= CLEAR_SUPPORT_RADIUS or state.region.distance(Point(self.position)) > 1000:
                continue
            previous_vector = center - state.positives[-1][0]
            new_vector = center - self.position
            sine = abs(previous_vector[0] * new_vector[1] - previous_vector[1] * new_vector[0]) / max(1e-9, np.linalg.norm(previous_vector) * np.linalg.norm(new_vector))
            if sine < 0.2 and np.linalg.norm(new_vector) > 100:
                continue
            if self.problem == 'q4':
                probability = reception_proxy(self.hypotheses(state), self.position)
                if probability < 0.25:
                    continue
            candidates.append((radius * max(sine, 0.2), state.channel))
        for _score, channel in sorted(candidates, reverse=True)[:6]:
            self.measure(channel, self.position, opportunistic=True)

    def reuse_stop(self):
        if self.options['joint']:
            self.scan_unknown(self.position)
            self.reuse_directions()
            self.cluster_probe()

    def cluster_probe(self):
        if (not self.options.get('adaptive') or self.problem != 'q4' or self.cluster_busy
                or self.discovery_done or not self.unknown_channels() or len(self.cluster_centers) >= 2):
            return
        neighbors = [point for point in self.clear_positions if np.linalg.norm(point - self.position) <= 50]
        if len(neighbors) < 2 or any(np.linalg.norm(self.position - center) < 200 for center in self.cluster_centers):
            return
        center = np.mean(neighbors, axis=0)
        self.cluster_centers.append(center)
        self.cluster_busy = True
        for offset in ([25, 0], [0, 25], [-25, 0], [0, -25]):
            self.scan_unknown(center + offset, force=True)
            self.reuse_directions()
            if self.discovery_done:
                break
        self.cluster_busy = False

    def choose_probe(self, state):
        center, radius = self.center_radius(state)
        anchor = state.positives[-1][0]
        radial = center - anchor
        radial /= max(np.linalg.norm(radial), 1e-9)
        tangent = np.array([-radial[1], radial[0]])
        offset = min(220.0, max(35.0, radius * 0.4))
        anchor_distance = np.linalg.norm(center - anchor)
        back = min(250.0, max(25.0, 0.2 * anchor_distance))
        candidates = [center - back * radial + sign * offset * tangent for sign in (-1, 1)]
        candidates += [center - 0.5 * anchor_distance * radial + sign * offset * tangent for sign in (-1, 1)]
        if self.options.get('short_probe') and radius > 130:
            forward = min(350.0, 0.6 * anchor_distance)
            lateral = min(180.0, max(60.0, 0.35 * forward))
            candidates = [anchor + forward * radial + sign * lateral * tangent for sign in (-1, 1)]
        if radius < 100:
            candidates += [center + 40 * np.array([math.cos(angle), math.sin(angle)])
                           for angle in np.arange(8) * math.pi / 4]
        bank = source_proxy_hypotheses(state.region, state.positives, state.negatives,
                                       self.problem == 'q4', sample_count=20)
        choices = []
        for candidate in candidates:
            if any(np.linalg.norm(candidate - previous) < 2.0 for previous in state.measured_positions):
                continue
            distances = np.linalg.norm(hull_vertices(state.region) - candidate, axis=1)
            if self.problem == 'q3' and distances.max() > 999.9 and not self.options.get('short_probe'):
                continue
            receive = 1.0 if self.problem == 'q3' else reception_proxy(bank, candidate)
            from_anchor = center - anchor
            from_candidate = center - candidate
            sine = abs(from_anchor[0] * from_candidate[1] - from_anchor[1] * from_candidate[0]) / max(1e-9, np.linalg.norm(from_anchor) * np.linalg.norm(from_candidate))
            expected_radius = min(radius, math.tan(math.radians(1.005)) * np.linalg.norm(from_candidate) / max(sine, 0.08))
            score = (np.linalg.norm(candidate - self.position) + np.linalg.norm(candidate - center)) / 5 + 6
            score += 2 * expected_radius + (1 - receive) * (80 + min(300, radius))
            if state.consecutive_no_signal:
                score += 50 * sum(np.linalg.norm(candidate - negative) < 100 for negative in state.negatives)
            choices.append((float(score), candidate))
        return min(choices, key=lambda item: item[0])[1] if choices else None

    def execute_clear_plan(self, state, plan, opportunistic=False):
        if not plan.certified:
            raise ValueError('Uncertified clearance plan')
        self.certificates.append((state.region, plan.centers, plan.certificate_kind))
        remaining = list(range(len(plan.centers)))
        while remaining:
            viable = [index for index in remaining if state.region.distance(Point(plan.centers[index])) <= 20.000001]
            self.stats['safe_pruned'] += len(remaining) - len(viable)
            if not viable:
                break
            selected = min(viable, key=lambda index: float(np.linalg.norm(plan.centers[index] - self.position)))
            remaining = [index for index in viable if index != selected]
            if self.clear(state.channel, plan.centers[selected]):
                self.stats['mec_clears' if plan.certificate_kind == 'mec' else 'multi_circle_clears'] += 1
                self.reuse_stop()
                return
        raise ValueError('Certified clearance exhausted')

    def localize(self, state):
        self.active_target = state.channel
        for iteration in range(9):
            if state.status == 'NEAR':
                self.clear_near(state)
                self.reuse_stop()
                break
            plan = small_clear_plan(state.region, self.position, 4)
            if plan is not None and (plan.certificate_kind == 'mec' or len(plan.centers) <= 2 or iteration >= 3):
                self.execute_clear_plan(state, plan)
                break
            probe = self.choose_probe(state) if iteration < 8 and state.consecutive_no_signal < 3 else None
            if probe is None:
                plan = plan or fallback_plan(state.region, state.positives[0][1], self.position)
                self.stats['certified_fallbacks'] += 1
                self.execute_clear_plan(state, plan)
                break
            response = self.measure(state.channel, probe, active=True)
            self.probe_history[state.channel].append(np.asarray(probe).copy())
            self.stats['close_probes'] += 1
            self.stats['close_no_signal'] += int(response['result'] == 'no_signal')
            self.reuse_stop()
        if state.status != 'CLEARED':
            raise ValueError('Unfinished localization')
        self.active_target = None

    def candidate_target(self, next_station):
        candidates = []
        for state in self.states.values():
            if state.status not in ('DETECTED', 'NEAR'):
                continue
            center = state.near_position if state.status == 'NEAR' else self.center_radius(state)[0]
            distance = np.linalg.norm(center - self.position)
            detour = distance + np.linalg.norm(center - next_station) - np.linalg.norm(self.position - next_station)
            if detour <= self.options['detour'] and distance <= 1700:
                candidates.append((distance + 0.4 * detour, state.channel))
        return self.states[min(candidates)[1]] if candidates else None

    def run(self):
        if self.options.get('dynamic'):
            return self.run_dynamic()
        if self.options.get('tour'):
            return self.run_tour()
        for station in self.route:
            if self.discovery_done:
                break
            if self.options['joint']:
                for _attempt in range(16):
                    target = self.candidate_target(station)
                    if target is None or self.discovery_done:
                        break
                    self.stats['joint_targets'] += 1
                    self.localize(target)
            if self.discovery_done:
                break
            self.scan_unknown(station, force=True)
            self.stats['visited_stations'] += 1
            self.reuse_directions()
        if not self.discovery_done:
            self.finish_discovery('full_route')
        while True:
            unresolved = [state for state in self.states.values() if state.status in ('DETECTED', 'NEAR')]
            if not unresolved:
                break
            self.localize(self.select_unresolved(unresolved))
        return self.result()

    def result(self):
        return {'stats': self.stats, 'virtual_seconds': self.virtual_seconds,
                'declared_absent': [state.channel for state in self.states.values() if state.status == 'ABSENT'],
                'cleared_channels': [state.channel for state in self.states.values() if state.status == 'CLEARED'],
                'snapshots': self.snapshots, 'certificates': self.certificates}

    def run_tour(self):
        remaining = list(range(len(self.route)))
        self.scan_unknown(self.position, force=True)
        remaining.remove(0)
        outer_first = self.options.get('adaptive') and self.problem == 'q4' and self.known_count() <= 2
        self.stats['outer_first'] = bool(outer_first)
        for _iteration in range(120):
            unresolved = [state for state in self.states.values() if state.status in ('DETECTED', 'NEAR')]
            if self.discovery_done and not unresolved:
                return self.result()
            if not remaining and not self.discovery_done:
                self.finish_discovery('full_route')
            if self.problem == 'q3' and self.options.get('adaptive') and not self.discovery_done:
                irrelevant = [index for index in remaining if self.coverage.region.intersection(
                    disk_polygon(self.route[index], 999.9)).is_empty]
                remaining = [index for index in remaining if index not in irrelevant]
                self.stats['coverage_skips'] += len(irrelevant)
                if not remaining:
                    if not self.coverage.empty:
                        raise ValueError('Pruned search stations without completing coverage')
                    self.finish_discovery('coverage')
            stops = []
            for state in unresolved:
                center, radius = (state.near_position, 0) if state.status == 'NEAR' else self.center_radius(state)
                if radius <= self.options.get('target_radius', 130) or not remaining or self.discovery_done:
                    stops.append(('target', state.channel, center))
            if not self.discovery_done:
                search_ids = remaining
                if outer_first and any(index >= 9 for index in remaining):
                    search_ids = [index for index in remaining if index >= 9]
                stops.extend(('station', index, self.route[index]) for index in search_ids)
            if not stops:
                return self.result()
            destinations = np.asarray([stop[2] for stop in stops])
            route = open_route(destinations, self.position)
            selected = int(np.argmin(np.linalg.norm(destinations - route[0], axis=1)))
            kind, identifier, destination = stops[selected]
            if kind == 'target':
                self.stats['joint_targets'] += 1
                self.localize(self.states[identifier])
            else:
                self.scan_unknown(destination, force=True)
                self.stats['visited_stations'] += 1
                self.reuse_directions()
                remaining.remove(identifier)
        raise ValueError('Tour action limit exceeded')

    def run_dynamic(self):
        planner = WitnessPlanner(self.problem)
        self.scan_unknown(self.position, force=True)
        for _iteration in range(120):
            unresolved = [state for state in self.states.values() if state.status in ('DETECTED', 'NEAR')]
            if self.discovery_done and not unresolved:
                return self.result()
            stops = [('target', state.channel,
                      state.near_position if state.status == 'NEAR' else self.center_radius(state)[0])
                     for state in unresolved]
            if not self.discovery_done:
                if self.stats['visited_stations'] >= 40:
                    for station in self.route:
                        self.scan_unknown(station, force=True)
                        if self.discovery_done:
                            break
                    if not self.discovery_done:
                        self.finish_discovery('full_route')
                    continue
                planner.update(self.coverage.observations)
                search_positions = planner.plan(self.position, [stop[2] for stop in stops])
                if not search_positions:
                    if self.problem == 'q3':
                        search_positions = [np.asarray(self.coverage.region.representative_point().coords[0])]
                    else:
                        index = max(range(self.coverage.bins), key=lambda item: self.coverage.regions[item].area)
                        region = self.coverage.regions[index]
                        if region.is_empty:
                            index = next(item for item, region in enumerate(self.coverage.regions) if not region.is_empty)
                            region = self.coverage.regions[index]
                        source_proxy = np.asarray(region.representative_point().coords[0])
                        angle = math.radians(index * 360.0 / self.coverage.bins)
                        search_positions = [source_proxy + 700 * np.array([math.cos(angle), math.sin(angle)])]
                stops.extend(('search', index, point) for index, point in enumerate(search_positions))
            if not stops:
                return self.result()
            destinations = np.asarray([stop[2] for stop in stops])
            route = open_route(destinations, self.position)
            selected = int(np.argmin(np.linalg.norm(destinations - route[0], axis=1)))
            kind, identifier, destination = stops[selected]
            if kind == 'target':
                self.stats['joint_targets'] += 1
                self.localize(self.states[identifier])
            else:
                self.scan_unknown(destination, force=True)
                self.stats['visited_stations'] += 1
                self.reuse_directions()
        raise ValueError('Dynamic search limit exceeded')
