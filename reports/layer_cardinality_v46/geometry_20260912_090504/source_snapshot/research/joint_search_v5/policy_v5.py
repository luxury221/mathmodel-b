from __future__ import annotations

import math

import numpy as np
import shapely
from coverage import OmniCoverage
from coverage_certificate import TriangleCoverage
from geometry import minimum_circle, open_route, ring
from plan_geometry import disk_polygon, fallback_plan, hull_vertices, reception_proxy, small_clear_plan
from policy import JointSearchPolicy
from search_planner import WitnessPlanner
from probe_geometry import certified_v_probe
from shapely.geometry import Point
from patrol_planner import CertifiedPatrol


MODES = ('source_tour', 'action_tour', 'fixed_union', 'guaranteed_fixed', 'guaranteed_dynamic', 'certified_fixed', 'certified_center', 'chase', 'perimeter_chase')


class OmniOpticalCoverage(OmniCoverage):
    def __init__(self):
        super().__init__()
        self.clear_observations = []

    def observe_clear_absence(self, position):
        self.region = self.region.difference(disk_polygon(position, 19.99))
        self.clear_observations.append(np.asarray(position).copy())


class ContinuousPolicy(JointSearchPolicy):
    def __init__(self, port, problem, stations, variant, network, mode='action_tour'):
        super().__init__(port, problem, stations, variant, network, mode='tour_all')
        self.options = {**self.options, 'scan_spacing': 400.0}
        self.continuous_mode = mode
        self.coverage = TriangleCoverage() if problem == 'q4' else OmniOpticalCoverage()
        self.search_planner = WitnessPlanner(problem)
        self.clear_event_log = []
        self.sweeping = False
        self.sweep_positions = []
        self.active_counts = {channel: 0 for channel in self.states}
        self.pending_probes = {}
        self.probe_certificates = []
        self.remaining_stations = list(range(1, len(self.route)))
        self.patrol = CertifiedPatrol(self)
        self.stats.update({'optical_unknown_attempts': 0, 'optical_unknown_discoveries': 0,
                           'optical_completion_stops': 0, 'common_optical_sweeps': 0,
                           'one_step_probes': 0, 'coverage_fallbacks': 0})
        self.stats.update({'v_probes': 0, 'v_mirrors': 0, 'anchor_optical_probes': 0})
        self.stats.update({'certified_station_prunes': 0, 'station_replacements': 0, 'optical_absent': 0})

    @property
    def v_enabled(self):
        return self.continuous_mode.startswith('guaranteed') or self.continuous_mode in ('certified_fixed', 'chase', 'perimeter_chase')

    def measure(self, channel, position, active=False, opportunistic=False):
        response = super().measure(channel, position, active, opportunistic)
        if response['result'] == 'direction' and self.v_enabled:
            state = self.states[channel]
            state.region = state.region.difference(disk_polygon(position, 4.999))
            self.snapshot(state)
        return response

    def common_clear(self, position, all_unresolved=False):
        channels = [state.channel for state in self.states.values()
                    if state.status not in ('CLEARED', 'ABSENT') and (all_unresolved or state.status == 'UNKNOWN')]
        self.sweeping = True
        for channel in channels:
            was_unknown = self.states[channel].status == 'UNKNOWN'
            state = self.states[channel]
            if not was_unknown and state.status == 'DETECTED' and state.region.distance(Point(position)) > 20.000001:
                continue
            self.stats['optical_unknown_attempts'] += int(was_unknown)
            success = self.clear(channel, position)
            self.stats['optical_unknown_discoveries'] += int(success and was_unknown)
        self.sweeping = False
        if self.unknown_channels():
            self.coverage.observe_clear_absence(position)
            self.clear_event_log.append((tuple(self.unknown_channels()), np.asarray(position).copy()))
        self.stats['common_optical_sweeps'] += 1
        self.check_discovery()

    def check_discovery(self):
        if self.discovery_done:
            return
        if self.known_count() == 16:
            self.finish_discovery('maximum_count')
        elif self.coverage.empty:
            self.finish_discovery('coverage')
        elif not self.unknown_channels():
            self.discovery_done = True
            self.stats['discovery_stop_reason'] = 'all_channel_decisions'
            self.stats['discovery_end_seconds'] = self.virtual_seconds

    def clear(self, channel, position):
        was_near = self.states[channel].status == 'NEAR'
        if self.states[channel].status == 'UNKNOWN':
            response = self.port.clear(position, channel)
            self.account(position, channel, 'clear', response['result'])
            state = self.states[channel]
            if response['result'] == 'success':
                state.status = 'CLEARED'
                self.clear_positions.append(np.asarray(position).copy())
                success = True
            elif response['result'] == 'no_target_in_range':
                state.region = state.region.difference(disk_polygon(position, 19.99))
                self.snapshot(state)
                if state.region.is_empty:
                    state.status = 'ABSENT'
                    self.stats['optical_absent'] += 1
                success = False
            else:
                raise ValueError('Unexpected optical result')
        else:
            success = super().clear(channel, position)
        if not success or self.sweeping:
            return success
        clustered = sum(np.linalg.norm(previous - position) <= 50 for previous in self.clear_positions) >= 2
        local_chase = self.continuous_mode in ('chase', 'perimeter_chase')
        limit, spacing = (6, 18) if local_chase else (2, 200)
        if ((was_near or clustered) and len(self.sweep_positions) < limit
                and all(np.linalg.norm(previous - position) >= spacing for previous in self.sweep_positions)):
            self.sweep_positions.append(np.asarray(position).copy())
            self.common_clear(position, all_unresolved=True)
        return True

    def reuse_stop(self):
        if self.continuous_mode.startswith('certified'):
            self.patrol.replace_from_current_position()
        else:
            if self.continuous_mode in ('chase', 'perimeter_chase'):
                if not self.scan_positions or min(np.linalg.norm(self.position - point) for point in self.scan_positions) >= 650:
                    self.scan_unknown(self.position, force=True)
            else:
                self.scan_unknown(self.position)
        self.reuse_directions()
        self.check_discovery()

    def optical_components(self):
        region = self.coverage.region
        if region.is_empty or region.area > 30000:
            return []
        parts = list(region.geoms) if hasattr(region, 'geoms') else [region]
        if len(parts) > 12:
            return []
        centers = []
        for part in parts:
            center, radius = minimum_circle(hull_vertices(part))
            if radius > 19.95:
                return []
            centers.append(center)
        return centers

    def next_target_action(self, state):
        if state.status == 'NEAR':
            return 'near', state.near_position, None
        plan = small_clear_plan(state.region, self.position, 4)
        if plan is not None:
            return 'clear', plan.centers[0], plan
        if self.v_enabled and self.active_counts[state.channel] < 16:
            if state.channel in self.pending_probes:
                return 'v_mirror', self.pending_probes[state.channel], None
            anchor_candidates = []
            for anchor, bearing in state.positives:
                if state.region.distance(Point(anchor)) < 19.9:
                    anchor_candidates.append((float(np.linalg.norm(self.position - anchor)), 'anchor_clear', anchor, None))
                    continue
                probe = certified_v_probe(state.region, anchor, bearing)
                if probe is None:
                    continue
                bank = self.hypotheses(state)
                costs = [float(np.linalg.norm(candidate - self.position))
                         + 2 * probe.forward * (1 - reception_proxy(bank, candidate)) for candidate in probe.candidates]
                first = int(np.argmin(costs))
                anchor_candidates.append((costs[first], 'v_probe', probe.candidates[first],
                                          (probe.candidates[1 - first], probe)))
            if anchor_candidates:
                _cost, action, destination, extra = min(anchor_candidates, key=lambda item: item[0])
                return action, destination, extra
        probe = None
        if self.active_counts[state.channel] < 6 and state.consecutive_no_signal < 3:
            probe = self.choose_probe(state)
        if probe is not None:
            return 'probe', probe, None
        plan = fallback_plan(state.region, state.positives[0][1], self.position)
        return 'clear', plan.centers[0], plan

    def search_positions(self, targets):
        planner = self.search_planner
        planner.update(self.coverage.observations)
        for point in self.coverage.clear_observations:
            planner.alive &= np.linalg.norm(planner.positions - point, axis=1) > 19.99
        planner.alive &= shapely.covers(self.coverage.region, shapely.points(planner.positions))
        positions = planner.plan(self.position, targets)
        if positions:
            return positions
        region = self.coverage.region
        parts = list(region.geoms) if hasattr(region, 'geoms') else [region]
        part = max(parts, key=lambda item: item.area)
        center, radius = minimum_circle(hull_vertices(part))
        if self.problem == 'q3':
            return [center if radius < 950 else np.asarray(part.representative_point().coords[0])]
        if radius <= 320:
            return list(ring(max(30.0, 2.05 * radius), 3) + center)
        representative = np.asarray(part.representative_point().coords[0])
        return list(ring(600.0, 3) + representative)

    def fallback_discovery(self):
        self.stats['coverage_fallbacks'] += 1
        for station in self.route:
            if self.discovery_done:
                return
            self.scan_unknown(station, force=True)
            self.reuse_directions()
        if not self.discovery_done:
            self.finish_discovery('full_route')

    def run(self):
        if self.continuous_mode in ('chase', 'perimeter_chase'):
            return self.run_chase()
        self.common_clear(self.position, all_unresolved=True)
        self.scan_unknown(self.position, force=True)
        remaining = self.remaining_stations
        for _iteration in range(250):
            self.check_discovery()
            unresolved = [state for state in self.states.values() if state.status in ('DETECTED', 'NEAR')]
            if self.discovery_done and not unresolved:
                return self.result()
            stops = []
            for state in unresolved:
                if self.continuous_mode in ('action_tour', 'guaranteed_fixed', 'guaranteed_dynamic', 'certified_fixed'):
                    action, destination, plan = self.next_target_action(state)
                    stops.append((action, state.channel, destination, plan))
                else:
                    destination = state.near_position if state.status == 'NEAR' else self.center_radius(state)[0]
                    stops.append(('target', state.channel, destination, None))
            if not self.discovery_done:
                if self.stats['visited_stations'] >= 40:
                    self.fallback_discovery()
                    continue
                if self.continuous_mode.startswith('certified'):
                    self.patrol.reduce([stop[2] for stop in stops])
                optical = self.optical_components()
                if optical:
                    stops.extend(('optical', index, point, None) for index, point in enumerate(optical))
                elif self.continuous_mode in ('fixed_union', 'guaranteed_fixed', 'certified_fixed', 'certified_center'):
                    if not remaining:
                        self.fallback_discovery()
                        continue
                    stops.extend(('fixed', index, self.route[index], None) for index in remaining)
                else:
                    positions = self.search_positions([stop[2] for stop in stops])
                    stops.extend(('search', index, point, None) for index, point in enumerate(positions))
            if not stops:
                raise ValueError('No legal continuation')
            destinations = np.asarray([stop[2] for stop in stops])
            route = open_route(destinations, self.position)
            selected = int(np.argmin(np.linalg.norm(destinations - route[0], axis=1)))
            action, identifier, destination, plan = stops[selected]
            if action in ('fixed', 'search'):
                self.scan_unknown(destination, force=True)
                self.stats['visited_stations'] += 1
                self.reuse_directions()
                if action == 'fixed':
                    remaining.remove(identifier)
            elif action == 'optical':
                self.stats['optical_completion_stops'] += 1
                self.common_clear(destination)
                self.reuse_directions()
            elif action == 'target':
                self.stats['joint_targets'] += 1
                self.localize(self.states[identifier])
            elif action == 'near':
                self.clear_near(self.states[identifier])
                self.reuse_stop()
            elif action == 'clear':
                self.execute_clear_plan(self.states[identifier], plan)
            elif action == 'probe':
                self.measure(identifier, destination, active=True)
                self.active_counts[identifier] += 1
                self.stats['one_step_probes'] += 1
                self.reuse_stop()
            elif action == 'anchor_clear':
                self.stats['anchor_optical_probes'] += 1
                self.clear(identifier, destination)
                self.reuse_stop()
            elif action == 'v_probe':
                mirror, certificate = plan
                self.probe_certificates.append(certificate)
                response = self.measure(identifier, destination, active=True)
                self.active_counts[identifier] += 1
                self.stats['v_probes'] += 1
                if response['result'] == 'no_signal':
                    self.pending_probes[identifier] = mirror
                self.reuse_stop()
            elif action == 'v_mirror':
                response = self.measure(identifier, destination, active=True)
                self.active_counts[identifier] += 1
                self.stats['v_mirrors'] += 1
                self.pending_probes.pop(identifier)
                if response['result'] == 'no_signal':
                    raise ValueError('Both members of a guaranteed V-probe pair failed')
                self.reuse_stop()
            else:
                raise ValueError('Unknown scheduled action')
        raise ValueError('Continuous search action limit exceeded')

    def resolve_with_v(self, state):
        for _iteration in range(40):
            if state.status == 'CLEARED':
                return
            action, destination, plan = self.next_target_action(state)
            if action == 'near':
                self.clear_near(state)
                self.reuse_stop()
            elif action == 'clear':
                self.execute_clear_plan(state, plan)
            elif action == 'anchor_clear':
                self.stats['anchor_optical_probes'] += 1
                self.clear(state.channel, destination)
            elif action == 'v_probe':
                mirror, certificate = plan
                self.probe_certificates.append(certificate)
                response = self.measure(state.channel, destination, active=True)
                self.active_counts[state.channel] += 1
                self.stats['v_probes'] += 1
                if response['result'] == 'no_signal':
                    self.pending_probes[state.channel] = mirror
            elif action == 'v_mirror':
                response = self.measure(state.channel, destination, active=True)
                self.active_counts[state.channel] += 1
                self.stats['v_mirrors'] += 1
                self.pending_probes.pop(state.channel)
                if response['result'] == 'no_signal':
                    raise ValueError('Both guaranteed V-probes failed')
            elif action == 'probe':
                self.measure(state.channel, destination, active=True)
                self.active_counts[state.channel] += 1
            else:
                raise ValueError('Unexpected chase action')
            self.reuse_directions()
        raise ValueError('Chase localization exhausted')

    def run_chase(self):
        self.common_clear(self.position, all_unresolved=True)
        self.scan_unknown(self.position, force=True)
        initial_known = self.known_count()
        perimeter = self.continuous_mode == 'perimeter_chase' and initial_known <= 2
        perimeter_decided = False
        for _iteration in range(100):
            self.check_discovery()
            unresolved = [state for state in self.states.values() if state.status in ('DETECTED', 'NEAR')]
            if unresolved:
                self.stats['joint_targets'] += 1
                self.resolve_with_v(self.select_unresolved(unresolved))
                continue
            if self.discovery_done:
                return self.result()
            optical = self.optical_components()
            if optical:
                center = min(optical, key=lambda point: np.linalg.norm(point - self.position))
                self.common_clear(center)
                self.stats['optical_completion_stops'] += 1
                continue
            self.patrol.reduce([])
            if not self.remaining_stations:
                self.fallback_discovery()
                continue
            available = self.remaining_stations
            if perimeter and self.problem == 'q4' and any(index >= 9 for index in available):
                available = [index for index in available if index >= 9]
            positions = self.route[available]
            route = open_route(positions, self.position)
            selected = available[int(np.argmin(np.linalg.norm(positions - route[0], axis=1)))]
            destination = self.route[selected].copy()
            self.scan_unknown(destination, force=True)
            self.stats['visited_stations'] += 1
            self.remaining_stations.remove(selected)
            self.reuse_directions()
            if perimeter and self.problem == 'q3' and not perimeter_decided and not self.discovery_done:
                bearings = [bearing for state in self.states.values() for point, bearing in state.positives
                            if np.array_equal(point, destination)]
                if bearings:
                    ordered = np.sort(np.mod(bearings, 360))
                    span = 360 - float(np.max(np.diff(np.r_[ordered, ordered[0] + 360])))
                else:
                    span = 360.0
                compact = len(bearings) >= 3 and span <= 20
                if not compact:
                    for index in self.remaining_stations:
                        self.route[index] *= 1500.0 / np.linalg.norm(self.route[index])
                    self.patrol = CertifiedPatrol(self)
                    self.stats['peripheral_patrol'] = True
                perimeter_decided = True
        raise ValueError('Chase search exhausted')
