from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from geometry import minimum_circle, open_route, ring
from plan_geometry import (
    CLEAR_SUPPORT_RADIUS,
    clear_failure_update,
    disk_polygon,
    fallback_plan,
    hull_vertices,
    initial_outer_region,
    minimax_measurement,
    positive_update,
    reception_proxy,
    sampled_clear_cost,
    small_clear_plan,
    source_proxy_hypotheses,
)

VARIANTS = {
    'A_cover': 0,
    'B_minimax': 1,
    'C_multi_clear': 2,
    'D_direction_gate': 3,
    'E_joint': 4,
    'F_route': 5,
}


@dataclass
class ChannelBelief:
    channel: int
    region: object
    status: str = 'UNKNOWN'
    positives: list = field(default_factory=list)
    negatives: list = field(default_factory=list)
    measured_positions: list = field(default_factory=list)
    near_position: np.ndarray | None = None
    consecutive_no_signal: int = 0
    revision: int = 0
    proxy_revision: int = -1
    proxy_bank: object = None


class PlanPolicy:
    def __init__(self, port, problem, stations, variant, network):
        self.port = port
        self.problem = problem
        self.network = network
        self.stage = VARIANTS[variant]
        self.states = {channel: ChannelBelief(channel, initial_outer_region()) for channel in range(1, 21)}
        self.position = np.zeros(2)
        self.receiver_channel = 1
        self.virtual_seconds = 0.0
        self.started = time.perf_counter()
        self.stations = np.asarray(stations)
        if network == 'dual21':
            self.route = np.vstack(([0, 0], ring(1000, 8), ring(1870, 12, 315)))
        else:
            ordered = open_route(self.stations)
            self.route = np.vstack(([0, 0], ordered[np.linalg.norm(ordered, axis=1) > 1e-8]))
        self.snapshots = []
        self.certificates = []
        self.cache = {}
        self.stats = {
            'active_measures': 0, 'active_no_signal': 0, 'opportunistic_measures': 0,
            'opportunistic_clears': 0, 'multi_circle_clears': 0, 'mec_clears': 0,
            'fallback_clears': 0, 'cost_gate_rejections': 0, 'empty_proxy_banks': 0,
            'strict_receiving_checks': 0, 'geometric_absent': 0, 'visited_stations': 0,
            'negative_position_cuts': 0, 'route_refinements': 0,
        }

    def snapshot(self, state):
        self.snapshots.append((state.channel, state.region))
        state.revision += 1

    def account(self, position, channel, operation, result):
        self.virtual_seconds += float(np.linalg.norm(position - self.position)) / 5
        self.position = np.asarray(position).copy()
        if operation == 'measure':
            self.virtual_seconds += 5 + int(channel != self.receiver_channel)
            self.receiver_channel = channel
        else:
            self.virtual_seconds += 5 if result == 'success' else 3
        if self.virtual_seconds > 100 * 3600 or time.perf_counter() - self.started > 1100:
            raise RuntimeError('Plan policy exceeded offline execution budget')

    def measure(self, channel, position, active=False, opportunistic=False):
        key = (channel, float(position[0]).hex(), float(position[1]).hex())
        if key in self.cache:
            return self.cache[key]
        state = self.states[channel]
        response = self.port.measure(position, channel)
        result = response['result']
        self.account(position, channel, 'measure', result)
        self.cache[key] = response
        state.measured_positions.append(np.asarray(position).copy())
        if active:
            self.stats['active_measures'] += 1
            self.stats['active_no_signal'] += int(result == 'no_signal')
        if opportunistic:
            self.stats['opportunistic_measures'] += 1
        if result == 'direction':
            state.region = positive_update(state.region, position, response['bearing_deg'])
            state.positives.append((np.asarray(position).copy(), response['bearing_deg']))
            state.status = 'DETECTED'
            state.consecutive_no_signal = 0
            self.snapshot(state)
        elif result == 'near':
            state.status = 'NEAR'
            state.near_position = np.asarray(position).copy()
            state.consecutive_no_signal = 0
        elif result == 'no_signal':
            state.negatives.append(np.asarray(position).copy())
            state.consecutive_no_signal += 1
            state.revision += 1
            if self.problem == 'q3':
                state.region = state.region.difference(disk_polygon(position, 999.99))
                self.stats['negative_position_cuts'] += 1
                if state.region.is_empty:
                    if state.status != 'UNKNOWN':
                        raise ValueError('Known source contradicted by omnidirectional negative measurement')
                    state.status = 'ABSENT'
                    self.stats['geometric_absent'] += 1
                else:
                    self.snapshot(state)
        else:
            raise ValueError(f'Unknown measurement result: {result}')
        return response

    def clear(self, channel, position):
        state = self.states[channel]
        response = self.port.clear(position, channel)
        result = response['result']
        self.account(position, channel, 'clear', result)
        if result == 'success':
            state.status = 'CLEARED'
        elif result == 'no_target_in_range':
            state.region = clear_failure_update(state.region, position)
            self.snapshot(state)
        else:
            raise ValueError(f'Unknown clearance result: {result}')
        return result == 'success'

    def hypotheses(self, state):
        if state.proxy_revision != state.revision:
            state.proxy_bank = source_proxy_hypotheses(
                state.region, state.positives, state.negatives, self.problem == 'q4',
            )
            state.proxy_revision = state.revision
            self.stats['empty_proxy_banks'] += int(state.proxy_bank is None)
        return state.proxy_bank

    def execute_clear_plan(self, state, plan, opportunistic=False):
        assert plan.certified
        self.certificates.append((state.region, plan.centers, plan.certificate_kind))
        for center in plan.centers:
            if self.clear(state.channel, center):
                if plan.certificate_kind == 'mec':
                    self.stats['mec_clears'] += 1
                elif plan.certificate_kind == 'rectangle_partition_circumradius_bound':
                    self.stats['fallback_clears'] += 1
                else:
                    self.stats['multi_circle_clears'] += 1
                self.stats['opportunistic_clears'] += int(opportunistic)
                return
        raise ValueError('A certified clearance cover was exhausted without success')

    def clear_near(self, state, opportunistic=False):
        if not self.clear(state.channel, state.near_position):
            raise ValueError('Near observation did not clear at the same position')
        self.stats['mec_clears'] += 1
        self.stats['opportunistic_clears'] += int(opportunistic)

    def localize(self, state):
        if state.status == 'NEAR':
            self.clear_near(state)
            return
        extra_measures = 0
        while state.status == 'DETECTED':
            small = small_clear_plan(state.region, self.position, 4 if self.stage >= 2 else 1)
            if small is not None and small.certificate_kind == 'mec':
                self.execute_clear_plan(state, small)
                return
            fallback = fallback_plan(state.region, state.positives[0][1], self.position)
            clear_option = small if small is not None else fallback
            if self.stage == 0 or extra_measures >= 4:
                self.execute_clear_plan(state, clear_option)
                return
            if self.stage >= 3 and state.consecutive_no_signal >= 2:
                self.stats['cost_gate_rejections'] += 1
                self.execute_clear_plan(state, clear_option)
                return
            measurement = minimax_measurement(
                state.region, state.positives[-1][0], self.position, state.measured_positions,
            )
            if measurement is None:
                self.execute_clear_plan(state, clear_option)
                return
            assert measurement.max_receiving_distance <= 999.900001
            self.stats['strict_receiving_checks'] += 1
            predicted = measurement.immediate_seconds + measurement.positive_proxy_seconds
            prefer_clear = small is not None and small.worst_seconds <= predicted
            if self.stage >= 3:
                bank = self.hypotheses(state)
                if bank is None:
                    prefer_clear = True
                else:
                    positions, _headings, _radii, weights = bank
                    reception = reception_proxy(bank, measurement.position)
                    direct_cost = sampled_clear_cost(clear_option, positions, self.position, weights)
                    no_signal_cost = sampled_clear_cost(clear_option, positions, measurement.position, weights)
                    predicted = (
                        measurement.immediate_seconds + reception * measurement.positive_proxy_seconds
                        + (1 - reception) * no_signal_cost
                    )
                    prefer_clear = prefer_clear or predicted >= direct_cost - max(3.0, 0.03 * direct_cost)
            if prefer_clear:
                self.stats['cost_gate_rejections'] += 1
                self.execute_clear_plan(state, clear_option)
                return
            response = self.measure(state.channel, measurement.position, active=True)
            extra_measures += 1
            if self.problem == 'q3' and response['result'] == 'no_signal':
                raise ValueError('Guaranteed omnidirectional reception failed')
            if state.status == 'NEAR':
                self.clear_near(state)
                return

    def opportunistic_actions(self, next_station):
        candidates = []
        for state in self.states.values():
            if state.status != 'DETECTED' or not state.positives:
                continue
            if any(np.linalg.norm(self.position - previous) < 1e-6 for previous in state.measured_positions):
                continue
            center, radius = minimum_circle(hull_vertices(state.region))
            if radius <= CLEAR_SUPPORT_RADIUS:
                continue
            first = center - state.positives[-1][0]
            second = center - self.position
            product = np.linalg.norm(first) * np.linalg.norm(second)
            sine = abs(first[0] * second[1] - first[1] * second[0]) / max(product, 1e-9)
            if sine < 0.5:
                continue
            probability = reception_proxy(self.hypotheses(state), self.position)
            if probability >= 0.3:
                candidates.append((probability * sine * radius, state.channel))
        for _score, channel in sorted(candidates, reverse=True)[:4]:
            self.measure(channel, self.position, opportunistic=True)
        for _ in range(2):
            clear_candidates = []
            for state in self.states.values():
                if state.status == 'NEAR':
                    destination = state.near_position
                    plan = None
                elif state.status == 'DETECTED':
                    plan = small_clear_plan(state.region, self.position, 1)
                    if plan is None:
                        continue
                    destination = plan.centers[0]
                else:
                    continue
                detour = np.linalg.norm(self.position - destination)
                if next_station is not None:
                    detour += np.linalg.norm(destination - next_station) - np.linalg.norm(self.position - next_station)
                if detour <= 200.0:
                    clear_candidates.append((detour, state.channel, plan))
            if not clear_candidates:
                break
            _detour, channel, plan = min(clear_candidates, key=lambda candidate: candidate[0])
            state = self.states[channel]
            if plan is None:
                self.clear_near(state, opportunistic=True)
            else:
                self.execute_clear_plan(state, plan, opportunistic=True)

    def known_count(self):
        return sum(state.status in ('DETECTED', 'NEAR', 'CLEARED') for state in self.states.values())

    def select_unresolved(self, unresolved):
        destinations = []
        for state in unresolved:
            destinations.append(
                state.near_position if state.status == 'NEAR' else minimum_circle(hull_vertices(state.region))[0]
            )
        destinations = np.asarray(destinations)
        if self.stage >= 5 and len(unresolved) > 2:
            route = open_route(destinations, self.position)
            index = int(np.argmin(np.linalg.norm(destinations - route[0], axis=1)))
            self.stats['route_refinements'] += 1
        else:
            index = int(np.argmin(np.linalg.norm(destinations - self.position, axis=1)))
        return unresolved[index]

    def run(self):
        for station_index, station in enumerate(self.route):
            unknown = [state.channel for state in self.states.values() if state.status == 'UNKNOWN']
            for channel in sorted(unknown, reverse=bool(station_index % 2)):
                self.measure(channel, station)
                if self.known_count() == 16:
                    break
            self.stats['visited_stations'] += 1
            if self.stage >= 4:
                next_station = self.route[station_index + 1] if station_index + 1 < len(self.route) else None
                self.opportunistic_actions(next_station)
            if self.known_count() == 16 or not any(state.status == 'UNKNOWN' for state in self.states.values()):
                break
        assert self.stats['visited_stations'] == len(self.route) or self.known_count() == 16 or not any(
            state.status == 'UNKNOWN' for state in self.states.values()
        )
        for state in self.states.values():
            if state.status == 'UNKNOWN':
                state.status = 'ABSENT'
        while True:
            unresolved = [state for state in self.states.values() if state.status in ('DETECTED', 'NEAR')]
            if not unresolved:
                break
            self.localize(self.select_unresolved(unresolved))
        assert all(state.status in ('CLEARED', 'ABSENT') for state in self.states.values())
        return {
            'stats': self.stats, 'virtual_seconds': self.virtual_seconds,
            'declared_absent': [state.channel for state in self.states.values() if state.status == 'ABSENT'],
            'cleared_channels': [state.channel for state in self.states.values() if state.status == 'CLEARED'],
            'snapshots': self.snapshots, 'certificates': self.certificates,
        }
