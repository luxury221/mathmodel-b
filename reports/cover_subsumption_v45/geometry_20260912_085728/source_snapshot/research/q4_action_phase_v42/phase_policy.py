from __future__ import annotations

import copy
import math

import numpy as np
from geometry import open_route, route_length
from shaped_policy import ShapedProbePolicy
from shapely.geometry import Point


MODES = ('identity', 'action_phase')
PHASES = tuple(range(0, 90, 5))


def rotate_stations(stations, phase):
    proposal = np.asarray(stations, dtype=float).copy()
    if phase == 0:
        return proposal
    angle = math.radians(phase)
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    proposal[1:] = proposal[1:] @ rotation.T
    return proposal


def first_clear_position(state, plan, position):
    viable = [point for point in plan.centers if state.region.distance(Point(point)) <= 20.000001]
    if not viable:
        raise ValueError('Certified clear plan has no viable first physical action')
    return np.asarray(min(viable, key=lambda point: float(np.linalg.norm(point - position)))).copy()


class ActionPhasePolicy(ShapedProbePolicy):
    def __init__(self, *args, mode='action_phase'):
        if mode not in MODES:
            raise ValueError('Unregistered Q4 phase mode')
        self.phase_mode = mode
        self.phase_decided = False
        self.phase_events = []
        super().__init__(*args, mode='shaped_cost')
        self.stats.update({'phase_degrees': 0, 'phase_valid_candidates': 0, 'phase_actual_rotations': 0})

    def action_targets(self):
        targets = []
        for state in self.states.values():
            if state.status not in ('DETECTED', 'NEAR'):
                continue
            kind, destination, extra = self.next_target_action(state)
            if kind == 'clear':
                destination = first_clear_position(state, extra, self.position)
            targets.append({'channel': state.channel, 'kind': kind, 'position': np.asarray(destination).tolist(),
                            'revision': state.revision, 'status': state.status})
        return targets

    def scan_unknown(self, position, force=False):
        result = super().scan_unknown(position, force)
        if self.phase_decided:
            return result
        self.phase_decided = True
        if self.phase_mode == 'identity' or self.discovery_done:
            return result
        if not np.array_equal(np.asarray(position), np.zeros(2)) or len(self.coverage.observations) != 1:
            raise ValueError('Phase selection requires the first real origin scan')
        if self.remaining_stations != list(range(1, len(self.route))):
            raise ValueError('Cannot rotate after off-origin station commitments')
        before = self.coverage.region.wkb
        targets = self.action_targets()
        if not targets:
            return result
        old_route = self.route.copy()
        candidates = []
        for phase in PHASES:
            proposal = rotate_stations(old_route, phase)
            trial = copy.deepcopy(self.coverage)
            for station in proposal[1:]:
                trial.observe_absence(station)
            cost = None
            if trial.empty:
                points = np.vstack((proposal[1:], [target['position'] for target in targets]))
                cost = route_length(open_route(points, self.position), self.position)
            candidates.append({'phase_degrees': phase, 'complete': trial.empty,
                               'remaining_area': trial.area, 'proxy_meters': cost})
        if not candidates[0]['complete']:
            raise ValueError('Original network has no complete certificate with actual evidence')
        chosen = candidates[0]
        for candidate in candidates[1:]:
            if candidate['complete'] and candidate['proxy_meters'] < chosen['proxy_meters'] - 1e-6:
                chosen = candidate
        if self.coverage.region.wkb != before:
            raise ValueError('Planning changed real coverage evidence')
        proposal = rotate_stations(old_route, chosen['phase_degrees'])
        self.phase_events.append({'phase_degrees': chosen['phase_degrees'], 'original_route': old_route.tolist(),
                                  'selected_route': proposal.tolist(), 'targets': targets, 'candidates': candidates,
                                  'original_proxy_meters': candidates[0]['proxy_meters'],
                                  'selected_proxy_meters': chosen['proxy_meters'],
                                  'before_radio': [point.tolist() for point in self.coverage.observations],
                                  'before_optical': [point.tolist() for point in self.coverage.clear_observations],
                                  'unknown_channels': self.unknown_channels(), 'actual_position': self.position.tolist(),
                                  'actual_virtual_seconds': self.virtual_seconds})
        self.stats['phase_degrees'] = chosen['phase_degrees']
        self.stats['phase_valid_candidates'] = sum(candidate['complete'] for candidate in candidates)
        if chosen['phase_degrees']:
            self.route = proposal
            self.patrol.matrix = np.array([self.search_planner.visible(point) for point in self.route])
            self.patrol.revision = None
            self.stats['phase_actual_rotations'] = 1
        return result
