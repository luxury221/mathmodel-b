from __future__ import annotations

import copy
import math

import numpy as np
from geometry import open_route, route_length
from scan_policy import ScanEconomyPolicy


MODES = ('identity', 'joint_phase')


def rotated_layout(points, angle_degrees):
    if angle_degrees == 0:
        return np.asarray(points).copy()
    angle = math.radians(angle_degrees)
    matrix = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    rotated = np.asarray(points).copy()
    rotated[1:] = rotated[1:] @ matrix.T
    return rotated


class SymmetryLayoutPolicy(ScanEconomyPolicy):
    def __init__(self, *args, mode='joint_phase'):
        if mode not in MODES:
            raise ValueError('Unknown symmetry-layout mode')
        self.symmetry_mode = mode
        self.rotation_decided = False
        self.rotation_events = []
        super().__init__(*args, mode='station_only')
        self.stats.update({'rotation_degrees': 0, 'rotation_certified_proposals': 0})

    def scan_unknown(self, position, force=False):
        response = super().scan_unknown(position, force)
        if self.rotation_decided or not self.origin_assessed:
            return response
        self.rotation_decided = True
        if self.symmetry_mode == 'identity' or self.discovery_done:
            return response
        if not np.array_equal(np.asarray(position), np.zeros(2)):
            raise ValueError('Rotation requires the actual initial origin scan')
        if len(self.coverage.observations) != 1:
            raise ValueError('Rotation must precede every off-origin scan')
        targets = [state.near_position if state.status == 'NEAR' else self.center_radius(state)[0]
                   for state in self.states.values() if state.status in ('DETECTED', 'NEAR')]
        if not targets:
            return response
        choices = []
        for phase in range(0, 60, 5):
            proposal = rotated_layout(self.route, phase)
            trial = copy.deepcopy(self.coverage)
            for station in proposal[1:]:
                trial.observe_absence(station)
            if not trial.empty:
                continue
            points = np.vstack((proposal[1:], targets))
            cost = route_length(open_route(points, self.position), self.position)
            choices.append((cost, phase, proposal))
        if not choices or choices[0][1] != 0:
            raise ValueError('Original scan network lost its coverage certificate')
        cost, phase, proposal = choices[0]
        for candidate in choices[1:]:
            if candidate[0] < cost - 1e-6:
                cost, phase, proposal = candidate
        self.rotation_events.append({'phase_degrees': phase, 'original_proxy_meters': choices[0][0],
                                     'selected_proxy_meters': cost, 'certified_candidates': len(choices),
                                     'actual_coverage_observations': len(self.coverage.observations),
                                     'actual_virtual_seconds': self.virtual_seconds,
                                     'planned_stations': proposal.tolist()})
        self.route = proposal
        self.stats['rotation_degrees'] = phase
        self.stats['rotation_certified_proposals'] = len(choices)
        return response
