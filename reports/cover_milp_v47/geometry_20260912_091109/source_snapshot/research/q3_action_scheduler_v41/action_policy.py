from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from geometry import open_route
from plan_geometry import disk_polygon, fallback_plan, small_clear_plan
from scan_policy import ScanEconomyPolicy
from shapely.geometry import Point


MODES = ('identity', 'atomic_action', 'stepwise_action')


@dataclass
class LocalAction:
    kind: str
    position: np.ndarray
    plan: object = None
    fallback: bool = False


class ActionSchedulerPolicy(ScanEconomyPolicy):
    def __init__(self, *args, mode='stepwise_action'):
        if mode not in MODES:
            raise ValueError('Unknown Q3 action-scheduler mode')
        super().__init__(*args, mode='station_only')
        self.action_mode = mode
        self.probe_steps = {channel: 0 for channel in self.states}
        self.action_events = []
        self.stats.update({'scheduler_probe_steps': 0, 'scheduler_source_selections': 0})
        if self.options.get('target_radius') != 2000:
            raise ValueError('The frozen Q3 admission threshold changed')

    def plan_action(self, state, plan, fallback=False):
        viable = [index for index, point in enumerate(plan.centers) if state.region.distance(Point(point)) <= 20.000001]
        if not viable:
            raise ValueError('Certified clear plan has no viable first action')
        selected = min(viable, key=lambda index: float(np.linalg.norm(plan.centers[index] - self.position)))
        return LocalAction('clear', np.asarray(plan.centers[selected]).copy(), plan, fallback)

    def next_local_action(self, state, iteration):
        if state.status == 'NEAR':
            return LocalAction('near', state.near_position.copy())
        plan = small_clear_plan(state.region, self.position, 4)
        if plan is not None and (plan.certificate_kind == 'mec' or len(plan.centers) <= 2 or iteration >= 3):
            return self.plan_action(state, plan)
        probe = self.choose_probe(state) if iteration < 8 and state.consecutive_no_signal < 3 else None
        if probe is None:
            plan = plan or fallback_plan(state.region, state.positives[0][1], self.position)
            return self.plan_action(state, plan, fallback=True)
        return LocalAction('probe', np.asarray(probe).copy())

    def execute_one(self, state, action):
        self.active_target = state.channel
        try:
            if action.kind == 'near':
                self.clear_near(state)
                self.reuse_stop()
            elif action.kind == 'clear':
                self.stats['certified_fallbacks'] += int(action.fallback)
                self.execute_clear_plan(state, action.plan)
            elif action.kind == 'probe':
                response = self.measure(state.channel, action.position, active=True)
                self.probe_history[state.channel].append(action.position.copy())
                self.probe_steps[state.channel] += 1
                self.stats['scheduler_probe_steps'] += 1
                self.stats['close_probes'] += 1
                self.stats['close_no_signal'] += int(response['result'] == 'no_signal')
                self.reuse_stop()
            else:
                raise ValueError('Unknown local action')
        finally:
            self.active_target = None

    def execute_locked(self, state):
        for _iteration in range(9):
            if state.status == 'CLEARED':
                return
            action = self.next_local_action(state, self.probe_steps[state.channel])
            self.execute_one(state, action)
        if state.status != 'CLEARED':
            raise ValueError('Unfinished locked localization')

    def run(self):
        remaining = list(range(len(self.route)))
        self.scan_unknown(self.position, force=True)
        remaining.remove(0)
        for _iteration in range(400):
            unresolved = [state for state in self.states.values() if state.status in ('DETECTED', 'NEAR')]
            if self.discovery_done and not unresolved:
                return self.result()
            if not remaining and not self.discovery_done:
                self.finish_discovery('full_route')
            if self.options.get('adaptive') and not self.discovery_done:
                irrelevant = [index for index in remaining if self.coverage.region.intersection(disk_polygon(self.route[index], 999.9)).is_empty]
                remaining = [index for index in remaining if index not in irrelevant]
                self.stats['coverage_skips'] += len(irrelevant)
                if not remaining:
                    if not self.coverage.empty:
                        raise ValueError('Pruned source discovery without complete coverage')
                    self.finish_discovery('coverage')
            stops = []
            for state in unresolved:
                center, radius = (state.near_position, 0) if state.status == 'NEAR' else self.center_radius(state)
                if radius <= self.options.get('target_radius', 130) or not remaining or self.discovery_done:
                    action = None
                    if self.action_mode != 'identity':
                        attempts = self.probe_steps[state.channel] if self.action_mode == 'stepwise_action' else 0
                        action = self.next_local_action(state, attempts)
                    stops.append(('target', state.channel, center if action is None else action.position, action))
            if not self.discovery_done:
                stops.extend(('station', index, self.route[index], None) for index in remaining)
            if not stops:
                return self.result()
            positions = np.asarray([stop[2] for stop in stops])
            itinerary = open_route(positions, self.position)
            selected = int(np.argmin(np.linalg.norm(positions - itinerary[0], axis=1)))
            kind, identifier, destination, action = stops[selected]
            if kind == 'station':
                self.scan_unknown(destination, force=True)
                self.stats['visited_stations'] += 1
                self.reuse_directions()
                remaining.remove(identifier)
                continue
            self.stats['joint_targets'] += 1
            self.stats['scheduler_source_selections'] += 1
            event = {'channel': identifier, 'mode': self.action_mode, 'start_seconds': self.virtual_seconds,
                     'predicted_position': None if action is None else action.position.tolist(),
                     'predicted_kind': None if action is None else action.kind,
                     'active_steps_before': self.probe_steps[identifier]}
            self.action_events.append(event)
            state = self.states[identifier]
            if self.action_mode == 'identity':
                self.execute_locked(state)
            elif self.action_mode == 'atomic_action':
                self.localize(state)
            else:
                self.execute_one(state, action)
            event['end_seconds'] = self.virtual_seconds
            event['status_after'] = state.status
            event['active_steps_after'] = self.probe_steps[identifier]
        raise ValueError('Q3 action scheduler exhausted its action limit')
