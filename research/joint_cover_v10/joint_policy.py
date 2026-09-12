from __future__ import annotations

import numpy as np
from joint_planner import JointCoverPlanner
from plan_geometry import fallback_plan, small_clear_plan
from policy_v5 import ContinuousPolicy


class JointCoverPolicy(ContinuousPolicy):
    def __init__(self, port, problem, stations, variant, network, mode='joint_cover'):
        super().__init__(port, problem, stations, variant, network,
                         mode='certified_fixed' if problem == 'q4' else 'certified_center')
        self.joint_planner = JointCoverPlanner(self)
        self.stats.update({'joint_cover_replans': 0, 'joint_cover_changed': 0, 'joint_single_clears': 0})

    def reuse_stop(self):
        self.reuse_directions()
        self.check_discovery()

    def next_omni_action(self, state):
        if state.status == 'NEAR':
            return 'near', state.near_position, None
        plan = small_clear_plan(state.region, self.position, 4)
        attempts = self.active_counts[state.channel]
        if plan is not None and (plan.certificate_kind == 'mec' or len(plan.centers) <= 2 or attempts >= 3):
            return 'clear', plan.centers[0], plan
        if attempts < 8 and state.consecutive_no_signal < 3:
            probe = self.choose_probe(state)
            if probe is not None:
                return 'probe', probe, None
        plan = plan or fallback_plan(state.region, state.positives[0][1], self.position)
        return 'clear', plan.centers[0], plan

    def execute_action(self, action):
        kind, channel, destination, extra = action
        state = self.states[channel]
        if state.status == 'CLEARED':
            return
        if kind == 'near':
            self.clear_near(state)
        elif kind == 'clear':
            if not extra.certified:
                raise ValueError('Clearance proposal lacks a full feasible-region certificate')
            self.certificates.append((state.region, extra.centers, extra.certificate_kind))
            if self.clear(channel, destination):
                self.stats['joint_single_clears'] += 1
        elif kind == 'probe':
            self.measure(channel, destination, active=True)
            self.active_counts[channel] += 1
        elif kind == 'anchor_clear':
            self.stats['anchor_optical_probes'] += 1
            self.clear(channel, destination)
        elif kind == 'v_probe':
            mirror, certificate = extra
            self.probe_certificates.append(certificate)
            response = self.measure(channel, destination, active=True)
            self.active_counts[channel] += 1
            self.stats['v_probes'] += 1
            if response['result'] == 'no_signal':
                self.pending_probes[channel] = mirror
            else:
                self.pending_probes.pop(channel, None)
        elif kind == 'v_mirror':
            response = self.measure(channel, destination, active=True)
            self.active_counts[channel] += 1
            self.stats['v_mirrors'] += 1
            self.pending_probes.pop(channel, None)
            if response['result'] == 'no_signal':
                raise ValueError('Both certified V endpoints were unavailable')
        else:
            raise ValueError('Unknown joint-cover action')
        self.reuse_stop()

    def run(self):
        self.common_clear(self.position, all_unresolved=True)
        self.scan_unknown(self.position, force=True)
        for _iteration in range(300):
            self.check_discovery()
            unresolved = [state for state in self.states.values() if state.status in ('DETECTED', 'NEAR')]
            if self.discovery_done and not unresolved:
                return self.result()
            actions = []
            for state in unresolved:
                kind, destination, extra = self.next_target_action(state) if self.problem == 'q4' else self.next_omni_action(state)
                actions.append((kind, state.channel, destination, extra))
            plan = self.joint_planner.choose(actions)
            if not len(plan.itinerary):
                raise ValueError('Joint-cover planner produced no executable continuation')
            destination = plan.itinerary[0]
            if any(np.array_equal(destination, point) for point in plan.scans):
                self.scan_unknown(destination, force=True)
                self.stats['visited_stations'] += 1
                self.reuse_directions()
            matching = [action for action in actions if np.array_equal(action[2], destination)]
            if matching:
                self.execute_action(matching[0])
        raise ValueError('Joint-cover action budget exhausted')

    def finish_discovery(self, reason):
        if reason == 'full_route':
            if not self.coverage.empty:
                raise ValueError('Joint-cover termination requires actual coverage')
            reason = 'coverage'
        return super().finish_discovery(reason)
