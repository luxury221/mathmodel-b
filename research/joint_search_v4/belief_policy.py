from __future__ import annotations

from candidate_v2 import Q4OptimizationPolicy
from coverage import DirectionalBelief
import numpy as np


class BeliefPreviousPolicy(Q4OptimizationPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, mode='probe_scan', **kwargs)
        self.direction_beliefs = {}
        self.stats['directional_belief_updates'] = 0

    def measure(self, channel, position, active=False, opportunistic=False):
        response = super().measure(channel, position, active, opportunistic)
        state = self.states[channel]
        if self.problem == 'q4' and state.status == 'DETECTED':
            if channel not in self.direction_beliefs:
                self.direction_beliefs[channel] = DirectionalBelief()
            state.region = self.direction_beliefs[channel].update(state.region, state.positives, state.negatives)
            self.snapshot(state)
            self.stats['directional_belief_updates'] += 1
        return response


class SweepPreviousPolicy(BeliefPreviousPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clear_positions = []
        self.sweep_positions = []
        self.sweeping = False
        self.stats.update({'sweep_attempts': 0, 'sweep_hits': 0, 'sweep_discoveries': 0})

    def clear(self, channel, position):
        was_near = self.states[channel].status == 'NEAR'
        success = super().clear(channel, position)
        if not success:
            return False
        self.clear_positions.append(np.asarray(position).copy())
        clustered = sum(np.linalg.norm(previous - position) < 50 for previous in self.clear_positions) >= 2
        eligible = (not self.sweeping and len(self.sweep_positions) < 2 and (was_near or clustered)
                    and all(np.linalg.norm(previous - position) >= 200 for previous in self.sweep_positions))
        if eligible:
            self.sweep_positions.append(np.asarray(position).copy())
            self.sweeping = True
            for other in self.states.values():
                if other.status in ('ABSENT', 'CLEARED'):
                    continue
                was_unknown = other.status == 'UNKNOWN'
                self.stats['sweep_attempts'] += 1
                hit = self.clear(other.channel, position)
                self.stats['sweep_hits'] += int(hit)
                self.stats['sweep_discoveries'] += int(hit and was_unknown)
            self.sweeping = False
        return True
