from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research' / 'offline_validation'))

from geometry import minimum_circle, route_length
from plan_geometry import ClearPlan, hull_vertices, small_clear_plan
from plan_policy import PlanPolicy

MODES = ('baseline', 'prune', 'joint', 'combined')
EXTRA_DETOUR_SECONDS = 120.0
EXTRA_CLEARS_PER_STOP = 4


def continuation_cost(centers, start, next_station):
    elapsed = 0.0
    previous = np.asarray(start)
    worst = 0.0
    for index, center in enumerate(centers):
        elapsed += float(np.linalg.norm(center - previous)) / 5.0
        continuation = 0.0 if next_station is None else (
            float(np.linalg.norm(center - next_station) - np.linalg.norm(start - next_station)) / 5.0
        )
        worst = max(worst, elapsed + 3 * index + 5 + continuation)
        previous = center
    return worst


def continuation_plan(plan, start, next_station):
    if not plan.certified or not 1 <= len(plan.centers) <= 4:
        raise ValueError('Only certified one-to-four-center plans may be interleaved')
    orders = (plan.centers[list(order)] for order in itertools.permutations(range(len(plan.centers))))
    centers = min(orders, key=lambda ordered: continuation_cost(ordered, start, next_station))
    worst = route_length(centers, start) / 5 + 3 * (len(centers) - 1) + 5
    return ClearPlan(centers, worst, True, plan.certificate_kind)


class CandidatePolicy(PlanPolicy):
    def __init__(self, port, problem, stations, variant, network, mode='combined'):
        if mode not in MODES:
            raise ValueError('Unknown candidate mode')
        super().__init__(port, problem, stations, variant, network)
        self.joint_enabled = mode in ('joint', 'combined')
        self.prune_enabled = mode in ('prune', 'combined')
        self.stats.update({'pruned_centers': 0, 'extended_opportunistic_clears': 0, 'cleanup_opportunity_stops': 0})

    def execute_clear_plan(self, state, plan, opportunistic=False):
        if not self.prune_enabled or plan.certificate_kind != 'rectangle_partition_circumradius_bound':
            return super().execute_clear_plan(state, plan, opportunistic)
        if not plan.certified:
            raise ValueError('Uncertified fallback')
        self.certificates.append((state.region, plan.centers, plan.certificate_kind))
        remaining = list(range(len(plan.centers)))
        while remaining:
            viable = [index for index in remaining
                      if state.region.distance(Point(plan.centers[index])) <= 20.000001]
            self.stats['pruned_centers'] += len(remaining) - len(viable)
            if not viable:
                break
            selected = min(viable, key=lambda index: float(np.linalg.norm(plan.centers[index] - self.position)))
            remaining = [index for index in viable if index != selected]
            if self.clear(state.channel, plan.centers[selected]):
                self.stats['fallback_clears'] += 1
                self.stats['opportunistic_clears'] += int(opportunistic)
                return
        raise ValueError('Certified fallback exhausted after safe geometric pruning')

    def opportunistic_actions(self, next_station):
        super().opportunistic_actions(next_station)
        if not self.joint_enabled:
            return
        for _ in range(EXTRA_CLEARS_PER_STOP):
            candidates = []
            for state in self.states.values():
                if state.status == 'NEAR':
                    plan = None
                    centers = np.asarray([state.near_position])
                elif state.status == 'DETECTED':
                    plan = small_clear_plan(state.region, self.position, 4)
                    if plan is None:
                        continue
                    plan = continuation_plan(plan, self.position, next_station)
                    centers = plan.centers
                else:
                    continue
                extra_seconds = continuation_cost(centers, self.position, next_station)
                if extra_seconds <= EXTRA_DETOUR_SECONDS:
                    candidates.append((extra_seconds, state.channel, plan))
            if not candidates:
                break
            _cost, channel, plan = min(candidates, key=lambda candidate: (candidate[0], candidate[1]))
            state = self.states[channel]
            if plan is None:
                self.clear_near(state, opportunistic=True)
            else:
                self.execute_clear_plan(state, plan, opportunistic=True)
            self.stats['extended_opportunistic_clears'] += 1

    def localize(self, state):
        super().localize(state)
        if not self.joint_enabled:
            return
        unresolved = [other for other in self.states.values() if other.status in ('DETECTED', 'NEAR')]
        if not unresolved:
            return
        next_state = self.select_unresolved(unresolved)
        next_station = (next_state.near_position if next_state.status == 'NEAR'
                        else minimum_circle(hull_vertices(next_state.region))[0])
        self.stats['cleanup_opportunity_stops'] += 1
        self.opportunistic_actions(next_station)
