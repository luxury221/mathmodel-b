from __future__ import annotations

import copy

import numpy as np
from cover_planner import AdaptiveCoverPlanner
from geometry import open_route
from scan_policy import ScanEconomyPolicy


MODES = ('source_first', 'source_repack')


class AdaptiveSourceCoverPolicy(ScanEconomyPolicy):
    def __init__(self, *args, mode='source_repack'):
        if mode not in MODES:
            raise ValueError('Unknown adaptive source-cover mode')
        super().__init__(*args, mode='station_only')
        self.adaptive_mode = mode
        self.pending_scans = []
        self.cover_planner = AdaptiveCoverPlanner(self)
        self.cover_events = []
        self.stats.update({'adaptive_extra_scans': 0, 'adaptive_relocations': 0,
                           'adaptive_prunes': 0, 'adaptive_certificates': 0,
                           'adaptive_executed_nonbaseline_scans': 0})

    def refine_pending(self):
        if self.discovery_done:
            self.pending_scans = []
            return
        self.pending_scans = self.cover_planner.refine(self.coverage, self.pending_scans,
                                                      self.adaptive_mode == 'source_repack' and not self.pilot_pending)

    def try_current_scan(self):
        if self.discovery_done or not self.unknown_channels() or self.pilot_pending:
            return False
        if any(np.linalg.norm(self.position - point) < 25 for point in self.coverage.observations):
            return False
        trial = copy.deepcopy(self.coverage)
        trial.observe_absence(self.position)
        future = self.cover_planner.refine(trial, self.pending_scans, self.adaptive_mode == 'source_repack')
        current_price = self.cover_planner.price(self.pending_scans)
        proposed_price = 6 * len(self.unknown_channels()) + self.cover_planner.price(future)
        if proposed_price >= current_price - 5:
            return False
        point = self.position.copy()
        event = {'kind': 'current_scan', 'point': point.tolist(), 'clock_before': self.virtual_seconds,
                 'original_plan': [position.tolist() for position in self.pending_scans],
                 'future_plan': [position.tolist() for position in future],
                 'predicted_before': current_price, 'predicted_after': proposed_price,
                 'actual_observations_before': len(self.coverage.observations)}
        self.scan_unknown(point, force=True)
        self.stats['adaptive_extra_scans'] += 1
        self.pending_scans = [] if self.discovery_done else future
        if not self.discovery_done and not self.cover_planner.complete(self.coverage, self.pending_scans):
            raise ValueError('Actual scan did not support the committed future plan')
        event['clock_after'] = self.virtual_seconds
        event['actual_observations_after'] = len(self.coverage.observations)
        self.cover_events.append(event)
        self.reuse_directions()
        return True

    def run(self):
        self.scan_unknown(self.position, force=True)
        self.pending_scans = [point.copy() for point in self.route[1:]]
        for _iteration in range(220):
            unresolved = [state for state in self.states.values() if state.status in ('DETECTED', 'NEAR')]
            if self.discovery_done and not unresolved:
                self.stats.update({'adaptive_relocations': self.cover_planner.relocations,
                                   'adaptive_prunes': self.cover_planner.prunes,
                                   'adaptive_certificates': self.cover_planner.certificates_checked})
                return self.result()
            if unresolved:
                self.stats['joint_targets'] += 1
                self.localize(self.select_unresolved(unresolved))
                self.refine_pending()
                self.try_current_scan()
                continue
            self.refine_pending()
            if not self.pending_scans:
                if not self.coverage.empty:
                    raise ValueError('Scanning stopped with a nonempty continuous remainder')
                self.finish_discovery('coverage')
                continue
            ordered = open_route(np.asarray(self.pending_scans), self.position)
            point = ordered[0]
            was_pilot = self.pilot_pending
            self.scan_unknown(point, force=True)
            self.stats['visited_stations'] += 1
            self.stats['adaptive_executed_nonbaseline_scans'] += int(
                not any(np.linalg.norm(point - original) < 1e-5 for original in self.route))
            self.reuse_directions()
            if was_pilot and not self.pilot_pending:
                self.pending_scans = [position.copy() for position in self.route[1:]
                                      if not any(np.array_equal(position, observed) for observed in self.coverage.observations)]
            else:
                self.pending_scans = [position for position in self.pending_scans if not np.array_equal(point, position)]
            self.cover_events.append({'kind': 'planned_scan', 'point': point.tolist(), 'clock': self.virtual_seconds,
                                     'remaining_count': len(self.pending_scans),
                                     'actual_observations': len(self.coverage.observations)})
        raise ValueError('Adaptive source-cover action budget exhausted')
