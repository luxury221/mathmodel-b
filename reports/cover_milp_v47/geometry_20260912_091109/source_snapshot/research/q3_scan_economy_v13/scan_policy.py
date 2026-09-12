from __future__ import annotations

import copy

import numpy as np
from radial_policy import RadialPolicy


MODES = ('station_only', 'certified_reuse')


class ScanEconomyPolicy(RadialPolicy):
    def __init__(self, *args, mode='certified_reuse'):
        if mode not in MODES:
            raise ValueError('Unknown Q3 scan-economy mode')
        self.scan_mode = mode
        self.skipped_positions = set()
        self.replacement_proofs = []
        super().__init__(*args, mode='pilot1600')
        self.stats.update({'suppressed_opportunistic_scans': 0, 'certified_scan_substitutions': 0,
                           'skipped_certified_stations': 0})

    def planned_positions(self):
        return [position for position in self.route
                if tuple(position) not in self.skipped_positions
                and not any(np.linalg.norm(position - observed) < 1e-8 for observed in self.scan_positions)]

    def replacement(self, position):
        if self.pilot_pending or not self.origin_assessed or self.discovery_done or not self.unknown_channels():
            return None
        if any(np.linalg.norm(position - observed) < 25 for observed in self.scan_positions):
            return None
        planned = self.planned_positions()
        for removed in sorted(planned, key=lambda point: float(np.linalg.norm(point - position)), reverse=True):
            trial = copy.deepcopy(self.coverage)
            trial.observe_absence(position)
            for point in planned:
                if not np.array_equal(point, removed):
                    trial.observe_absence(point)
            if trial.empty:
                return removed.copy(), [point.copy() for point in planned if not np.array_equal(point, removed)]
        return None

    def scan_unknown(self, position, force=False):
        position = np.asarray(position, dtype=float)
        if force:
            if tuple(position) in self.skipped_positions:
                self.stats['skipped_certified_stations'] += 1
                return None
            return super().scan_unknown(position, force=True)
        if self.scan_mode == 'station_only':
            self.stats['suppressed_opportunistic_scans'] += 1
            return None
        proposal = self.replacement(position)
        if proposal is None:
            self.stats['suppressed_opportunistic_scans'] += 1
            return None
        removed, future = proposal
        response = super().scan_unknown(position, force=True)
        observed = any(np.array_equal(position, observed) for observed in self.coverage.observations)
        if not observed and not self.discovery_done:
            raise ValueError('Replacement scan has no actual observation')
        self.skipped_positions.add(tuple(removed))
        self.replacement_proofs.append({'replacement_position': position.tolist(), 'removed_position': removed.tolist(),
                                        'future_positions': [point.tolist() for point in future],
                                        'actual_observation_count': len(self.coverage.observations),
                                        'actual_clock': self.virtual_seconds})
        self.stats['certified_scan_substitutions'] += 1
        return response
