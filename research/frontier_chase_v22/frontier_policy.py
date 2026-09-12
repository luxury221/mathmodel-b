from __future__ import annotations

import numpy as np
from adaptive_policy import AdaptiveSourceCoverPolicy


class FrontierChasePolicy(AdaptiveSourceCoverPolicy):
    def __init__(self, *args, mode='frontier'):
        if mode != 'frontier':
            raise ValueError('Unknown frontier-chase mode')
        super().__init__(*args, mode='source_repack')
        self.stats.update({'frontier_scans': 0, 'frontier_area_removed': 0.0})

    def try_current_scan(self):
        if self.discovery_done or not self.unknown_channels() or self.pilot_pending:
            return False
        if any(np.linalg.norm(self.position - point) < 25 for point in self.coverage.observations):
            return False
        fee = 6 * len(self.unknown_channels())
        gain = self.coverage.gain(self.position)
        score = gain / fee
        alternatives = [self.coverage.gain(point) / (np.linalg.norm(point - self.position) / 5 + fee)
                        for point in self.pending_scans]
        threshold = max(alternatives, default=0.0)
        if gain <= 0 or score <= threshold:
            return False
        point = self.position.copy()
        before_area = self.coverage.area
        event = {'kind': 'frontier_scan', 'point': point.tolist(), 'clock_before': self.virtual_seconds,
                 'actual_observations_before': len(self.coverage.observations),
                 'area_gain_prediction': gain, 'gain_per_second': score,
                 'best_planned_gain_per_second': float(threshold)}
        self.scan_unknown(point, force=True)
        self.stats['frontier_scans'] += 1
        self.stats['adaptive_extra_scans'] += 1
        self.stats['frontier_area_removed'] += before_area - self.coverage.area
        self.refine_pending()
        if not self.discovery_done and not self.cover_planner.complete(self.coverage, self.pending_scans):
            raise ValueError('Frontier scan left an uncertified future plan')
        event['clock_after'] = self.virtual_seconds
        event['actual_observations_after'] = len(self.coverage.observations)
        event['area_removed'] = before_area - self.coverage.area
        event['remaining_plan'] = [position.tolist() for position in self.pending_scans]
        self.cover_events.append(event)
        self.reuse_directions()
        return True
