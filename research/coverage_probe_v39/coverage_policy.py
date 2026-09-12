from __future__ import annotations

import numpy as np
from coverage_planner import CoverageProbePlanner
from paired_policy import PairedScanPolicy


MODES = ('identity', 'station_bind', 'coverage_nudge')


class CoverageProbePolicy(PairedScanPolicy):
    def __init__(self, *args, mode='coverage_nudge'):
        if mode not in MODES:
            raise ValueError('Unknown coverage-probe mode')
        super().__init__(*args, mode='identity')
        self.coverage_mode = mode
        self.coverage_planner = CoverageProbePlanner(self)
        self.coverage_commitments = []
        self.stats.update({'co_probe_commitments': 0, 'co_probe_replacements': 0,
                           'co_probe_station_bindings': 0, 'co_probe_deformations': 0,
                           'co_probe_first_failures': 0})

    def transit_proposal(self, destination, stops):
        return None if self.coverage_mode == 'identity' else self.coverage_planner.choose(stops)

    def execute_transit(self, destination, proposal):
        remaining_before = tuple(self.remaining_stations)
        if tuple(index for index in remaining_before if index != proposal.station) != proposal.kept:
            raise ValueError('Stale coverage-probe commitment')
        certificate = proposal.certificate
        point = certificate.candidates[0]
        if not self.coverage_planner.complete(point, proposal.kept).empty:
            raise ValueError('Coverage-probe proposal lacks a complete continuous certificate')
        event = {'channel': proposal.channel, 'station': proposal.station, 'kept': list(proposal.kept),
                 'remaining_before': list(remaining_before), 'route': self.route.tolist(), 'point': point.tolist(),
                 'anchor': certificate.anchor.tolist(), 'vertices': certificate.vertices.tolist(),
                 'candidates': certificate.candidates.tolist(), 'direction': certificate.direction.tolist(), 'side': certificate.side,
                 'original_destination': proposal.original_destination.tolist(),
                 'before_radio': [prior.tolist() for prior in self.coverage.observations],
                 'before_optical': [prior.tolist() for prior in self.coverage.clear_observations],
                 'start_seconds': self.virtual_seconds, 'baseline_proxy_seconds': proposal.baseline_seconds,
                 'proposed_proxy_seconds': proposal.proposed_seconds, 'continuation_penalty': proposal.continuation_penalty,
                 'outcome': 'running'}
        self.coverage_commitments.append(event)
        self.stats['co_probe_commitments'] += 1
        self.pair_active = True
        try:
            self.execute_committed_action(('v_probe', proposal.channel, point, (certificate.candidates[1], certificate)))
            event['first_failed'] = proposal.channel in self.pending_probes
            self.stats['co_probe_first_failures'] += int(event['first_failed'])
            event['scan_start_seconds'] = self.virtual_seconds
            event['unknown_before'] = self.unknown_channels().copy()
            self.scan_unknown(point, force=True)
            event['scan_end_seconds'] = self.virtual_seconds
            event['remaining_after_scan'] = self.remaining_stations.copy()
            if tuple(self.remaining_stations) != remaining_before:
                raise ValueError('Fallback station removed before a real scan')
            self.reuse_directions()
            self.check_discovery()
            if self.discovery_done:
                event['outcome'] = 'actual_discovery_ended_no_deletion'
                return
            if not any(np.array_equal(point, prior) for prior in self.coverage.observations):
                raise ValueError('Proposed point was not actually scanned')
            if not self.coverage_planner.complete(None, proposal.kept).empty:
                raise ValueError('Actual scan did not certify the remaining patrol')
            self.remaining_stations[:] = proposal.kept
            self.patrol.revision = None
            self.stats['co_probe_replacements'] += 1
            self.stats['co_probe_station_bindings'] += int(np.linalg.norm(point - self.route[proposal.station]) < 1e-4)
            self.stats['co_probe_deformations'] += int(np.linalg.norm(point - proposal.original_destination) >= 1e-3)
            event['outcome'] = 'deleted_after_real_scan'
        except Exception:
            event['outcome'] = 'failed_no_deletion'
            raise
        finally:
            self.pair_active = False
            event['end_seconds'] = self.virtual_seconds
            event['remaining_after'] = self.remaining_stations.copy()
