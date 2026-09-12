from __future__ import annotations

import numpy as np
from motion_policy import JointMotionPolicy
from paired_planner import PairedScanPlanner
from shaped_policy import ShapedProbePolicy


MODES = ('identity', 'paired')


class PairedScanPolicy(ShapedProbePolicy):
    def __init__(self, *args, mode='paired'):
        if mode not in MODES:
            raise ValueError('Unknown paired-scan mode')
        super().__init__(*args, mode='shaped_cost')
        self.pair_mode = mode
        self.pair_planner = PairedScanPlanner(self)
        self.pair_active = False
        self.commitment_events = []
        self.stats.update({'pair_commitments': 0, 'pair_station_deletions': 0,
                           'pair_scans': 0, 'pair_early_discovery': 0})

    build_stops = JointMotionPolicy.build_stops
    execute_stop = JointMotionPolicy.execute_stop
    run = JointMotionPolicy.run

    def route_positions(self, stops):
        return np.asarray([stop[2] for stop in stops])

    def transit_proposal(self, destination, stops):
        return self.pair_planner.choose(stops) if self.pair_mode == 'paired' else None

    def reuse_stop(self):
        if self.pair_active:
            self.reuse_directions()
            self.check_discovery()
        else:
            super().reuse_stop()

    def execute_committed_action(self, stop):
        kind, channel, destination, extra = stop
        if kind == 'scan_here':
            return
        state = self.states[channel]
        if state.status in ('CLEARED', 'ABSENT'):
            return
        if kind == 'near':
            if not self.clear(channel, destination):
                raise ValueError('Committed near action failed at its actual certified point')
            self.stats['mec_clears'] += 1
            self.reuse_stop()
            return
        if kind in ('probe', 'v_probe', 'v_mirror'):
            if state.status == 'NEAR' or any(np.linalg.norm(destination - previous) < 1e-5
                                            for previous in state.measured_positions):
                return
            if kind == 'v_mirror' and (channel not in self.pending_probes
                                      or not np.array_equal(destination, self.pending_probes[channel])):
                return
        self.execute_stop((kind, channel, destination, extra))

    def execute_transit(self, destination, proposal):
        remaining_before = tuple(self.remaining_stations)
        if tuple(index for index in remaining_before if index not in proposal.removed) != proposal.kept:
            raise ValueError('Stale pair commitment')
        if not self.pair_planner.complete(proposal.stops, proposal.kept).empty:
            raise ValueError('Pair commitment lacks a continuous certificate')
        event = {'start_seconds': self.virtual_seconds,
                 'before_radio': [point.tolist() for point in self.coverage.observations],
                 'before_optical': [point.tolist() for point in self.coverage.clear_observations],
                 'remaining_before': list(remaining_before), 'removed': list(proposal.removed),
                 'kept': list(proposal.kept), 'route': self.route.tolist(),
                 'planned_points': [stop[2].tolist() for stop in proposal.stops],
                 'planned_actions': [(stop[0], stop[1]) for stop in proposal.stops],
                 'baseline_proxy_seconds': proposal.baseline_seconds,
                 'proposed_proxy_seconds': proposal.proposed_seconds, 'checkpoints': [], 'outcome': 'running'}
        self.commitment_events.append(event)
        self.stats['pair_commitments'] += 1
        self.pair_active = True
        try:
            for stop in proposal.stops:
                if self.discovery_done:
                    event['outcome'] = 'actual_discovery_ended_no_deletion'
                    self.stats['pair_early_discovery'] += 1
                    return
                self.execute_committed_action(stop)
                point = stop[2]
                checkpoint = {'scan_start_seconds': self.virtual_seconds,
                              'unknown_before': self.unknown_channels().copy(), 'point': point.tolist()}
                self.scan_unknown(point, force=True)
                checkpoint.update({'scan_end_seconds': self.virtual_seconds,
                                   'actual_scan_recorded': any(np.array_equal(point, prior) for prior in self.coverage.observations),
                                   'remaining': self.remaining_stations.copy()})
                event['checkpoints'].append(checkpoint)
                self.stats['pair_scans'] += int(checkpoint['actual_scan_recorded'])
                if tuple(self.remaining_stations) != remaining_before:
                    raise ValueError('Fallback stations deleted before pair completion')
                self.reuse_directions()
                self.check_discovery()
            if self.discovery_done:
                event['outcome'] = 'actual_discovery_ended_no_deletion'
                self.stats['pair_early_discovery'] += 1
                return
            if not all(checkpoint['actual_scan_recorded'] for checkpoint in event['checkpoints']):
                raise ValueError('Pair contains an unobserved scan')
            if not self.pair_planner.complete((), proposal.kept).empty:
                raise ValueError('Actual evidence does not certify the remaining patrol')
            self.remaining_stations[:] = proposal.kept
            self.patrol.revision = None
            self.stats['pair_station_deletions'] += len(proposal.removed)
            event['outcome'] = 'deleted_after_real_scans'
        except Exception:
            event['outcome'] = 'failed_no_deletion'
            raise
        finally:
            self.pair_active = False
            event['end_seconds'] = self.virtual_seconds
            event['remaining_after'] = self.remaining_stations.copy()
