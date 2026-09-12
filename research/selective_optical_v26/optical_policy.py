from __future__ import annotations

import numpy as np
from complete_network import radio_certificate
from shaped_policy import ShapedProbePolicy


MODES = ('network_only', 'sentinel', 'radio_first')


class SelectiveOpticalPolicy(ShapedProbePolicy):
    def __init__(self, port, problem, stations, variant, network, mode='sentinel'):
        if mode not in MODES or problem != 'q4':
            raise ValueError('Unknown selective optical policy')
        if not radio_certificate(stations)['complete_without_optical']:
            raise ValueError('Skipping origin optical requires a full radio-only coverage certificate')
        super().__init__(port, problem, stations, variant, network, mode='shaped_cost')
        self.optical_mode = mode
        self.initial_sweep_pending = True
        self.origin_radio_pending = True
        self.initial_optical_events = []
        self.stats.update({'origin_optical_suppressed': 0, 'sentinel_attempts': 0,
                           'sentinel_hits': 0, 'sentinel_triggered_sweeps': 0,
                           'radio_near_triggered_sweeps': 0})

    def sentinel_sweep(self, position):
        tried = set()
        hit = False
        self.sweeping = True
        try:
            for channel in (1, 2):
                self.stats['sentinel_attempts'] += 1
                self.stats['optical_unknown_attempts'] += 1
                success = self.clear(channel, position)
                tried.add(channel)
                self.stats['optical_unknown_discoveries'] += int(success)
                self.stats['sentinel_hits'] += int(success)
                self.initial_optical_events.append({'channel': channel, 'success': success,
                                                     'clock': self.virtual_seconds, 'sentinel': True})
                if success:
                    hit = True
                    break
            if hit:
                for state in self.states.values():
                    if state.channel in tried or state.status in ('CLEARED', 'ABSENT'):
                        continue
                    was_unknown = state.status == 'UNKNOWN'
                    self.stats['optical_unknown_attempts'] += int(was_unknown)
                    success = self.clear(state.channel, position)
                    self.stats['optical_unknown_discoveries'] += int(was_unknown and success)
                    self.initial_optical_events.append({'channel': state.channel, 'success': success,
                                                         'clock': self.virtual_seconds, 'sentinel': False})
        finally:
            self.sweeping = False
        if hit:
            self.stats['sentinel_triggered_sweeps'] += 1
            self.stats['common_optical_sweeps'] += 1
            if self.unknown_channels():
                self.coverage.observe_clear_absence(position)
                self.clear_event_log.append((tuple(self.unknown_channels()), np.asarray(position).copy()))
        else:
            self.stats['origin_optical_suppressed'] += 1
        self.check_discovery()

    def common_clear(self, position, all_unresolved=False):
        if not self.initial_sweep_pending:
            return super().common_clear(position, all_unresolved)
        self.initial_sweep_pending = False
        if not np.array_equal(position, np.zeros(2)) or self.virtual_seconds != 0:
            raise ValueError('Initial optical choice is restricted to the actual origin before any action')
        if self.optical_mode == 'network_only':
            return super().common_clear(position, all_unresolved)
        if self.optical_mode == 'sentinel':
            return self.sentinel_sweep(position)
        self.stats['origin_optical_suppressed'] += 1
        return None

    def scan_unknown(self, position, force=False):
        initial = self.origin_radio_pending and np.array_equal(position, np.zeros(2))
        if initial:
            self.origin_radio_pending = False
        result = super().scan_unknown(position, force)
        if initial and self.optical_mode == 'radio_first' and any(state.status == 'NEAR' for state in self.states.values()):
            self.stats['radio_near_triggered_sweeps'] += 1
            self.stats['origin_optical_suppressed'] -= 1
            super().common_clear(position, all_unresolved=True)
        return result
