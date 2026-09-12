from __future__ import annotations

from scan_policy import ScanEconomyPolicy


MODES = {'gate130': 130.0, 'gate300': 300.0, 'gate600': 600.0, 'gate2000': 2000.0}


class VerifiedGatePolicy(ScanEconomyPolicy):
    def __init__(self, *args, mode='gate130'):
        if mode not in MODES:
            raise ValueError('Unknown actual localization gate')
        super().__init__(*args, mode='station_only')
        if self.options.get('target_radius') != 2000:
            raise ValueError('Frozen gate assumption changed; re-audit before running')
        self.options = {**self.options, 'target_radius': MODES[mode]}
        self.stats['actual_target_radius'] = MODES[mode]
