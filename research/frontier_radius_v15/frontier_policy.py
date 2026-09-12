from __future__ import annotations

from radial_policy import RadialPolicy


MODES = ('pilot1700', 'pilot1725', 'pilot1725_station_only')


class FrontierPolicy(RadialPolicy):
    def __init__(self, *args, mode='pilot1725_station_only'):
        if mode not in MODES:
            raise ValueError('Unknown frontier-radius mode')
        self.station_only = mode.endswith('station_only')
        super().__init__(*args, mode='pilot1600')
        self.radial_radius = 1700.0 if mode == 'pilot1700' else 1725.0
        self.stats['suppressed_opportunistic_scans'] = 0

    def scan_unknown(self, position, force=False):
        if self.station_only and not force:
            self.stats['suppressed_opportunistic_scans'] += 1
            return None
        return super().scan_unknown(position, force)
