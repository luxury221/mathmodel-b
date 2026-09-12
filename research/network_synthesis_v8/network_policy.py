from __future__ import annotations

import numpy as np
from geometry import ring
from policy_v5 import ContinuousPolicy


NETWORK_MODES = ('omni_ring7', 'omni_ring8', 'omni_ring7_prescan', 'omni_ring8_prescan')


class NetworkPolicy(ContinuousPolicy):
    def __init__(self, port, problem, stations, variant, network, mode='omni_ring7'):
        if problem != 'q3' or mode not in NETWORK_MODES:
            raise ValueError('This policy only tests public omnidirectional ring layouts')
        count, radius = (7, 999.0) if 'ring7' in mode else (8, 940.0)
        points = np.vstack(([0, 0], ring(radius, count)))
        self.skip_origin_radio = not mode.endswith('_prescan')
        self.origin_radio_decided = False
        super().__init__(port, problem, points, variant, 'public_ring_v8', mode='certified_center')
        self.stats.update({'public_ring_receivers': count, 'public_ring_radius': radius, 'skipped_origin_radio': False})

    def scan_unknown(self, position, force=False):
        if not self.origin_radio_decided:
            self.origin_radio_decided = True
            if np.linalg.norm(position) > 1e-8:
                raise ValueError('Initial radio decision must occur at the origin')
            if self.skip_origin_radio:
                self.stats['skipped_origin_radio'] = True
                return
        return super().scan_unknown(position, force)

    def finish_discovery(self, reason):
        if reason == 'full_route':
            if not self.coverage.empty:
                raise ValueError('New network lacks actual continuous absence evidence')
            reason = 'coverage'
        return super().finish_discovery(reason)
