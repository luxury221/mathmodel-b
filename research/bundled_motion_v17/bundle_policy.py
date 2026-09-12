from __future__ import annotations

import numpy as np
from bundle_routes import bundle_order
from motion_policy import JointMotionPolicy


MODES = ('bundled', 'bundled_transit')


class BundledMotionPolicy(JointMotionPolicy):
    def __init__(self, *args, mode='bundled'):
        if mode not in MODES:
            raise ValueError('Unknown bundled-motion mode')
        super().__init__(*args, mode='transit' if mode == 'bundled_transit' else 'terminal')
        self.bundle_mode = mode
        self.bundle_events = []

    def bundle_endpoints(self, stops):
        entries = np.asarray([stop[2] for stop in stops])
        exits = []
        for kind, channel, point, _extra in stops:
            exits.append(self.center_radius(self.states[channel])[0]
                         if kind in ('v_probe', 'v_mirror', 'probe', 'anchor_clear') else point)
        return entries, np.asarray(exits)

    def run(self):
        self.common_clear(self.position, all_unresolved=True)
        self.scan_unknown(self.position, force=True)
        for _iteration in range(400):
            self.check_discovery()
            if self.discovery_done and not any(state.status in ('DETECTED', 'NEAR') for state in self.states.values()):
                return self.result()
            stops = self.build_stops()
            if not stops:
                continue
            entries, exits = self.bundle_endpoints(stops)
            order = bundle_order(entries, exits, self.position)
            stop = stops[order[0]]
            self.bundle_events.append({'position': self.position.tolist(), 'clock': self.virtual_seconds,
                                       'first_action': stop[0], 'first_identifier': stop[1],
                                       'bundle_count': len(stops), 'first_entry': entries[order[0]].tolist(),
                                       'predicted_exit_only': exits[order[0]].tolist()})
            proposal = self.transit_proposal(stop[2], stops)
            if proposal is not None:
                self.execute_transit(stop[2], proposal)
            else:
                self.execute_stop(stop)
        raise ValueError('Bundled-motion action limit exceeded')
