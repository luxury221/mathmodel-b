from __future__ import annotations

from range_geometry import RangeCoupler
from scan_policy import ScanEconomyPolicy
from shaped_policy import ShapedProbePolicy


class RangeCouplingMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.range_coupler = RangeCoupler()
        self.range_events = []
        self.stats.update({'range_pairs': 0, 'range_branch_cuts': 0, 'range_region_cuts': 0})

    def measure(self, channel, position, active=False, opportunistic=False):
        response = super().measure(channel, position, active, opportunistic)
        state = self.states[channel]
        if state.status != 'DETECTED' or not state.negatives:
            return response
        before = state.region
        old_pairs = self.range_coupler.pairs
        old_cuts = self.range_coupler.effective_cuts
        if self.problem == 'q3':
            updated = self.range_coupler.refine_omni(state)
        else:
            updated = self.range_coupler.refine_directional(state, self.direction_beliefs[channel])
        changed = not updated.equals(before)
        if changed:
            state.region = updated
            self.snapshot(state)
        self.stats['range_pairs'] = self.range_coupler.pairs
        self.stats['range_branch_cuts'] = self.range_coupler.effective_cuts
        self.stats['range_region_cuts'] += int(changed)
        if self.range_coupler.pairs != old_pairs:
            self.range_events.append({'channel': channel, 'clock': self.virtual_seconds,
                                      'positive_count': len(state.positives), 'negative_count': len(state.negatives),
                                      'new_pairs': self.range_coupler.pairs - old_pairs,
                                      'branch_cuts': self.range_coupler.effective_cuts - old_cuts,
                                      'area_before': before.area, 'area_after': updated.area,
                                      'region_changed': changed})
        return response


class OmniRangePolicy(RangeCouplingMixin, ScanEconomyPolicy):
    def __init__(self, *args, mode='coupled'):
        if mode != 'coupled':
            raise ValueError('Unknown omnidirectional coupling mode')
        super().__init__(*args, mode='station_only')


class DirectionalRangePolicy(RangeCouplingMixin, ShapedProbePolicy):
    def __init__(self, *args, mode='coupled'):
        if mode != 'coupled':
            raise ValueError('Unknown directional coupling mode')
        super().__init__(*args, mode='shaped_cost')
