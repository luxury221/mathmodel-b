from __future__ import annotations

from flex_planner import FlexiblePatrol
from policy_v5 import ContinuousPolicy


class FlexiblePolicy(ContinuousPolicy):
    def __init__(self, port, problem, stations, variant, network, mode='flex'):
        self.flex_assignment = 'nearest' if mode == 'flex_nearest' else 'greedy'
        base_mode = 'certified_center' if problem == 'q3' else 'certified_fixed'
        super().__init__(port, problem, stations, variant, network, mode=base_mode)
        self.patrol = FlexiblePatrol(self)
        self.stats.update({key: 0 for key in ('flex_assignment_rejections', 'flex_solver_calls',
                           'flex_solver_rejections', 'flex_constraint_rejections', 'flex_certificate_rejections',
                           'flex_layouts_accepted', 'flex_stations_removed', 'flex_extra_scans', 'flex_solver_warnings')})

    def finish_discovery(self, reason):
        if reason == 'full_route':
            if not self.coverage.empty:
                raise ValueError('Moved stations cannot inherit a fixed-network termination proof')
            reason = 'coverage'
        return super().finish_discovery(reason)

    def result(self):
        result = super().result()
        result['layout_log'] = self.patrol.layout_log
        return result
