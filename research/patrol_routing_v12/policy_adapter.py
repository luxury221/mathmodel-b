from __future__ import annotations

from contextlib import ExitStack, contextmanager
import importlib
from unittest.mock import patch

import numpy as np
from geometry import ring
from shaped_policy import ShapedProbePolicy


@contextmanager
def route_context(router):
    with ExitStack() as stack:
        if router is not None:
            for name in ('geometry', 'plan_policy', 'policy', 'policy_v5'):
                module = importlib.import_module(name)
                stack.enter_context(patch.object(module, 'open_route', router))
        yield


class LayoutPolicy(ShapedProbePolicy):
    def __init__(self, port, problem, stations, variant, network, mode='shaped_cost'):
        points = np.vstack(([0, 0], ring(1011.5, 7), ring(1860, 13, 15)))
        super().__init__(port, problem, points, variant, 'public_layout_v12', mode=mode)

    def finish_discovery(self, reason):
        if reason == 'full_route':
            if not self.coverage.empty:
                raise ValueError('New layout cannot terminate without actual continuous coverage')
            reason = 'coverage'
        return super().finish_discovery(reason)
