from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_anchored
from anchored_policy import original_ordered_route, select_preserved_network
from complete_network import radio_certificate


def test_no_registered_preserved_outer_network_is_falsely_accepted():
    with pytest.raises(ValueError, match='No registered inner contraction'):
        select_preserved_network()


def test_every_rejected_network_keeps_its_nonempty_fragments():
    original = original_ordered_route()
    for radius in (999.0, 997.5, 995.0, 990.0, 985.0, 975.0):
        route = original.copy()
        route[1:9] *= radius / 1000
        assert np.array_equal(route[[0, *range(9, 21)]], original[[0, *range(9, 21)]])
        certificate = radio_certificate(route)
        assert not certificate['complete_without_optical']
        assert certificate['remaining_area'] > 0
