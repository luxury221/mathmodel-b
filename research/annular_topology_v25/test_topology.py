from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from topology import certify_layout, configurations, dual_ring_network, public_witnesses, uncovered_witnesses


def test_original_network_remains_continuously_complete():
    assert certify_layout(dual_ring_network())['complete']


def test_screen_rejects_one_sided_observations():
    stations = np.array([[100, 0], [100, 10], [100, -10]])
    assert uncovered_witnesses(stations, np.array([[0, 0]])).all()


def test_screen_accepts_surrounding_observations():
    stations = np.array([[100, 0], [-100, 100], [-100, -100]])
    assert not uncovered_witnesses(stations, np.array([[0, 0]])).any()


def test_incomplete_network_is_not_certified():
    result = certify_layout(np.array([[0, 0], [500, 0], [-500, 0]]))
    assert not result['complete']
    assert result['remaining_area'] > 0


def test_grid_is_public_and_node_capped():
    options = list(configurations())
    assert all(option['inner_count'] + option['outer_count'] + 1 <= 21 for option in options)
    assert len(public_witnesses()) > 1000
    assert all('source_count' not in option for option in options)
