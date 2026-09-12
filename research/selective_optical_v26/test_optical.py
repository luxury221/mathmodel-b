from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_study
from complete_network import radio_certificate, select_network
from geometry import dual_ring_network
from offline_benchmark import OfflineRuleWorld, Source
from optical_policy import SelectiveOpticalPolicy


@pytest.fixture(scope='module')
def stations():
    selection = select_network()
    assert selection['selected'] is not None
    return np.asarray(selection['selected']['stations'])


def test_original_network_requires_optical_but_new_network_does_not(stations):
    assert not radio_certificate(dual_ring_network())['complete_without_optical']
    assert radio_certificate(stations)['complete_without_optical']


def test_sentinel_misses_do_not_create_global_optical_evidence(stations):
    sources = [Source(channel, (800.0, 0.0), 1000, None) for channel in range(3, 13)]
    world = OfflineRuleWorld(sources, 361413001)
    policy = SelectiveOpticalPolicy(world.port(), 'q4', stations, 'F_route', 'radio_complete_v26', mode='sentinel')
    before = policy.coverage.region.wkb
    policy.common_clear(np.zeros(2), True)
    assert policy.coverage.region.wkb == before
    assert not policy.coverage.clear_observations
    assert len(world.trace) == 2
    assert world.virtual_seconds == 6
    assert len(policy.unknown_channels()) == 20


def test_sentinel_hit_checks_each_channel_once(stations):
    sources = [Source(channel, (0.01, 0.0), 1000, 0) for channel in range(2, 12)]
    world = OfflineRuleWorld(sources, 361413002)
    policy = SelectiveOpticalPolicy(world.port(), 'q4', stations, 'F_route', 'radio_complete_v26', mode='sentinel')
    policy.common_clear(np.zeros(2), True)
    assert len(world.trace) == 20
    assert len({action['channel'] for action in world.trace}) == 20
    assert not world.remaining
    assert len(policy.coverage.clear_observations) == 1
    assert world.virtual_seconds == policy.virtual_seconds == 80


@pytest.mark.parametrize('mode', ['sentinel', 'radio_first'])
def test_all_nearby_sources_can_face_away_from_origin(stations, mode):
    scene = {'id': 'unit_v26_hidden_origin_' + mode, 'phase': 'unit', 'problem': 'q4',
             'profile': 'hidden_origin', 'count': 10, 'seed': 361413100 + int(mode == 'radio_first'),
             'error_mode': 'constant_extreme',
             'sources': [{'channel': channel, 'position': [0.01, 0.0], 'radius': 1000.0, 'heading_deg': 0.0}
                         for channel in range(3, 13)]}
    row, trace = run_study.evaluate(scene, mode, stations)
    assert row['success'], row['failure']
    assert row['cleared_count'] == 10
    assert not any(action['response']['result'] == 'near' for action in trace
                   if action['position'] == [0.0, 0.0] and action['action'] == 'measure')
    assert row['policy_stats']['origin_optical_suppressed'] == 1
