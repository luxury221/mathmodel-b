from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_radial_optical
from anchored_policy import original_ordered_route
from complete_network import radio_certificate
from offline_benchmark import OfflineRuleWorld, Source
from radial_optical_policy import RadialOpticalPolicy, select_network


@pytest.fixture(scope='module')
def stations():
    choice = select_network()
    assert choice['selected'] is not None
    return np.asarray(choice['selected']['stations'])


def test_network_preserves_angles_and_is_radio_complete(stations):
    original = original_ordered_route()
    assert radio_certificate(stations)['complete_without_optical']
    assert np.allclose(stations[1:] / np.linalg.norm(stations[1:], axis=1)[:, None],
                       original[1:] / np.linalg.norm(original[1:], axis=1)[:, None])


def test_identity_adapter_reproduces_the_frozen_trace(stations):
    scene = next(scene for scene in run_radial_optical.experiment.make_scenes(361415100, 'unit', (13,))
                 if scene['problem'] == 'q4' and scene['profile'] == 'random')
    original, _trace = run_radial_optical.evaluate(scene, 'previous', stations)
    identity, _trace = run_radial_optical.evaluate(scene, 'identity', stations)
    assert original['success'] and identity['success']
    assert original['trace_sha256'] == identity['trace_sha256']


def test_missed_sentinels_and_backside_near_sources_still_clear(stations):
    scene = {'id': 'unit_v28_hidden_origin', 'phase': 'unit', 'problem': 'q4', 'profile': 'hidden_origin',
             'count': 10, 'seed': 361415101, 'error_mode': 'constant_extreme',
             'sources': [{'channel': channel, 'position': [0.01, 0.0], 'radius': 1000.0, 'heading_deg': 0.0}
                         for channel in range(3, 13)]}
    row, trace = run_radial_optical.evaluate(scene, 'sentinel', stations)
    assert row['success'], row['failure']
    assert row['policy_stats']['origin_optical_suppressed'] == 1
    assert not any(action['response']['result'] == 'near' for action in trace
                   if action['action'] == 'measure' and action['position'] == [0.0, 0.0])


def test_nonempty_coverage_blocks_route_exhaustion(stations):
    world = OfflineRuleWorld([Source(channel, (700.0, 0.0), 1000, None) for channel in range(1, 11)], 361415102)
    policy = RadialOpticalPolicy(world.port(), 'q4', stations, 'F_route', 'dual21', mode='sentinel')
    with pytest.raises(ValueError, match='actual complete coverage'):
        policy.finish_discovery('full_route')
