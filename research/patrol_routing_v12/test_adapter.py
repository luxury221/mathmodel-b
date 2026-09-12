from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_benchmark import LayoutPolicy, evaluate, experiment, route_context
from geometry import dual_ring_network
from offline_benchmark import OfflineRuleWorld
from strong_routes import StrongRouter
import policy_v5


def test_route_hooks_restore_frozen_module_functions_after_failure():
    original = policy_v5.open_route
    router = StrongRouter()
    with pytest.raises(RuntimeError), route_context(router):
        assert policy_v5.open_route is router
        raise RuntimeError('injected local exception')
    assert policy_v5.open_route is original


def test_new_layout_constructor_does_not_create_fake_observations():
    world = OfflineRuleWorld([], 252511103)
    policy = LayoutPolicy(world.port(), 'q4', dual_ring_network(), 'F_route', 'dual21')
    assert policy.coverage.observations == []
    assert policy.coverage.clear_observations == []
    assert world.trace == []
    assert world.virtual_seconds == 0


def test_routed_q3_replay_has_identical_real_actions():
    scene = next(scene for scene in experiment.make_scenes(253100000, 'unit_test', (10,)) if scene['problem'] == 'q3' and scene['profile'] == 'random')
    first, trace = evaluate(scene, 'guided')
    second, repeated = evaluate(scene, 'guided')
    assert first['success'] and second['success']
    assert first['trace_sha256'] == second['trace_sha256']
    assert trace == repeated
    assert first['timing_audit']['max_clock_error'] == 0


def test_new_layout_empty_world_finishes_only_after_real_coverage():
    world = OfflineRuleWorld([], 252511104)
    policy = LayoutPolicy(world.port(), 'q4', dual_ring_network(), 'F_route', 'dual21')
    result = policy.run()
    assert policy.coverage.empty
    assert result['virtual_seconds'] == world.virtual_seconds
    assert set(result['declared_absent']) == set(range(1, 21))
    assert world.stats['measure_count'] > 20
