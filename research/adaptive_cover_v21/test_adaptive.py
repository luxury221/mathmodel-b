import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_adaptive import AdaptiveSourceCoverPolicy, evaluate, experiment
from geometry import seven_network
from offline_benchmark import OfflineRuleWorld


def test_repacking_does_not_count_future_points_as_observations():
    world = OfflineRuleWorld([], 341411201)
    policy = AdaptiveSourceCoverPolicy(world.port(), 'q3', seven_network(), 'E_joint', 'grid7')
    policy.scan_unknown([0, 0], force=True)
    planner = policy.cover_planner
    before = (policy.coverage.region.wkb, len(policy.coverage.observations), world.virtual_seconds, len(world.trace))
    future = planner.refine(policy.coverage, list(policy.route[1:]), True)
    assert planner.complete(policy.coverage, future)
    assert (policy.coverage.region.wkb, len(policy.coverage.observations), world.virtual_seconds, len(world.trace)) == before
    assert not policy.discovery_done
    assert not policy.coverage.empty


def test_incomplete_scan_plan_cannot_be_accepted():
    world = OfflineRuleWorld([], 341411202)
    policy = AdaptiveSourceCoverPolicy(world.port(), 'q3', seven_network(), 'E_joint', 'grid7')
    with pytest.raises(ValueError):
        policy.cover_planner.refine(policy.coverage, [np.array([0.0, 0.0])], True)


@pytest.mark.parametrize('mode', ['source_first', 'source_repack'])
def test_full_task_and_clock(mode):
    scene = next(scene for scene in experiment.make_scenes(341411300, 'unit_test', (10,))
                 if scene['problem'] == 'q3' and scene['profile'] == 'random')
    row, _trace = evaluate(scene, mode)
    assert row['success'], row['failure']
    assert row['cleared_count'] == 10


def test_empty_world_requires_full_actual_coverage():
    world = OfflineRuleWorld([], 341411204)
    policy = AdaptiveSourceCoverPolicy(world.port(), 'q3', seven_network(), 'E_joint', 'grid7')
    result = policy.run()
    assert policy.coverage.empty
    assert len(result['declared_absent']) == 20
    assert world.virtual_seconds == result['virtual_seconds']
