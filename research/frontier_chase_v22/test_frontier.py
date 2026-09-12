import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_frontier import FrontierChasePolicy, evaluate, experiment
from geometry import seven_network
from offline_benchmark import OfflineRuleWorld


def test_same_scan_position_cannot_be_refreshed():
    world = OfflineRuleWorld([], 351411201)
    policy = FrontierChasePolicy(world.port(), 'q3', seven_network(), 'E_joint', 'grid7')
    policy.scan_unknown([0, 0], force=True)
    policy.pilot_pending = False
    before = world.virtual_seconds, len(world.trace)
    assert not policy.try_current_scan()
    assert before == (world.virtual_seconds, len(world.trace))


def test_full_task_and_continuous_coverage():
    scene = next(scene for scene in experiment.make_scenes(351411300, 'unit_test', (10,))
                 if scene['problem'] == 'q3' and scene['profile'] == 'random')
    row, _trace = evaluate(scene, 'frontier')
    assert row['success'], row['failure']
    assert row['cleared_count'] == 10
    assert row['policy_stats']['discovery_stop_reason'] == 'coverage'
