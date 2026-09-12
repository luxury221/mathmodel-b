from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gate
from gate_policy import VerifiedGatePolicy
from geometry import seven_network
from offline_benchmark import OfflineRuleWorld, Source
from policy import MODES
from scan_policy import ScanEconomyPolicy


def test_actual_parameter_is_not_the_get_default_and_is_not_shared():
    world = OfflineRuleWorld([Source(channel, (500.0, 0.0), 1000, None) for channel in range(1, 11)], 361416101)
    arguments = (world.port(), 'q3', seven_network(), 'E_joint', 'grid7')
    previous = ScanEconomyPolicy(*arguments, mode='station_only')
    candidate = VerifiedGatePolicy(*arguments, mode='gate130')
    after = ScanEconomyPolicy(*arguments, mode='station_only')
    assert previous.options['target_radius'] == after.options['target_radius'] == MODES['adaptive']['target_radius'] == 2000
    assert candidate.options['target_radius'] == 130
    assert candidate.options is not previous.options


@pytest.mark.parametrize('profile,count', [('random', 10), ('boundary', 16)])
def test_gate2000_is_exactly_the_frozen_policy(profile, count):
    scene = next(scene for scene in run_gate.experiment.make_scenes(361416200, 'unit', (count,))
                 if scene['problem'] == 'q3' and scene['profile'] == profile)
    previous, _trace = run_gate.evaluate(scene, 'previous')
    identity, _trace = run_gate.evaluate(scene, 'gate2000')
    assert previous['success'] and identity['success']
    assert previous['trace_sha256'] == identity['trace_sha256']


def test_low_gate_eventually_resolves_all_known_sources():
    scene = next(scene for scene in run_gate.experiment.make_scenes(361416400, 'unit', (13,))
                 if scene['problem'] == 'q3' and scene['profile'] == 'adversarial_heading')
    row, _trace = run_gate.evaluate(scene, 'gate130')
    assert row['success'], row['failure']
    assert row['cleared_count'] == 13
