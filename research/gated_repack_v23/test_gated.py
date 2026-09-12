import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gated import evaluate, experiment


@pytest.mark.parametrize('profile,count', [('random', 10), ('boundary', 13), ('near_origin', 16)])
def test_identity_matches_frozen_trace(profile, count):
    scene = next(scene for scene in experiment.make_scenes(361411300, 'unit_test', (count,))
                 if scene['problem'] == 'q3' and scene['profile'] == profile)
    previous, _trace = evaluate(scene, 'previous')
    identity, _trace = evaluate(scene, 'identity')
    assert previous['success'] and identity['success']
    assert previous['trace_sha256'] == identity['trace_sha256']


@pytest.mark.parametrize('mode', ['repack', 'repack_scan'])
def test_new_modes_clear_full_task(mode):
    scene = next(scene for scene in experiment.make_scenes(361411400, 'unit_test', (10,))
                 if scene['problem'] == 'q3' and scene['profile'] == 'random')
    row, _trace = evaluate(scene, mode)
    assert row['success'], row['failure']
    assert row['cleared_count'] == 10
