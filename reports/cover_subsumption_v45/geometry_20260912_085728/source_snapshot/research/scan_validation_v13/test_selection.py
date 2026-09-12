from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_scan import make_scenes, select_candidate


def test_selection_prefers_simple_mode_and_rejects_incomplete_pairs():
    rows = [{'scene_id': str(index), 'mode': mode, 'success': True, 'seconds_per_source': seconds}
            for index in range(18) for mode, seconds in (('previous', 239), ('station_only', 231), ('certified_reuse', 230.99))]
    assert select_candidate(rows)[0] == 'station_only'
    with pytest.raises(ValueError):
        select_candidate(rows[:-1])


def test_fresh_validation_scene_definitions_respect_q3_rules():
    scenes = make_scenes()
    assert len(scenes) == len({scene['seed'] for scene in scenes}) == 32
    assert sum(scene['phase'] == 'holdout' for scene in scenes) == 24
    assert sum(scene['phase'] == 'stress' for scene in scenes) == 8
    for scene in scenes:
        assert 10 <= scene['count'] <= 16
        assert len({source['channel'] for source in scene['sources']}) == scene['count']
        assert all(source['heading_deg'] is None for source in scene['sources'])
