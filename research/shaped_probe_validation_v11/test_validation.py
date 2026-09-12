from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_shaped import prior_seed_audit, promotion_gate, select_candidate


def test_candidate_selection_requires_all_paired_development_cases():
    rows = [{'scene_id': str(index), 'mode': mode, 'success': True, 'seconds_per_source': score}
            for index in range(18) for mode, score in (('previous', 470), ('narrow70', 455), ('narrow85', 451), ('shaped_cost', 450.5))]
    chosen, _scores = select_candidate(rows)
    assert chosen == 'narrow85'
    rows[0]['success'] = False
    with pytest.raises(ValueError):
        select_candidate(rows)


def test_seed_audit_rejects_prior_or_duplicate_seeds(tmp_path):
    (tmp_path / 'prior_scenes.json').write_text(json.dumps([{'seed': 17}]), encoding='utf-8')
    with pytest.raises(ValueError):
        prior_seed_audit([{'seed': 17}], tmp_path)
    with pytest.raises(ValueError):
        prior_seed_audit([{'seed': 19}, {'seed': 19}], tmp_path)
    assert prior_seed_audit([{'seed': 19}], tmp_path)['overlap'] == []


def test_mean_gain_does_not_override_failures_or_large_regressions():
    summary = {'combined': {'all_passed': True, 'improvement_percent': 5, 'worst_regression_percent': 10},
               'holdout': {'improvement_percent': 5}, 'stress': {'improvement_percent': 3}}
    assert promotion_gate(summary)
    summary['combined']['worst_regression_percent'] = 16
    assert not promotion_gate(summary)
    summary['combined']['worst_regression_percent'] = 10
    summary['combined']['all_passed'] = False
    assert not promotion_gate(summary)
