from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import verify_published_results as checks


def test_exact_copy_manifest_matches_all_published_archives():
    assert checks.verify_manifest() > 1000


def test_official_numeric_summary_recomputes_from_all_ten_cases():
    rows = checks.read_csv(checks.RESULTS / 'official_practice_cases.csv')
    summary = checks.load(checks.RESULTS / 'official_practice_summary.json')
    assert checks.verify_official(rows, summary)['cases'] == 10


def test_recorded_microsecond_rounding_residual_is_preserved():
    rows = checks.read_csv(checks.RESULTS / 'official_practice_cases.csv')
    summary = checks.load(checks.RESULTS / 'official_practice_summary.json')
    result = checks.verify_official(rows, summary)
    assert result['rounding_residuals_verified'] == 10
    assert 2e-6 < result['maximum_absolute_rounding_residual_seconds'] < 3e-6


def test_changed_recorded_rounding_residual_is_rejected():
    rows = checks.read_csv(checks.RESULTS / 'official_practice_cases.csv')
    summary = checks.load(checks.RESULTS / 'official_practice_summary.json')
    selected = max(rows, key=lambda row: abs(float(row['rounding_residual_seconds'])))
    selected['rounding_residual_seconds'] = '0'
    with pytest.raises(ValueError, match='Numeric mismatch'):
        checks.verify_official(rows, summary)


def test_rounding_residual_cannot_hide_a_changed_mission_time():
    rows = checks.read_csv(checks.RESULTS / 'official_practice_cases.csv')
    summary = checks.load(checks.RESULTS / 'official_practice_summary.json')
    rows[0]['rounding_residual_seconds'] = str(float(rows[0]['rounding_residual_seconds']) + 0.01)
    rows[0]['virtual_seconds'] = str(float(rows[0]['virtual_seconds']) + 0.01)
    rows[0]['seconds_per_source'] = str(float(rows[0]['virtual_seconds']) / int(rows[0]['source_count']))
    with pytest.raises(ValueError, match='per-action rounding bound'):
        checks.verify_official(rows, summary)


def test_offline_results_are_not_accepted_as_official_evidence():
    rows = checks.read_csv(checks.RESULTS / 'official_practice_cases.csv')
    summary = checks.load(checks.RESULTS / 'official_practice_summary.json')
    rows[0]['evidence_type'] = 'local_offline_simulation'
    with pytest.raises(ValueError, match='Official table'):
        checks.verify_official(rows, summary)


def test_a_failed_official_case_cannot_be_silently_ignored():
    rows = checks.read_csv(checks.RESULTS / 'official_practice_cases.csv')
    summary = checks.load(checks.RESULTS / 'official_practice_summary.json')
    rows[0]['all_cleared'] = 'False'
    with pytest.raises(ValueError, match='complete successful'):
        checks.verify_official(rows, summary)


def test_complete_local_paired_means_trace_digests_and_physical_replays_match():
    rows = checks.read_csv(checks.RESULTS / 'offline_validation_cases.csv')
    summary = checks.load(checks.RESULTS / 'offline_validation_summary.json')
    result = checks.verify_offline(rows, summary, replay=True)
    assert result['runs'] == result['synthetic_physics_replays'] == 128
    assert result['official_calls'] == 0


def test_historical_pressure_profiles_and_correlated_noise_are_supported():
    import run_offline_baselines

    assert 'boundary_noise' in run_offline_baselines.PROFILES
    assert 'spatial_correlated' in run_offline_baselines.ERROR_MODES
    assert run_offline_baselines.EvaluationWorld is run_offline_baselines.frozen_evaluator.EvaluationWorld


def test_invented_summary_improvement_is_rejected():
    rows = checks.read_csv(checks.RESULTS / 'official_practice_cases.csv')
    summary = checks.load(checks.RESULTS / 'official_practice_summary.json')
    damaged = copy.deepcopy(summary)
    damaged['summaries']['q4']['mean_seconds_per_source'] = 299.0
    with pytest.raises(ValueError, match='Numeric mismatch'):
        checks.verify_official(rows, damaged)


def test_draft_stage_is_not_reported_as_validated():
    status = checks.load(checks.RESULTS / 'current_status.json')
    assert status['latest_unfinished_stage'] == 'v48'
    assert status['v48_status'] == 'partial_code_and_protocol_only_no_unit_tests_no_main_runs_no_scores'
    assert status['official_targets_met'] is False
