from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'research/motion_stage_audit'))
from audit_motion import audit_directory, comparisons
from validate_shaped import frozen_verifier


def statistics_for(rows):
    result = {}
    for mode in sorted({row['mode'] for row in rows}):
        selected = [row for row in rows if row['mode'] == mode]
        entry = {'runs': len(selected), 'successes': sum(row['success'] for row in selected)}
        if all(row['success'] for row in selected):
            entry.update({'mean': statistics.mean(row['seconds_per_source'] for row in selected),
                          'movement': statistics.mean(row['world_stats']['move_meters'] / 5 / row['count'] for row in selected),
                          'radio': statistics.mean((5 * row['world_stats']['measure_count'] + row['world_stats']['switch_count']) / row['count'] for row in selected),
                          'optical': statistics.mean((5 * row['world_stats']['successful_clear_count'] + 3 * row['world_stats']['failed_clear_count']) / row['count'] for row in selected),
                          'mean_active_probes': statistics.mean(row['policy_stats'].get('active_measures', 0) for row in selected),
                          'mean_wall_seconds': statistics.mean(row['wall_seconds'] for row in selected),
                          'max_wall_seconds': max(row['wall_seconds'] for row in selected)})
        result[mode] = entry
    return result


def forecast_calibration(rows):
    result = {}
    for mode in ('rollout', 'cost_clear'):
        events = [event for row in rows if row['mode'] == mode for event in row.get('forecast_events', [])]
        if not events:
            continue
        errors = [event['actual_minus_predicted_mean'] for event in events]
        result[mode] = {'events': len(events), 'mean_signed_error': statistics.mean(errors),
                        'mean_absolute_error': statistics.mean(abs(error) for error in errors),
                        'max_absolute_error': max(abs(error) for error in errors),
                        'forecast_failures_penalized': sum(len(option['failures']) for event in events for option in event['options']),
                        'caveat': 'Actual local interval includes other-channel opportunistic radio fees; forecast does not'}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a new D-drive audit directory')
    groups = {}
    audits = []
    roots = ('range_coupling_v19', 'local_rollout_v20', 'adaptive_cover_v21', 'frontier_chase_v22')
    for version in roots:
        for directory in sorted((ROOT / 'reports' / version).iterdir()):
            if not directory.is_dir() or not (directory / 'source_integrity.json').is_file():
                continue
            audit, rows = audit_directory(directory)
            if audit['archived_changes'] or audit['tiny_distinct_remeasurements'] or audit['max_clock_error'] != 0:
                raise ValueError('Archive, timing, or remeasurement audit failed: ' + str(directory))
            revision = 'r3' if directory.name.startswith(('dev3_', 'q4_expand3_')) else 'r2' if directory.name.startswith('dev2_') else 'r1'
            historical = version == 'range_coupling_v19' and revision != 'r3'
            audit['historical_source_revision'] = historical
            if audit['current_changes'] and not historical:
                raise ValueError('Unexpected current source changes: ' + str(directory))
            audits.append(audit)
            for problem in sorted({row['problem'] for row in rows}):
                key = version + ('_' + revision if version == 'range_coupling_v19' else '') + '_' + problem
                groups.setdefault(key, []).extend(row for row in rows if row['problem'] == problem)
    selection_path = ROOT / 'reports/scan_validation_v13/validation_20260911_235435/selection.json'
    frozen = json.loads(selection_path.read_text(encoding='utf-8'))['source_hashes']
    frozen_changes = [relative for relative, digest in frozen.items()
                      if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest]
    if frozen_changes:
        raise ValueError('Frozen dependencies changed')
    result = {'official_calls': 0, 'new_holdout_used': False, 'frozen_files': len(frozen), 'frozen_changes': frozen_changes,
              'runs': sum(audit['runs'] for audit in audits), 'successes': sum(audit['successes'] for audit in audits),
              'actions_audited': sum(audit['actions'] for audit in audits), 'development_audits': audits,
              'comparisons': {key: comparisons(rows) for key, rows in groups.items()},
              'cost_breakdown': {key: statistics_for(rows) for key, rows in groups.items()},
              'forecast_calibration': forecast_calibration(groups.get('local_rollout_v20_q3', [])),
              'goal_achieved': False}
    frozen_verifier.experiment.save(output / 'analysis.json', result)
    print(json.dumps({key: value for key, value in result.items() if key not in ('development_audits', 'cost_breakdown')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
