from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'research/motion_stage_audit'), str(ROOT / 'research/practice_baseline')]
from audit_motion import audit_directory, comparisons
from baseline import verify_hashes


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def describe(rows):
    modes = {}
    for mode in sorted({row['mode'] for row in rows}):
        selected = [row for row in rows if row['mode'] == mode]
        entry = {'runs': len(selected), 'successes': sum(row['success'] for row in selected)}
        if all(row['success'] for row in selected):
            entry.update({
                'mean': statistics.mean(row['seconds_per_source'] for row in selected),
                'movement': statistics.mean(row['world_stats']['move_meters'] / 5 / row['count'] for row in selected),
                'radio': statistics.mean((5 * row['world_stats']['measure_count'] + row['world_stats']['switch_count']) / row['count'] for row in selected),
                'optical': statistics.mean((5 * row['world_stats']['successful_clear_count'] + 3 * row['world_stats']['failed_clear_count']) / row['count'] for row in selected),
            })
        entry['first_changes'] = sum(row['policy_stats'].get('latency_first_changes', 0) for row in selected)
        events = [event for row in selected for event in row.get('latency_events', [])]
        if events:
            entry['route_calls'] = len(events)
            entry['proxy_worsening_count'] = sum(event['selected_proxy_seconds'] > event['reference_proxy_seconds'] + 1e-7 for event in events)
            entry['mean_local_proxy_saving'] = statistics.mean(event['reference_proxy_seconds'] - event['selected_proxy_seconds'] for event in events)
        predictions = [event for row in selected for event in row.get('scan_calibration', [])]
        if predictions:
            errors = [event['expected_discoveries'] - event['actual_discoveries'] for event in predictions]
            entry['prediction_diagnostics'] = {'scans': len(errors), 'mean_error': statistics.mean(errors), 'mae': statistics.mean(abs(error) for error in errors)}
        entry['effective_target_radii'] = sorted({row['effective_target_radius'] for row in selected if 'effective_target_radius' in row})
        modes[mode] = entry
    return {'modes': modes, 'comparisons': comparisons(rows)}


def audit_stage(directories, output):
    output = output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a fresh D-drive output directory')
    reports = []
    tail_diagnostics = []
    for directory in directories:
        summary, rows = audit_directory(directory)
        summary.pop('modes')
        summary['problems'] = {problem: describe([row for row in rows if row['problem'] == problem]) for problem in sorted({row['problem'] for row in rows})}
        reports.append(summary)
        if summary['current_changes'] or summary['archived_changes'] or summary['tiny_distinct_remeasurements'] or summary['max_clock_error'] > 1e-6:
            raise ValueError('Integrity/action audit failed')
        for row in rows:
            if row['mode'] != 'previous' or not row['success']:
                continue
            path = directory / 'traces' / (row['scene_id'] + '__previous.json')
            trace = load(path)
            successes = [index for index, action in enumerate(trace) if action['action'] == 'clear' and action['response']['result'] == 'success']
            if len(successes) != row['count']:
                raise ValueError('Clear count mismatch')
            last = successes[-1]
            tail_diagnostics.append({'problem': row['problem'], 'scene_id': row['scene_id'], 'count': row['count'], 'full_seconds_per_source': row['seconds_per_source'], 'post_last_clear_seconds_per_source': (row['virtual_seconds'] - trace[last]['virtual_seconds']) / row['count'], 'post_last_clear_actions': len(trace) - last - 1})
    frozen_path = ROOT / 'reports/scan_validation_v13/validation_20260911_235435/selection.json'
    selection = load(frozen_path)
    frozen = selection['source_hashes']
    changed = [relative for relative, digest in frozen.items() if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest]
    if changed:
        raise ValueError('Frozen dependencies changed')
    verify_hashes()
    report = {'official_calls': 0, 'new_holdout_used': False, 'acceptance_criterion': 'Official practice case-equal complete-mission seconds per source; offline scores are screening only', 'directories': reports, 'frozen_dependency_count': len(frozen), 'frozen_dependencies_changed': changed, 'original_client_hashes_verified': True, 'post_last_clear_diagnostics_evaluator_only': tail_diagnostics, 'diagnostic_warning': 'Post-last-clear time is not deducted from the metric and is not visible to the policy'}
    output.mkdir(parents=True)
    (output / 'analysis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'output': str(output), 'runs': sum(report['runs'] for report in reports), 'successes': sum(report['successes'] for report in reports), 'actions': sum(report['actions'] for report in reports), 'frozen_files': len(frozen)}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directories', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit_stage(args.directories, args.output)


if __name__ == '__main__':
    main()
