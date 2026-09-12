from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'research/shaped_probe_validation_v11'))
from validate_shaped import frozen_verifier


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def audit_directory(directory):
    manifest = load(directory / 'protocol.json')['source_hashes']
    current_changes = []
    archived_changes = []
    for relative, expected in manifest.items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            current_changes.append(relative)
        if hashlib.sha256((directory / 'source_snapshot' / relative).read_bytes()).hexdigest() != expected:
            archived_changes.append(relative)
    rows = []
    audits = []
    tiny_remeasures = []
    for path in sorted((directory / 'records').glob('*.json')):
        row = load(path)
        trace = load(directory / 'traces' / path.name)
        if hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest() != row['trace_sha256']:
            raise ValueError('Trace digest changed: ' + str(path))
        audits.append(frozen_verifier.audit_trace(trace, row['virtual_seconds']))
        prior = {}
        for index, action in enumerate(trace):
            if action['action'] != 'measure':
                continue
            positions = prior.setdefault(action['channel'], [])
            if any(0 < math.dist(position, action['position']) < 1e-5 for position in positions):
                tiny_remeasures.append({'record': path.name, 'action': index, 'channel': action['channel']})
            positions.append(action['position'])
        rows.append(row)
    summary = {'directory': str(directory.resolve()), 'runs': len(rows),
               'successes': sum(row['success'] for row in rows), 'source_files': len(manifest),
               'current_changes': current_changes, 'archived_changes': archived_changes,
               'actions': sum(audit['actions'] for audit in audits),
               'max_clock_error': max(audit['max_clock_error'] for audit in audits),
               'tiny_distinct_remeasurements': tiny_remeasures, 'modes': {}}
    for mode in sorted({row['mode'] for row in rows}):
        selected = [row for row in rows if row['mode'] == mode]
        costs = {}
        if all(row['success'] for row in selected):
            costs['mean'] = statistics.mean(row['seconds_per_source'] for row in selected)
            costs['movement'] = statistics.mean(row['world_stats']['move_meters'] / 5 / row['count'] for row in selected)
            costs['radio'] = statistics.mean((5 * row['world_stats']['measure_count'] + row['world_stats']['switch_count']) / row['count'] for row in selected)
            costs['optical'] = statistics.mean((5 * row['world_stats']['successful_clear_count'] + 3 * row['world_stats']['failed_clear_count']) / row['count'] for row in selected)
        summary['modes'][mode] = {'runs': len(selected), 'successes': sum(row['success'] for row in selected), **costs}
    return summary, rows


def comparisons(rows):
    reference = {row['scene_id']: row for row in rows if row['mode'] == 'previous'}
    results = {}
    for mode in sorted({row['mode'] for row in rows} - {'previous'}):
        candidates = [row for row in rows if row['mode'] == mode]
        if len({row['scene_id'] for row in candidates}) != len(candidates):
            raise ValueError('Duplicate candidate cases in aggregate')
        paired = [(reference[row['scene_id']], row) for row in candidates]
        entry = {'cases': len(paired), 'all_cleared': all(old['success'] and new['success'] for old, new in paired)}
        if entry['all_cleared']:
            previous_mean = statistics.mean(old['seconds_per_source'] for old, _new in paired)
            candidate_mean = statistics.mean(new['seconds_per_source'] for _old, new in paired)
            saved = [old['seconds_per_source'] - new['seconds_per_source'] for old, new in paired]
            entry.update({'previous_mean': previous_mean, 'candidate_mean': candidate_mean,
                          'improvement_percent': (1 - candidate_mean / previous_mean) * 100,
                          'faster': sum(value > 1e-7 for value in saved), 'slower': sum(value < -1e-7 for value in saved),
                          'tied': sum(abs(value) <= 1e-7 for value in saved),
                          'worst_regression_percent': max((new['seconds_per_source'] / old['seconds_per_source'] - 1) * 100 for old, new in paired)})
        results[mode] = entry
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--development', nargs='+', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a new D-drive audit directory')
    groups = {}
    audits = []
    for directory in args.development:
        audit, rows = audit_directory(directory)
        audits.append(audit)
        groups.setdefault(directory.parent.name, []).extend(rows)
    manifest_path = ROOT / 'reports/scan_validation_v13/validation_20260911_235435/selection.json'
    frozen = load(manifest_path)['source_hashes']
    frozen_changes = [relative for relative, expected in frozen.items()
                      if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected]
    analysis = {'official_calls': 0, 'frozen_files': len(frozen), 'frozen_changes': frozen_changes,
                'development_audits': audits, 'comparisons': {group: comparisons(rows) for group, rows in groups.items()},
                'total_runs': sum(audit['runs'] for audit in audits),
                'total_successes': sum(audit['successes'] for audit in audits),
                'total_actions': sum(audit['actions'] for audit in audits)}
    frozen_verifier.experiment.save(output / 'analysis.json', analysis)
    print(json.dumps({key: value for key, value in analysis.items() if key != 'development_audits'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
