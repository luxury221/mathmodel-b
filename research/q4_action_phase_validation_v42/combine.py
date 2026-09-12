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
sys.path.insert(0, str(ROOT / 'research/q4_action_phase_v42'))
import run_phase as runner


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batches', type=Path, nargs=2, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a fresh D-drive aggregate output')
    rows, audit_rows, input_files = [], [], []
    total_actions, replays, identities = 0, 0, 0
    for directory in args.batches:
        analysis = load(directory / 'analysis.json')
        if analysis['max_clock_error'] != 0:
            raise ValueError('Batch clock audit failed')
        total_actions += analysis['actions_audited']
        replays += analysis['exact_physics_replays']
        identities += analysis['identity_cases']
        audit_rows.extend(analysis['paired_cases'])
        input_files.extend([directory / 'protocol.json', directory / 'analysis.json', directory / 'source_integrity.json'])
        if not all(load(directory / 'source_integrity.json')[field] for field in ('unchanged', 'identity_exact')):
            raise ValueError('Source integrity or identity failed')
        rows.extend(load(path) for path in sorted((directory / 'records').glob('*.json')))
    expected = {(profile, count, mode) for profile in ('random', 'boundary', 'clustered', 'near_origin', 'max_radius', 'adversarial_heading')
                for count in (10, 13, 16) for mode in ('previous', 'identity', 'action_phase')}
    keys = {(row['profile'], row['count'], row['mode']) for row in rows}
    if keys != expected or len(rows) != len(expected) or not all(row['success'] for row in rows):
        raise ValueError('Aggregate has missing, repeated or failed registered tasks')
    summary = runner.summarize(rows, ('previous', 'identity', 'action_phase'))
    improvement = 100 * summary['action_phase']['saved_seconds_per_source'] / summary['previous']['mean']
    selected = improvement >= 1 and summary['action_phase']['worst_regression_percent'] <= 10
    components = {}
    for mode in summary:
        chosen = [row for row in audit_rows if row['mode'] == mode]
        components[mode] = {key: statistics.mean(row[key] for row in chosen) for key in ('movement', 'radio', 'optical')}
        if not math.isclose(sum(components[mode].values()), summary[mode]['mean'], abs_tol=1e-9):
            raise ValueError('Aggregate cost components do not sum to complete time')
    count_summary = {count: runner.summarize([row for row in rows if row['count'] == count], ('previous', 'identity', 'action_phase'))
                     for count in (10, 13, 16)}
    change_rows = [row for row in audit_rows if row['mode'] == 'action_phase' and row['phase'] != 0]
    result = {'runs': len(rows), 'successes': len(rows), 'cases_per_mode': 18, 'actions_audited': total_actions,
              'exact_physics_replays': replays, 'identity_cases': identities, 'summary': summary,
              'improvement_percent': improvement, 'independent_validation_gate_passed': selected,
              'decision': 'eligible_for_separately_registered_holdout' if selected else 'reject_keep_frozen_v11',
              'cost_components': components, 'by_count': count_summary, 'changed_cases': change_rows,
              'new_holdout': False, 'official_calls': 0, 'formal_calls': 0,
              'input_hashes': {str(path.resolve().relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in [*input_files, Path(__file__), HERE / 'PROTOCOL.md']},
              'frozen_manifests': runner.frozen_integrity()}
    runner.experiment.save(output, result)
    print(json.dumps({key: value for key, value in result.items() if key not in ('input_hashes', 'by_count', 'changed_cases')}, indent=2))


if __name__ == '__main__':
    main()
