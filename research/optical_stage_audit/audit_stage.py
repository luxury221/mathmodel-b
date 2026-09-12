from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import xml.etree.ElementTree as element_tree
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'research/motion_stage_audit'), str(ROOT / 'research/gated_repack_v23'),
                str(ROOT / 'research/selective_optical_v26'), str(ROOT / 'research/radial_optical_v28'),
                str(ROOT / 'research/practice_baseline'), str(ROOT / 'src')]
from audit_motion import audit_directory, comparisons
from baseline import verify_hashes
from b2026_candidate.validation import require_validation
import run_gated
import run_study
import run_radial_optical


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def check_manifest(folder, manifest, allowed_changes=()):
    changed, archived = [], []
    for relative, expected in manifest.items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            changed.append(relative)
        if hashlib.sha256((folder / 'source_snapshot' / relative).read_bytes()).hexdigest() != expected:
            archived.append(relative)
    normalized = {value.replace('\\', '/') for value in changed}
    if archived or normalized != set(allowed_changes):
        raise ValueError('Source archive mismatch: ' + str(folder))
    return {'folder': str(folder), 'source_files': len(manifest), 'current_changes': changed, 'archived_changes': archived}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--regression', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a new D-drive audit directory')
    directories = {'v23_q3': ROOT / 'reports/gated_repack_v23/dev1_20260912_020758',
                   'v26_q4': ROOT / 'reports/selective_optical_v26/dev1_20260912_023432',
                   'v28_q4': ROOT / 'reports/radial_optical_v28/dev1_20260912_025127'}
    audits, rows_by_version = {}, {}
    for version, directory in directories.items():
        audit, rows = audit_directory(directory)
        expected = len(load(directory / 'scenes_evaluator_only.json')) * len(load(directory / 'protocol.json')['modes'])
        if (audit['runs'] != expected or audit['successes'] != expected or audit['current_changes']
                or audit['archived_changes'] or audit['tiny_distinct_remeasurements'] or audit['max_clock_error'] != 0):
            raise ValueError('Full-task audit failed: ' + version)
        audits[version] = audit
        rows_by_version[version] = rows
    regressions = []
    for path in sorted(args.regression.glob('*.xml')):
        root = element_tree.parse(path).getroot()
        suites = [root] if root.tag == 'testsuite' else list(root.iter('testsuite'))
        entry = {'path': str(path.resolve()), **{key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
                                                for key in ('tests', 'errors', 'failures', 'skipped')}}
        if entry['errors'] or entry['failures']:
            raise ValueError('Regression failures: ' + str(path))
        regressions.append(entry)
    if len(regressions) != 7:
        raise ValueError('Expected seven isolated regression modules')
    geometry = ROOT / 'reports/annular_topology_v25/geometry_20260912_022429'
    bound = ROOT / 'reports/patrol_cost_bound_v24/diagnostic_20260912_021849'
    rejected = ROOT / 'reports/anchored_optical_v27/geometry_rejected_20260912_024350'
    auxiliary = [check_manifest(folder, load(folder / 'protocol.json')['source_hashes']) for folder in (geometry, bound)]
    auxiliary.append(check_manifest(rejected, load(rejected / 'source_hashes.json'),
                                    ('research/anchored_optical_v27/test_anchored.py',)))
    frozen_path = ROOT / 'reports/scan_validation_v13/validation_20260911_235435/selection.json'
    frozen = load(frozen_path)['source_hashes']
    changed = [relative for relative, expected in frozen.items() if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected]
    if changed:
        raise ValueError('Permanent frozen dependencies changed')
    verify_hashes()
    client_validation = require_validation()
    replay_specs = [('v23_q3', 'identity', 'boundary'), ('v26_q4', 'sentinel', 'adversarial_heading'),
                    ('v28_q4', 'sentinel', 'random'), ('v28_q4', 'sentinel', 'clustered')]
    replays = []
    for version, mode, profile in replay_specs:
        directory = directories[version]
        scene = next(scene for scene in load(directory / 'scenes_evaluator_only.json') if scene['profile'] == profile)
        reference = next(row for row in rows_by_version[version] if row['scene_id'] == scene['id'] and row['mode'] == mode)
        if version == 'v23_q3':
            row, trace = run_gated.evaluate(scene, mode)
        else:
            stations = load(directory / 'network.json')['selected']['stations']
            evaluate = run_study.evaluate if version == 'v26_q4' else run_radial_optical.evaluate
            row, trace = evaluate(scene, mode, stations)
        passed = row['success'] and row['trace_sha256'] == reference['trace_sha256'] and row['virtual_seconds'] == reference['virtual_seconds']
        save(output / 'replay_records' / f'{version}_{profile}.json', row)
        save(output / 'replay_traces' / f'{version}_{profile}.json', trace)
        replays.append({'version': version, 'mode': mode, 'profile': profile, 'passed': passed,
                        'trace_sha256': row['trace_sha256'], 'seconds_per_source': row['seconds_per_source']})
        print('replay', version, profile, passed, flush=True)
        if not passed:
            raise ValueError('Deterministic replay mismatch')
    previous_v26 = {row['scene_id']: row['trace_sha256'] for row in rows_by_version['v26_q4'] if row['mode'] == 'previous'}
    previous_v28 = {row['scene_id']: row['trace_sha256'] for row in rows_by_version['v28_q4'] if row['mode'] == 'previous'}
    if previous_v26 != previous_v28:
        raise ValueError('Repeated frozen Q4 control traces differ')
    result = {'official_calls': 0, 'new_holdout_used': False, 'goal_achieved': False,
              'runs': sum(audit['runs'] for audit in audits.values()), 'successes': sum(audit['successes'] for audit in audits.values()),
              'q4_new_main_runs': audits['v26_q4']['runs'] + audits['v28_q4']['runs'],
              'actions_audited': sum(audit['actions'] for audit in audits.values()),
              'development_audits': audits, 'comparisons': {version: comparisons(rows) for version, rows in rows_by_version.items()},
              'regression_tests_passed': sum(entry['tests'] - entry['skipped'] for entry in regressions),
              'regressions': regressions, 'deterministic_replays': replays,
              'q4_baseline_reproduced_across_batches': True, 'auxiliary_source_audits': auxiliary,
              'v27_initial_fixture_errors_preserved': load(rejected / 'initial_test_failure.json'),
              'frozen_files': len(frozen), 'frozen_changes': changed,
              'actual_client_validation': str(client_validation), 'actual_client_unchanged': True}
    save(output / 'analysis.json', result)
    verification = {'passed': True, 'frozen_hashes_unchanged_at_end': all(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest for relative, digest in frozen.items()),
                    'main_stage_source_unchanged_at_end': all(not audit_directory(directory)[0]['current_changes'] for directory in directories.values()),
                    'official_calls': 0, 'new_holdout_used': False, 'goal_achieved': False}
    if not all(verification[key] for key in ('frozen_hashes_unchanged_at_end', 'main_stage_source_unchanged_at_end')):
        raise ValueError('Final verification failed')
    save(output / 'final_verification.json', verification)
    print(json.dumps({key: result[key] for key in ('runs', 'successes', 'q4_new_main_runs', 'actions_audited', 'regression_tests_passed', 'comparisons')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
