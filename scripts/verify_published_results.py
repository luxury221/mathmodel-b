from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/2026-09-12'
BATCHES = {
    'q3': ROOT / 'reports/scan_validation_v13/validation_20260911_235435',
    'q4': ROOT / 'reports/shaped_probe_validation_v11/validation_20260911_223412',
}


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def equal(actual, expected, tolerance=1e-7):
    if not math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=tolerance):
        raise ValueError(f'Numeric mismatch: {actual} != {expected}')


def verify_official(rows, summary):
    if len(rows) != 10 or any(row['evidence_type'] != 'official_practice' for row in rows):
        raise ValueError('Official table must contain only the ten archived official practice cases')
    for row in rows:
        if row['all_cleared'] != 'True' or int(row['source_count']) <= 0:
            raise ValueError('Official case is not a complete successful mission')
        total = sum(float(row[key]) for key in ('movement_seconds', 'measurement_seconds', 'switch_seconds',
                                                'successful_clear_seconds', 'failed_clear_seconds'))
        residual = float(row['rounding_residual_seconds'])
        operation_count = int(row['measurement_count']) + int(row['source_count']) + float(row['failed_clear_seconds']) / 3
        rounding_bound = operation_count * 0.5e-6 + 1e-6
        if not math.isfinite(residual) or abs(residual) > rounding_bound:
            raise ValueError('Recorded rounding residual exceeds the per-action rounding bound')
        equal(total + residual, row['virtual_seconds'], tolerance=1e-8)
        equal(float(row['virtual_seconds']) / int(row['source_count']), row['seconds_per_source'], tolerance=1e-7)
    for problem in ('q3', 'q4'):
        selected = [row for row in rows if row['problem'] == problem]
        if len(selected) != 5:
            raise ValueError('Each problem must retain all five official cases')
        mean = statistics.mean(float(row['seconds_per_source']) for row in selected)
        equal(mean, summary['summaries'][problem]['mean_seconds_per_source'])
    return {'cases': len(rows), 'rounding_residuals_verified': len(rows),
            'maximum_absolute_rounding_residual_seconds': max(abs(float(row['rounding_residual_seconds'])) for row in rows),
            'scope': 'anonymized official numeric table only; raw responses are not published'}


def verify_offline(rows, summary, replay=False):
    if len(rows) != 128 or any(row['evidence_type'] != 'local_offline_simulation' for row in rows):
        raise ValueError('Offline table must remain separate and contain 128 archived paired runs')
    lookup = {(row['problem'], row['scene_id'], row['mode']): row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError('Duplicate offline rows')
    actions, replayed = 0, 0
    if replay:
        import run_offline_baselines
        import numpy as np
        from offline_benchmark import Source
    for problem, directory in BATCHES.items():
        scenes = {scene['id']: scene for scene in load(directory / 'scenes_evaluator_only.json')}
        records = sorted((directory / 'records').glob('*.json'))
        if len(scenes) != 32 or len(records) != 64:
            raise ValueError('Incomplete archived paired local validation')
        candidate = summary['summaries'][problem]['combined']['mode']
        modes = {mode: [] for mode in ('previous', candidate)}
        for path in records:
            row = load(path)
            exported = lookup[(problem, row['scene_id'], row['mode'])]
            if not row['success'] or row['cleared_count'] != row['count'] or exported['success'] != 'True':
                raise ValueError('Failed or incomplete offline run in a successful summary')
            equal(row['seconds_per_source'], exported['seconds_per_source'])
            equal(row['virtual_seconds'] / row['count'], row['seconds_per_source'])
            equal(sum(float(exported[key]) for key in ('movement_seconds', 'radio_seconds', 'optical_seconds')),
                  row['virtual_seconds'], tolerance=1e-6)
            trace = load(directory / 'traces' / path.name)
            digest = hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest()
            if digest != row['trace_sha256'] or digest != exported['trace_sha256']:
                raise ValueError('Offline trace digest differs')
            if replay:
                scene = scenes[row['scene_id']]
                if scene['error_mode'] not in run_offline_baselines.ERROR_MODES:
                    raise ValueError('Unsupported historical noise model')
                world = run_offline_baselines.EvaluationWorld([Source(**source) for source in scene['sources']], scene['seed'], scene['error_mode'])
                for action in trace:
                    getattr(world, action['action'])(np.asarray(action['position']), action['channel'])
                    if world.trace[-1] != action:
                        raise ValueError('Archived synthetic-world physical replay differs: ' + path.name)
                if world.remaining:
                    raise ValueError('Offline replay leaves uncleared sources')
                run_offline_baselines.baseline.audit_trace(trace, row['virtual_seconds'])
                replayed += 1
            actions += len(trace)
            modes[row['mode']].append(row['seconds_per_source'])
        for mode, numbers in modes.items():
            key = 'previous_mean' if mode == 'previous' else 'candidate_mean'
            if len(numbers) != 32:
                raise ValueError('Missing complete candidate or baseline cases')
            equal(statistics.mean(numbers), summary['summaries'][problem]['combined'][key])
    return {'runs': len(rows), 'cases_per_problem': 32, 'archived_actions': actions,
            'synthetic_physics_replays': replayed, 'official_calls': 0}


def verify_manifest():
    manifest = load(RESULTS / 'source_copy_manifest.json')
    for relative, expected in manifest['files'].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or '.git' in path.parts or not path.is_file():
            raise ValueError('Invalid source-copy path')
        content = path.read_bytes()
        if len(content) != expected['bytes'] or hashlib.sha256(content).hexdigest() != expected['sha256']:
            raise ValueError('Copied source or archived evidence differs: ' + relative)
    return len(manifest['files'])


def main():
    parser = argparse.ArgumentParser(description='Verify archived exports; never calls official services')
    parser.add_argument('--replay-offline', action='store_true')
    parser.add_argument('--output', type=Path)
    arguments = parser.parse_args()
    result = {
        'status': 'passed', 'exact_copied_files': verify_manifest(),
        'official_numeric_export': verify_official(read_csv(RESULTS / 'official_practice_cases.csv'),
                                                    load(RESULTS / 'official_practice_summary.json')),
        'offline_validation': verify_offline(read_csv(RESULTS / 'offline_validation_cases.csv'),
                                             load(RESULTS / 'offline_validation_summary.json'), arguments.replay_offline),
        'official_calls_in_this_verification': 0, 'formal_calls_in_this_verification': 0,
        'v48_tested_by_this_verification': False,
    }
    if arguments.output:
        output = arguments.output.resolve()
        if output.exists() or (output.drive and output.drive.upper() != 'D:'):
            raise ValueError('Use a fresh D-drive verification output')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
