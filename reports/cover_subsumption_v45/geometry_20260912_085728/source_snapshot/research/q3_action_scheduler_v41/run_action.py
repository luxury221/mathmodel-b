from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import sys
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / 'research/q3_scan_economy_v13'), str(ROOT / 'research/radial_patrol_v7'),
                str(ROOT / 'research/joint_search_v5')]
import run_scan_benchmark as baseline
from action_policy import ActionSchedulerPolicy, MODES
from scan_policy import ScanEconomyPolicy

experiment = baseline.experiment
MANIFESTS = ('reports/scan_validation_v13/validation_20260911_235435/selection.json',
             'reports/frozen_practice_v33/batch_20260912_040922/protocol.json',
             'reports/paired_scan_commitment_v38/dev1_20260912_060916/protocol.json',
             'reports/coverage_probe_v39/dev1_20260912_064358/protocol.json')


def frozen_integrity():
    counts = {}
    for relative in MANIFESTS:
        frozen = json.loads((ROOT / relative).read_text(encoding='utf-8'))['source_hashes']
        for name, digest in frozen.items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
                raise ValueError('Frozen source changed: ' + name)
        counts[relative] = len(frozen)
    return counts


def hashes():
    frozen_integrity()
    frozen = json.loads((ROOT / MANIFESTS[0]).read_text(encoding='utf-8'))['source_hashes']
    return {**frozen, **{str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in [*HERE.glob('*.py'), HERE / 'PROTOCOL.md']}}


def audit_predictions(events, trace):
    count = 0
    for event in events:
        if event['predicted_position'] is None:
            continue
        actions = [action for action in trace if event['start_seconds'] < action['virtual_seconds'] <= event['end_seconds']]
        if not actions or actions[0]['channel'] != event['channel'] or actions[0]['position'] != event['predicted_position']:
            raise ValueError('Predicted first physical action was not executed')
        expected = 'measure' if event['predicted_kind'] == 'probe' else 'clear'
        if actions[0]['action'] != expected:
            raise ValueError('First action kind differs from its prediction')
        count += 1
    return count


def evaluate(scene, mode):
    holder = []
    def factory(*arguments, mode):
        policy = ScanEconomyPolicy(*arguments, mode='station_only') if mode == 'previous' else ActionSchedulerPolicy(*arguments, mode=mode)
        holder.append(policy)
        return policy
    row, trace = experiment.evaluate(scene, mode, policy_factory=factory)
    row['action_events'] = getattr(holder[0], 'action_events', [])
    row['effective_target_radius'] = holder[0].options['target_radius']
    row['timing_audit'] = baseline.audit_trace(trace, row['virtual_seconds'])
    if row['success']:
        try:
            row['prediction_checks'] = audit_predictions(row['action_events'], trace)
        except ValueError as error:
            row['success'] = False
            row['seconds_per_source'] = None
            row['failure'] = str(error)
    return row, trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--counts', type=int, nargs='+', default=[13])
    parser.add_argument('--modes', nargs='+', choices=('previous', *MODES), default=['previous', *MODES])
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists() or not set(args.counts) <= {10, 13, 16}:
        raise ValueError('Use a fresh D-drive output and registered counts')
    if not {'previous', 'identity'} <= set(args.modes):
        raise ValueError('Retain original and locked-executor identity controls')
    frozen = hashes()
    experiment.save(output / 'protocol.json', {'created_local': datetime.now().isoformat(), 'source_hashes': frozen,
                                              'modes': args.modes, 'official_calls': 0,
                                              'metric': 'case-equal complete mission seconds/source; offline development only'})
    for relative in frozen:
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    shutil.copy2(HERE / 'PROTOCOL.md', output / 'PROTOCOL.md')
    scenes = [scene for scene in experiment.make_scenes(103100000, 'development', args.counts) if scene['problem'] == 'q3']
    experiment.save(output / 'scenes_evaluator_only.json', scenes)
    rows = []
    for scene in scenes:
        for mode in args.modes:
            row, trace = evaluate(scene, mode)
            rows.append(row)
            name = scene['id'] + '__' + mode + '.json'
            experiment.save(output / 'records' / name, row)
            experiment.save(output / 'traces' / name, trace)
            print(scene['profile'], scene['count'], mode, round(row['seconds_per_source'], 3) if row['success'] else row['failure'],
                  'source_selections', row['policy_stats']['joint_targets'], flush=True)
    reference = {row['scene_id']: row for row in rows if row['mode'] == 'previous'}
    identity_passed = all(row['success'] and row['trace_sha256'] == reference[row['scene_id']]['trace_sha256']
                          for row in rows if row['mode'] == 'identity')
    integrity = {'unchanged': frozen == hashes(), 'identity_passed': identity_passed,
                 'correct_admission': all(row['effective_target_radius'] == 2000 for row in rows)}
    summary = {}
    for mode in args.modes:
        selected = [row for row in rows if row['mode'] == mode]
        valid = all(integrity.values()) and all(row['success'] for row in selected)
        entry = {'runs': len(selected), 'successes': sum(row['success'] for row in selected),
                 'mean': statistics.mean(row['seconds_per_source'] for row in selected) if valid else None}
        if valid:
            entry['saved_seconds_per_source'] = statistics.mean(reference[row['scene_id']]['seconds_per_source'] - row['seconds_per_source'] for row in selected)
            entry['worst_regression_percent'] = max(100 * (row['seconds_per_source'] / reference[row['scene_id']]['seconds_per_source'] - 1) for row in selected)
            entry['expansion_gate_passed'] = entry['saved_seconds_per_source'] >= 1 and entry['worst_regression_percent'] <= 10
        summary[mode] = entry
    experiment.save(output / 'summary.json', summary)
    experiment.save(output / 'source_integrity.json', {**integrity, 'frozen_manifests': frozen_integrity()})
    print(json.dumps(summary, indent=2), flush=True)
    if not all(integrity.values()):
        raise ValueError('Source, admission or identity check failed')


if __name__ == '__main__':
    main()
