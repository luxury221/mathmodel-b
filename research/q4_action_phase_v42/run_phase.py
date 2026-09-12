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
sys.path[:0] = [str(HERE), str(ROOT / 'research/q3_action_scheduler_v41'), str(ROOT / 'research/shaped_probe_v11')]
import run_action as baseline
from phase_policy import ActionPhasePolicy, MODES
from shaped_policy import ShapedProbePolicy

experiment = baseline.experiment
MANIFESTS = (*baseline.MANIFESTS, 'reports/feasible_segment_bound_v40/bound_20260912_070654/protocol.json',
             'reports/q3_action_scheduler_v41/dev1_20260912_071723/protocol.json')


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
    return {**baseline.hashes(), **{str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in [*HERE.glob('*.py'), HERE / 'PROTOCOL.md']}}


def evaluate(scene, mode):
    holder = []
    def factory(*arguments, mode):
        policy = ShapedProbePolicy(*arguments, mode='shaped_cost') if mode == 'previous' else ActionPhasePolicy(*arguments, mode=mode)
        holder.append(policy)
        return policy
    row, trace = experiment.evaluate(scene, mode, policy_factory=factory)
    row['phase_events'] = getattr(holder[0], 'phase_events', [])
    row['timing_audit'] = baseline.baseline.audit_trace(trace, row['virtual_seconds'])
    return row, trace


def summarize(rows, modes):
    reference = {row['scene_id']: row for row in rows if row['mode'] == 'previous'}
    summary = {}
    for mode in modes:
        selected = [row for row in rows if row['mode'] == mode]
        all_clear = all(row['success'] for row in selected)
        mean = statistics.mean(row['seconds_per_source'] for row in selected) if all_clear else None
        saving = statistics.mean(reference[row['scene_id']]['seconds_per_source'] - row['seconds_per_source']
                                 for row in selected) if all_clear else None
        worst = max(100 * (row['seconds_per_source'] / reference[row['scene_id']]['seconds_per_source'] - 1)
                    for row in selected) if all_clear else None
        rotations = sum(row['policy_stats'].get('phase_actual_rotations', 0) for row in selected)
        summary[mode] = {'runs': len(selected), 'successes': sum(row['success'] for row in selected), 'mean': mean,
                         'saved_seconds_per_source': saving, 'worst_regression_percent': worst,
                         'actual_rotations': rotations,
                         'expansion_gate_passed': all_clear and rotations > 0 and saving >= 1 and worst <= 10}
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--counts', type=int, nargs='+', default=[13])
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists() or not set(args.counts) <= {10, 13, 16}:
        raise ValueError('Use a fresh D-drive output and registered development counts')
    modes = ('previous', *MODES)
    frozen = hashes()
    experiment.save(output / 'protocol.json', {'created_local': datetime.now().isoformat(), 'source_hashes': frozen,
                                              'modes': modes, 'official_calls': 0, 'formal_calls': 0,
                                              'metric': 'case-equal complete mission seconds/source; offline development only'})
    for relative in frozen:
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    shutil.copy2(HERE / 'PROTOCOL.md', output / 'PROTOCOL.md')
    scenes = [scene for scene in experiment.make_scenes(103100000, 'development', args.counts) if scene['problem'] == 'q4']
    experiment.save(output / 'scenes_evaluator_only.json', scenes)
    rows = []
    for scene in scenes:
        for mode in modes:
            row, trace = evaluate(scene, mode)
            rows.append(row)
            name = scene['id'] + '__' + mode + '.json'
            experiment.save(output / 'records' / name, row)
            experiment.save(output / 'traces' / name, trace)
            print(scene['profile'], scene['count'], mode, round(row['seconds_per_source'], 3) if row['success'] else row['failure'],
                  'phase', row['policy_stats'].get('phase_degrees', 0), flush=True)
    reference = {row['scene_id']: row for row in rows if row['mode'] == 'previous'}
    identity = all(row['success'] and row['trace_sha256'] == reference[row['scene_id']]['trace_sha256']
                   for row in rows if row['mode'] == 'identity')
    integrity = {'unchanged': frozen == hashes(), 'identity_exact': identity, 'frozen_manifests': frozen_integrity()}
    summary = summarize(rows, modes)
    experiment.save(output / 'summary.json', summary)
    experiment.save(output / 'source_integrity.json', integrity)
    print(json.dumps(summary, indent=2), flush=True)
    if not all(row['success'] for row in rows) or not integrity['unchanged'] or not identity:
        raise ValueError('Development correctness or integrity gate failed')


if __name__ == '__main__':
    main()
