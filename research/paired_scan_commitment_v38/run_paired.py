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
sys.path[:0] = [str(HERE), str(ROOT / 'research/q3_scan_economy_v13'),
                str(ROOT / 'research/radial_patrol_v7'), str(ROOT / 'research/shaped_probe_v11'),
                str(ROOT / 'research/joint_search_v5'), str(ROOT / 'research/joint_motion_v16')]
import run_scan_benchmark
import experiment_v5 as experiment
from paired_policy import MODES, PairedScanPolicy
from shaped_policy import ShapedProbePolicy


MANIFESTS = ('reports/scan_validation_v13/validation_20260911_235435/selection.json',
             'reports/frozen_practice_v33/batch_20260912_040922/protocol.json')


def frozen_integrity():
    result = {}
    for relative in MANIFESTS:
        frozen = json.loads((ROOT / relative).read_text(encoding='utf-8'))['source_hashes']
        changed = [name for name, digest in frozen.items()
                   if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
        if changed:
            raise ValueError('Frozen sources changed: ' + repr(changed))
        result[relative] = len(frozen)
    return result


def hashes():
    frozen_integrity()
    sources = [*HERE.glob('*.py'), HERE / 'PROTOCOL.md', ROOT / 'research/joint_motion_v16/motion_policy.py']
    return {**run_scan_benchmark.source_hashes(),
            **{str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}}


def evaluate(scene, mode):
    holder = []
    def factory(port, problem, stations, variant, network, mode):
        arguments = (port, problem, stations, variant, network)
        policy = ShapedProbePolicy(*arguments, mode='shaped_cost') if mode == 'previous' else PairedScanPolicy(*arguments, mode=mode)
        holder.append(policy)
        return policy
    row, trace = experiment.evaluate(scene, mode, policy_factory=factory)
    row['commitment_events'] = getattr(holder[0], 'commitment_events', [])
    row['pair_decisions'] = getattr(getattr(holder[0], 'pair_planner', None), 'decisions', [])
    row['timing_audit'] = run_scan_benchmark.audit_trace(trace, row['virtual_seconds'])
    return row, trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--counts', type=int, nargs='+', default=[13])
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists() or not set(args.counts) <= {10, 13, 16}:
        raise ValueError('Use a fresh D-drive output and registered development counts')
    frozen = hashes()
    modes = ('previous', *MODES)
    experiment.save(output / 'protocol.json', {'created_local': datetime.now().isoformat(), 'source_hashes': frozen,
                                              'modes': modes, 'official_calls': 0, 'formal_calls': 0,
                                              'metric': 'case-equal complete mission seconds/source; offline development only'})
    shutil.copy2(HERE / 'PROTOCOL.md', output / 'PROTOCOL.md')
    for relative in frozen:
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
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
            print(scene['profile'], scene['count'], mode,
                  round(row['seconds_per_source'], 3) if row['success'] else row['failure'],
                  'paired_deletions', row['policy_stats'].get('pair_station_deletions', 0),
                  'witness_passes', sum(event['witness_joint_passes'] for event in row['pair_decisions']), flush=True)
    reference = {row['scene_id']: row for row in rows if row['mode'] == 'previous'}
    identity_passed = all(row['success'] and row['trace_sha256'] == reference[row['scene_id']]['trace_sha256']
                          for row in rows if row['mode'] == 'identity')
    integrity = {'unchanged': frozen == hashes(), 'identity_passed': identity_passed}
    summary = {}
    for mode in modes:
        selected = [row for row in rows if row['mode'] == mode]
        valid = all(integrity.values()) and all(row['success'] for row in selected)
        entry = {'runs': len(selected), 'successes': sum(row['success'] for row in selected),
                 'mean': statistics.mean(row['seconds_per_source'] for row in selected) if valid else None,
                 'pair_commitments': sum(row['policy_stats'].get('pair_commitments', 0) for row in selected),
                 'pair_station_deletions': sum(row['policy_stats'].get('pair_station_deletions', 0) for row in selected),
                 'decision_counts': {key: sum(event[key] for row in selected for event in row['pair_decisions'])
                                     for key in ('pairs', 'witness_joint_passes', 'price_passes', 'continuous_checks', 'continuous_complete')}}
        if valid:
            entry['saved_seconds_per_source'] = statistics.mean(reference[row['scene_id']]['seconds_per_source'] - row['seconds_per_source'] for row in selected)
            entry['worst_regression_percent'] = max(100 * (row['seconds_per_source'] / reference[row['scene_id']]['seconds_per_source'] - 1) for row in selected)
            entry['expansion_gate_passed'] = (entry['saved_seconds_per_source'] >= 1 and entry['worst_regression_percent'] <= 10
                                               and entry['pair_station_deletions'] > 0)
        summary[mode] = entry
    experiment.save(output / 'summary.json', summary)
    experiment.save(output / 'source_integrity.json', {**integrity, 'frozen_manifests': frozen_integrity()})
    print(json.dumps(summary, indent=2), flush=True)
    if not all(integrity.values()):
        raise ValueError('Source or identity check failed')


if __name__ == '__main__':
    main()
