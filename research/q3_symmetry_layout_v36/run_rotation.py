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
                str(ROOT / 'research/radial_patrol_v7'), str(ROOT / 'research/joint_search_v5')]
import run_scan_benchmark
import experiment_v5 as experiment
from rotation_policy import MODES, SymmetryLayoutPolicy
from scan_policy import ScanEconomyPolicy


def hashes():
    files = [*HERE.glob('*.py'), HERE / 'PROTOCOL.md']
    return {**run_scan_benchmark.source_hashes(),
            **{str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}


def evaluate(scene, mode):
    holder = []
    def factory(port, problem, stations, variant, network, mode):
        arguments = (port, problem, stations, variant, network)
        policy = ScanEconomyPolicy(*arguments, mode='station_only') if mode == 'previous' else SymmetryLayoutPolicy(*arguments, mode=mode)
        holder.append(policy)
        return policy
    row, trace = experiment.evaluate(scene, mode, policy_factory=factory)
    row['rotation_events'] = getattr(holder[0], 'rotation_events', [])
    return row, trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--counts', type=int, nargs='+', default=[13])
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a fresh D-drive output')
    modes = ('previous', *MODES)
    frozen = hashes()
    experiment.save(output / 'protocol.json', {'created_local': datetime.now().isoformat(), 'source_hashes': frozen,
                                              'modes': modes, 'official_calls': 0,
                                              'metric': 'case-equal complete mission seconds/source; offline development only'})
    shutil.copy2(HERE / 'PROTOCOL.md', output / 'PROTOCOL.md')
    for relative in frozen:
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    scenes = [scene for scene in experiment.make_scenes(103100000, 'development', args.counts) if scene['problem'] == 'q3']
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
                  'phase', row['policy_stats'].get('rotation_degrees', 0), flush=True)
    reference = {row['scene_id']: row for row in rows if row['mode'] == 'previous'}
    identity = all(row['success'] and row['trace_sha256'] == reference[row['scene_id']]['trace_sha256']
                   for row in rows if row['mode'] == 'identity')
    integrity = {'unchanged': frozen == hashes(), 'identity_passed': identity}
    summary = {}
    for mode in modes:
        selected = [row for row in rows if row['mode'] == mode]
        valid = all(integrity.values()) and all(row['success'] for row in selected)
        entry = {'runs': len(selected), 'successes': sum(row['success'] for row in selected),
                 'mean': statistics.mean(row['seconds_per_source'] for row in selected) if valid else None}
        if valid:
            entry['saved_seconds_per_source'] = statistics.mean(reference[row['scene_id']]['seconds_per_source'] - row['seconds_per_source'] for row in selected)
            entry['worst_regression_percent'] = max(100 * (row['seconds_per_source'] / reference[row['scene_id']]['seconds_per_source'] - 1) for row in selected)
            entry['expansion_gate_passed'] = entry['saved_seconds_per_source'] >= 1 and entry['worst_regression_percent'] <= 10
        summary[mode] = entry
    experiment.save(output / 'source_integrity.json', integrity)
    experiment.save(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    if not all(integrity.values()):
        raise ValueError('Source or identity verification failed')


if __name__ == '__main__':
    main()
