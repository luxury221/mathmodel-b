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
sys.path[:0] = [str(HERE), str(ROOT / 'research/q3_scan_economy_v13'), str(ROOT / 'research/radial_patrol_v7'), str(ROOT / 'research/joint_search_v5')]
import run_scan_benchmark
import experiment_v5 as experiment
from gate_policy import MODES, VerifiedGatePolicy
from scan_policy import ScanEconomyPolicy


def hashes():
    return {**run_scan_benchmark.source_hashes(), **{str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in HERE.glob('*.py')}}


def evaluate(scene, mode, world_factory=None):
    def factory(port, problem, stations, variant, network, mode):
        arguments = (port, problem, stations, variant, network)
        return ScanEconomyPolicy(*arguments, mode='station_only') if mode == 'previous' else VerifiedGatePolicy(*arguments, mode=mode)
    options = {} if world_factory is None else {'world_factory': world_factory}
    return experiment.evaluate(scene, mode, policy_factory=factory, **options)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--counts', nargs='+', type=int, default=[13])
    parser.add_argument('--modes', nargs='+', choices=('previous', *MODES), default=['previous', 'gate130', 'gate300', 'gate600'])
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a new D-drive output')
    frozen = hashes()
    experiment.save(output / 'protocol.json', {'created_local': datetime.now().isoformat(), 'source_hashes': frozen,
                                              'modes': args.modes, 'official_calls': 0, 'verified_original_target_radius': 2000,
                                              'metric': 'case-equal complete mission virtual seconds/source; all sources cleared'})
    for name in ('PROTOCOL.md', 'CORRECTION.md'):
        shutil.copy2(HERE / name, output / name)
    for relative in frozen:
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
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
            print(scene['profile'], scene['count'], mode,
                  round(row['seconds_per_source'], 3) if row['success'] else row['failure'], flush=True)
    summary = {mode: {'runs': len(selected), 'successes': sum(row['success'] for row in selected),
                      'mean': statistics.mean(row['seconds_per_source'] for row in selected) if all(row['success'] for row in selected) else None}
               for mode in args.modes for selected in [[row for row in rows if row['mode'] == mode]]}
    experiment.save(output / 'summary.json', summary)
    experiment.save(output / 'source_integrity.json', {'unchanged': frozen == hashes()})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
