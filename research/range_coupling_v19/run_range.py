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
sys.path[:0] = [str(HERE), str(ROOT / 'research/joint_search_v5'), str(ROOT / 'research/radial_patrol_v7'),
                str(ROOT / 'research/q3_scan_economy_v13'), str(ROOT / 'research/shaped_probe_v11')]
import experiment_v5 as experiment
from range_policy import DirectionalRangePolicy, OmniRangePolicy
from scan_policy import ScanEconomyPolicy
from shaped_policy import ShapedProbePolicy


def hashes():
    frozen = json.loads((ROOT / 'reports/scan_validation_v13/validation_20260911_235435/selection.json').read_text(encoding='utf-8'))['source_hashes']
    if any(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest for relative, digest in frozen.items()):
        raise ValueError('Frozen dependency changed')
    return {**frozen, **{str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in HERE.glob('*.py')}}


def evaluate(scene, mode, world_factory=None):
    holder = []
    def factory(port, problem, stations, variant, network, mode):
        arguments = (port, problem, stations, variant, network)
        if mode == 'previous':
            policy = ScanEconomyPolicy(*arguments, mode='station_only') if problem == 'q3' else ShapedProbePolicy(*arguments, mode='shaped_cost')
        else:
            policy = OmniRangePolicy(*arguments, mode=mode) if problem == 'q3' else DirectionalRangePolicy(*arguments, mode=mode)
        holder.append(policy)
        return policy
    options = {} if world_factory is None else {'world_factory': world_factory}
    row, trace = experiment.evaluate(scene, mode, policy_factory=factory, **options)
    row['range_events'] = getattr(holder[0], 'range_events', [])
    return row, trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--counts', type=int, nargs='+', default=[13])
    parser.add_argument('--problems', nargs='+', choices=('q3', 'q4'), default=['q3', 'q4'])
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a new D-drive directory')
    frozen = hashes()
    scenes = [scene for scene in experiment.make_scenes(103100000, 'development', args.counts) if scene['problem'] in args.problems]
    experiment.save(output / 'protocol.json', {'source_hashes': frozen, 'modes': ['previous', 'coupled'],
                                              'created_local': datetime.now().isoformat(), 'official_calls': 0})
    experiment.save(output / 'scenes_evaluator_only.json', scenes)
    shutil.copy2(HERE / 'PROTOCOL.md', output / 'PROTOCOL.md')
    for relative in frozen:
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    records = []
    for scene in scenes:
        for mode in ('previous', 'coupled'):
            row, trace = evaluate(scene, mode)
            records.append(row)
            name = scene['id'] + '__' + mode + '.json'
            experiment.save(output / 'records' / name, row)
            experiment.save(output / 'traces' / name, trace)
            print(scene['problem'], scene['profile'], scene['count'], mode,
                  round(row['seconds_per_source'], 3) if row['success'] else row['failure'],
                  'region_cuts', row['policy_stats'].get('range_region_cuts', 0), flush=True)
    summary = {}
    for problem in args.problems:
        summary[problem] = {}
        for mode in ('previous', 'coupled'):
            rows = [row for row in records if row['problem'] == problem and row['mode'] == mode]
            summary[problem][mode] = {'runs': len(rows), 'successes': sum(row['success'] for row in rows),
                                      'mean': statistics.mean(row['seconds_per_source'] for row in rows) if all(row['success'] for row in rows) else None}
    experiment.save(output / 'summary.json', summary)
    experiment.save(output / 'source_integrity.json', {'unchanged': frozen == hashes()})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
