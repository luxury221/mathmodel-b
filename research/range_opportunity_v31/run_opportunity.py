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
                str(ROOT / 'research/shaped_probe_v11'), str(ROOT / 'research/joint_search_v5')]
import run_scan_benchmark
import experiment_v5 as experiment
from opportunity_policy import MODES, RangeOpportunityQ3, RangeOpportunityQ4
from scan_policy import ScanEconomyPolicy
from shaped_policy import ShapedProbePolicy


def hashes():
    return {**run_scan_benchmark.source_hashes(), **{str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in HERE.glob('*.py')}}


def evaluate(scene, mode, world_factory=None):
    holder = []
    def factory(port, problem, stations, variant, network, mode):
        arguments = (port, problem, stations, variant, network)
        if mode == 'previous':
            policy = ScanEconomyPolicy(*arguments, mode='station_only') if problem == 'q3' else ShapedProbePolicy(*arguments, mode='shaped_cost')
        else:
            policy = (RangeOpportunityQ3 if problem == 'q3' else RangeOpportunityQ4)(*arguments, mode=mode)
        holder.append(policy)
        return policy
    options = {} if world_factory is None else {'world_factory': world_factory}
    row, trace = experiment.evaluate(scene, mode, policy_factory=factory, **options)
    row['extended_events'] = getattr(holder[0], 'extended_events', [])
    row['effective_target_radius'] = holder[0].options.get('target_radius')
    return row, trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--counts', type=int, nargs='+', default=[13])
    parser.add_argument('--problems', nargs='+', choices=('q3', 'q4'), default=['q3', 'q4'])
    parser.add_argument('--modes', nargs='+', choices=('previous', *MODES), default=['previous', *MODES])
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists() or 'previous' not in args.modes:
        raise ValueError('Use a new D-drive output and retain the paired reference')
    frozen = hashes()
    experiment.save(output / 'protocol.json', {'created_local': datetime.now().isoformat(), 'source_hashes': frozen,
                                              'modes': args.modes, 'problems': args.problems, 'official_calls': 0,
                                              'metric': 'case-equal complete mission seconds/source; offline screening only'})
    shutil.copy2(HERE / 'PROTOCOL.md', output / 'PROTOCOL.md')
    for relative in frozen:
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    scenes = [scene for scene in experiment.make_scenes(103100000, 'development', args.counts) if scene['problem'] in args.problems]
    experiment.save(output / 'scenes_evaluator_only.json', scenes)
    rows = []
    for scene in scenes:
        for mode in args.modes:
            row, trace = evaluate(scene, mode)
            rows.append(row)
            name = scene['id'] + '__' + mode + '.json'
            experiment.save(output / 'records' / name, row)
            experiment.save(output / 'traces' / name, trace)
            print(scene['problem'], scene['profile'], scene['count'], mode,
                  round(row['seconds_per_source'], 3) if row['success'] else row['failure'],
                  'extra_radio', row['policy_stats'].get('extended_actions', 0), flush=True)
    reference = {row['scene_id']: row for row in rows if row['mode'] == 'previous'}
    identity_passed = all(row['success'] and row['trace_sha256'] == reference[row['scene_id']]['trace_sha256'] for row in rows if row['mode'] == 'identity')
    integrity = {'unchanged': frozen == hashes(), 'identity_passed': identity_passed}
    summary = {problem: {mode: {'runs': len(selected), 'successes': sum(row['success'] for row in selected),
                               'mean': statistics.mean(row['seconds_per_source'] for row in selected) if all(integrity.values()) and all(row['success'] for row in selected) else None}
                        for mode in args.modes for selected in [[row for row in rows if row['problem'] == problem and row['mode'] == mode]]}
               for problem in args.problems}
    experiment.save(output / 'summary.json', summary)
    experiment.save(output / 'source_integrity.json', integrity)
    print(json.dumps(summary, indent=2), flush=True)
    if not all(integrity.values()):
        raise ValueError('Source or identity check failed')


if __name__ == '__main__':
    main()
