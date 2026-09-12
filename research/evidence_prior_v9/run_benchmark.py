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
sys.path[:0] = [str(HERE), str(ROOT / 'research/joint_search_v5'), str(ROOT / 'research/radial_patrol_v7')]
import experiment_v5 as experiment
from prior_policy import PriorPolicy
from policy_v5 import ContinuousPolicy
from radial_policy import RadialPolicy


def hashes():
    folders = (HERE, ROOT / 'research/radial_patrol_v7')
    return {**experiment.source_hashes(), **{str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                            for folder in folders for path in folder.glob('*.py')}}


def evaluate(scene, mode):
    holder = []
    def factory(port, problem, stations, variant, network, mode):
        if mode == 'previous':
            policy = RadialPolicy(port, problem, stations, variant, network, mode='pilot1600') if problem == 'q3' else ContinuousPolicy(port, problem, stations, variant, network, mode='certified_fixed')
        elif mode == 'center':
            policy = ContinuousPolicy(port, problem, stations, variant, network, mode='certified_center')
        else:
            policy = PriorPolicy(port, problem, stations, variant, network, mode=mode)
        holder.append(policy)
        return policy
    record, trace = experiment.evaluate(scene, mode, policy_factory=factory)
    record['prior_events'] = getattr(holder[0], 'prior_events', [])
    return record, trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--counts', nargs='+', type=int, default=[10, 13, 16])
    parser.add_argument('--problems', nargs='+', default=['q3', 'q4'])
    parser.add_argument('--modes', nargs='+', default=['previous', 'center', 'prior_route', 'prior_trial'])
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a new D-drive output')
    frozen = hashes()
    scenes = [scene for scene in experiment.make_scenes(103100000, 'development', args.counts) if scene['problem'] in args.problems]
    experiment.save(output / 'protocol.json', {'created_local': datetime.now().isoformat(), 'source_hashes': frozen,
                                               'modes': args.modes, 'official_calls': 0, 'metric': 'case-equal virtual seconds/source'})
    experiment.save(output / 'scenes_evaluator_only.json', scenes)
    for relative in frozen:
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    records = []
    for scene in scenes:
        for mode in args.modes:
            row, trace = evaluate(scene, mode)
            records.append(row)
            name = scene['id'] + '__' + mode + '.json'
            experiment.save(output / 'records' / name, row)
            experiment.save(output / 'traces' / name, trace)
            print(scene['problem'], scene['profile'], scene['count'], mode,
                  row['failure'] if not row['success'] else round(row['seconds_per_source'], 3), flush=True)
    summary = {}
    for problem in args.problems:
        summary[problem] = {}
        for mode in args.modes:
            rows = [row for row in records if row['problem'] == problem and row['mode'] == mode]
            summary[problem][mode] = {'runs': len(rows), 'successes': sum(row['success'] for row in rows),
                                      'mean': statistics.mean(row['seconds_per_source'] for row in rows) if all(row['success'] for row in rows) else None}
    experiment.save(output / 'summary.json', summary)
    experiment.save(output / 'source_integrity.json', {'unchanged': frozen == hashes()})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
