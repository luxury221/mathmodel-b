from __future__ import annotations

import argparse
import hashlib
import shutil
import statistics
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / 'research/joint_motion_v16')]
import run_motion
from commit_policy import CommittedPatrolPolicy
from shaped_policy import ShapedProbePolicy

experiment = run_motion.experiment


def hashes():
    return {**run_motion.hashes(), **{str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in HERE.glob('*.py')}}


def evaluate(scene, mode, world_factory=None):
    holder = []
    def factory(port, problem, stations, variant, network, mode):
        arguments = (port, problem, stations, variant, network)
        policy = ShapedProbePolicy(*arguments, mode='shaped_cost') if mode == 'previous' else CommittedPatrolPolicy(*arguments, mode=mode)
        holder.append(policy)
        return policy
    options = {} if world_factory is None else {'world_factory': world_factory}
    row, trace = experiment.evaluate(scene, mode, policy_factory=factory, **options)
    row['commit_events'] = getattr(holder[0], 'commit_events', [])
    return row, trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--counts', type=int, nargs='+', default=[13])
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a new D-drive directory')
    frozen = hashes()
    scenes = [scene for scene in experiment.make_scenes(103100000, 'development', args.counts) if scene['problem'] == 'q4']
    experiment.save(output / 'protocol.json', {'source_hashes': frozen, 'modes': ['previous', 'committed'],
                                              'created_local': datetime.now().isoformat(), 'official_calls': 0})
    experiment.save(output / 'scenes_evaluator_only.json', scenes)
    shutil.copy2(HERE / 'PROTOCOL.md', output / 'PROTOCOL.md')
    for relative in frozen:
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    records = []
    for scene in scenes:
        for mode in ('previous', 'committed'):
            row, trace = evaluate(scene, mode)
            records.append(row)
            name = scene['id'] + '__' + mode + '.json'
            experiment.save(output / 'records' / name, row)
            experiment.save(output / 'traces' / name, trace)
            print(scene['profile'], scene['count'], mode,
                  round(row['seconds_per_source'], 3) if row['success'] else row['failure'], flush=True)
    summary = {mode: {'runs': len(rows), 'successes': sum(row['success'] for row in rows),
                       'mean': statistics.mean(row['seconds_per_source'] for row in rows) if all(row['success'] for row in rows) else None}
               for mode in ('previous', 'committed') for rows in [[row for row in records if row['mode'] == mode]]}
    experiment.save(output / 'summary.json', summary)
    experiment.save(output / 'source_integrity.json', {'unchanged': frozen == hashes()})
    print(summary, flush=True)


if __name__ == '__main__':
    main()
