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
from network_policy import NetworkPolicy, NETWORK_MODES
from radial_policy import RadialPolicy


def factory(port, problem, stations, variant, network, mode):
    if mode == 'previous':
        return RadialPolicy(port, problem, stations, variant, network, mode='pilot1600')
    return NetworkPolicy(port, problem, stations, variant, network, mode=mode)


def hashes():
    folders = (HERE, ROOT / 'research/radial_patrol_v7')
    return {**experiment.source_hashes(), **{str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                            for folder in folders for path in folder.glob('*.py')}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--modes', nargs='+', default=['previous', *NETWORK_MODES])
    parser.add_argument('--counts', nargs='+', type=int, default=[10, 13, 16])
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a new D-drive output')
    scenes = [scene for scene in experiment.make_scenes(103100000, 'development', args.counts) if scene['problem'] == 'q3']
    frozen = hashes()
    experiment.save(output / 'protocol.json', {'created_local': datetime.now().isoformat(), 'source_hashes': frozen,
                                               'modes': args.modes, 'official_calls': 0, 'baseline': 'Q3 frozen pilot1600',
                                               'metric': 'case-equal total virtual seconds/source; all failures retained'})
    experiment.save(output / 'scenes_evaluator_only.json', scenes)
    for relative in frozen:
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    records = []
    for scene in scenes:
        for mode in args.modes:
            record, trace = experiment.evaluate(scene, mode, policy_factory=factory)
            records.append(record)
            name = scene['id'] + '__' + mode + '.json'
            experiment.save(output / 'records' / name, record)
            experiment.save(output / 'traces' / name, trace)
            print(scene['profile'], scene['count'], mode, record['failure'] if not record['success'] else round(record['seconds_per_source'], 3), flush=True)
    summary = {}
    for mode in args.modes:
        rows = [record for record in records if record['mode'] == mode]
        summary[mode] = {'runs': len(rows), 'successes': sum(record['success'] for record in rows),
                         'mean': statistics.mean(record['seconds_per_source'] for record in rows) if all(record['success'] for record in rows) else None}
    experiment.save(output / 'summary.json', summary)
    experiment.save(output / 'source_integrity.json', {'unchanged': frozen == hashes()})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
