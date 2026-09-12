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
sys.path[:0] = [str(HERE), str(ROOT / 'research/joint_search_v5')]
import experiment_v5 as previous
from policy_v6 import FlexiblePolicy


def hashes():
    return {**previous.source_hashes(), **{str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                         for path in HERE.glob('*.py')}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--modes', nargs='+', default=['previous', 'certified_fixed', 'flex'])
    parser.add_argument('--counts', nargs='+', type=int, default=[13])
    parser.add_argument('--problems', nargs='+', default=['q3', 'q4'])
    parser.add_argument('--base-seed', type=int, default=103100000)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a new D-drive directory')
    scenes = [scene for scene in previous.make_scenes(args.base_seed) if scene['count'] in args.counts and scene['problem'] in args.problems]
    source_hashes = hashes()
    previous.save(output / 'protocol.json', {'created_local': datetime.now().isoformat(), 'official_calls': 0,
                                            'source_hashes': source_hashes, 'modes': args.modes, 'metric': 'case-equal virtual seconds/source'})
    previous.save(output / 'scenes_evaluator_only.json', scenes)
    for relative in source_hashes:
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    records = []
    for scene in scenes:
        for mode in args.modes:
            factory = FlexiblePolicy if mode in ('flex', 'flex_nearest') else None
            row, trace = previous.evaluate(scene, mode, policy_factory=factory)
            records.append(row)
            name = scene['id'] + '__' + mode + '.json'
            previous.save(output / 'records' / name, row)
            previous.save(output / 'traces' / name, trace)
            print(scene['problem'], scene['profile'], scene['count'], mode, row['failure'] if not row['success'] else round(row['seconds_per_source'], 3), flush=True)
    summary = {}
    for problem in args.problems:
        summary[problem] = {}
        for mode in args.modes:
            rows = [row for row in records if row['problem'] == problem and row['mode'] == mode]
            summary[problem][mode] = {'runs': len(rows), 'successes': sum(row['success'] for row in rows),
                                      'mean': statistics.mean(row['seconds_per_source'] for row in rows) if all(row['success'] for row in rows) else None}
    previous.save(output / 'summary.json', summary)
    previous.save(output / 'source_integrity.json', {'unchanged': source_hashes == hashes()})
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
