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
sys.path[:0] = [str(HERE), str(ROOT / 'research/q3_scan_economy_v13'), str(ROOT / 'research/joint_search_v5'),
                str(ROOT / 'research/radial_patrol_v7')]
import experiment_v5 as experiment
from final_validation import audit_trace
from empirical_policy import MODES, EmpiricalPointPolicy
from radial_policy import RadialPolicy
from run_scan_benchmark import source_hashes as scan_hashes
from scan_policy import ScanEconomyPolicy


def source_hashes():
    return {**scan_hashes(), **{str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in HERE.glob('*.py')}}


def evaluate(scene, mode):
    holder = []
    def factory(port, problem, stations, variant, network, mode):
        if mode == 'previous':
            policy = RadialPolicy(port, problem, stations, variant, network, mode='pilot1600')
        elif mode == 'station_only':
            policy = ScanEconomyPolicy(port, problem, stations, variant, network, mode=mode)
        else:
            policy = EmpiricalPointPolicy(port, problem, stations, variant, network, mode=mode)
        holder.append(policy)
        return policy
    row, trace = experiment.evaluate(scene, mode, policy_factory=factory)
    row['empirical_events'] = getattr(holder[0], 'empirical_events', [])
    row['timing_audit'] = audit_trace(trace, row['virtual_seconds'])
    return row, trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--counts', nargs='+', type=int, default=[13])
    parser.add_argument('--modes', nargs='+', choices=('previous', 'station_only', *MODES), default=['previous', 'station_only', *MODES])
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a new D-drive output')
    frozen = source_hashes()
    scenes = [scene for scene in experiment.make_scenes(103100000, 'development', args.counts) if scene['problem'] == 'q3']
    experiment.save(output / 'protocol.json', {'created_local': datetime.now().isoformat(), 'source_hashes': frozen,
                                               'modes': args.modes, 'official_calls': 0,
                                               'metric': 'case-equal total virtual seconds/source; all failures retained'})
    experiment.save(output / 'scenes_evaluator_only.json', scenes)
    for relative in frozen:
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    shutil.copy2(HERE / 'PROTOCOL.md', output / 'PROTOCOL.md')
    rows = []
    for scene in scenes:
        for mode in args.modes:
            row, trace = evaluate(scene, mode)
            rows.append(row)
            name = scene['id'] + '__' + mode + '.json'
            experiment.save(output / 'records' / name, row)
            experiment.save(output / 'traces' / name, trace)
            print(scene['profile'], scene['count'], mode, round(row['seconds_per_source'], 3) if row['success'] else row['failure'],
                  'trials', row['policy_stats'].get('empirical_trials', 0), 'hits', row['policy_stats'].get('empirical_hits', 0), flush=True)
    summary = {}
    for mode in args.modes:
        chosen = [row for row in rows if row['mode'] == mode]
        summary[mode] = {'runs': len(chosen), 'successes': sum(row['success'] for row in chosen),
                         'mean': statistics.mean(row['seconds_per_source'] for row in chosen) if all(row['success'] for row in chosen) else None}
    experiment.save(output / 'summary.json', summary)
    experiment.save(output / 'source_integrity.json', {'unchanged': frozen == source_hashes()})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
