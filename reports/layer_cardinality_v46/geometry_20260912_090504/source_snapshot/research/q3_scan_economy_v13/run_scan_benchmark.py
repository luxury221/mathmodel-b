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
from final_validation import audit_trace
from radial_policy import RadialPolicy
from scan_policy import MODES, ScanEconomyPolicy


def source_hashes():
    frozen = json.loads((ROOT / 'reports/shaped_probe_validation_v11/validation_20260911_223412/selection.json').read_text(encoding='utf-8'))['source_hashes']
    for relative, digest in frozen.items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
            raise ValueError('Previously frozen code changed')
    return {**frozen, **{str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in HERE.glob('*.py')}}


def evaluate(scene, mode):
    holder = []
    def factory(port, problem, stations, variant, network, mode):
        policy = RadialPolicy(port, problem, stations, variant, network, mode='pilot1600') if mode == 'previous' else ScanEconomyPolicy(port, problem, stations, variant, network, mode=mode)
        holder.append(policy)
        return policy
    row, trace = experiment.evaluate(scene, mode, policy_factory=factory)
    row['replacement_proofs'] = getattr(holder[0], 'replacement_proofs', [])
    row['timing_audit'] = audit_trace(trace, row['virtual_seconds'])
    return row, trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--counts', nargs='+', type=int, default=[13])
    parser.add_argument('--modes', nargs='+', choices=('previous', *MODES), default=['previous', *MODES])
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a new D-drive output')
    frozen = source_hashes()
    scenes = [scene for scene in experiment.make_scenes(103100000, 'development', args.counts) if scene['problem'] == 'q3']
    experiment.save(output / 'protocol.json', {'created_local': datetime.now().isoformat(), 'source_hashes': frozen,
                                               'modes': args.modes, 'official_calls': 0,
                                               'metric': 'case-equal total virtual seconds/source; failures invalidate a mode mean'})
    experiment.save(output / 'scenes_evaluator_only.json', scenes)
    for relative in frozen:
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    shutil.copy2(HERE / 'PROTOCOL.md', output / 'PROTOCOL.md')
    records = []
    for scene in scenes:
        for mode in args.modes:
            row, trace = evaluate(scene, mode)
            records.append(row)
            name = scene['id'] + '__' + mode + '.json'
            experiment.save(output / 'records' / name, row)
            experiment.save(output / 'traces' / name, trace)
            print(scene['profile'], scene['count'], mode, round(row['seconds_per_source'], 3) if row['success'] else row['failure'],
                  'scans', row['policy_stats']['scan_stops'], 'substitutions', row['policy_stats'].get('certified_scan_substitutions', 0), flush=True)
    summary = {}
    for mode in args.modes:
        rows = [row for row in records if row['mode'] == mode]
        summary[mode] = {'runs': len(rows), 'successes': sum(row['success'] for row in rows),
                         'mean': statistics.mean(row['seconds_per_source'] for row in rows) if all(row['success'] for row in rows) else None}
    experiment.save(output / 'summary.json', summary)
    experiment.save(output / 'source_integrity.json', {'unchanged': frozen == source_hashes()})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
