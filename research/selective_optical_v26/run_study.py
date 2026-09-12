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
sys.path[:0] = [str(HERE), str(ROOT / 'research/shaped_probe_v11'), str(ROOT / 'research/joint_search_v5')]
import experiment_v5 as experiment
import numpy as np
from complete_network import select_network
from optical_policy import MODES, SelectiveOpticalPolicy
from shaped_policy import ShapedProbePolicy


def hashes():
    paths = [*HERE.glob('*.py'), *(ROOT / 'research/shaped_probe_v11').glob('*.py')]
    return {**experiment.source_hashes(), **{str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}


def evaluate(scene, mode, stations, world_factory=None):
    holder = []
    def factory(port, problem, default_stations, variant, network, mode):
        policy = (ShapedProbePolicy(port, problem, default_stations, variant, network, mode='shaped_cost')
                  if mode == 'previous' else SelectiveOpticalPolicy(port, problem, np.asarray(stations), variant, 'radio_complete_v26', mode=mode))
        holder.append(policy)
        return policy
    options = {} if world_factory is None else {'world_factory': world_factory}
    row, trace = experiment.evaluate(scene, mode, policy_factory=factory, **options)
    row['initial_optical_events'] = getattr(holder[0], 'initial_optical_events', [])
    row['shaped_events'] = getattr(holder[0], 'shaped_events', [])
    return row, trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--counts', type=int, nargs='+', default=[13])
    parser.add_argument('--modes', nargs='+', choices=('previous', *MODES), default=['previous', *MODES])
    parser.add_argument('--network', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a new D-drive output')
    frozen = hashes()
    network_hash = hashlib.sha256(args.network.read_bytes()).hexdigest() if args.network else None
    experiment.save(output / 'protocol.json', {'created_local': datetime.now().isoformat(), 'source_hashes': frozen,
                                              'modes': args.modes, 'official_calls': 0,
                                              'network_input': str(args.network.resolve()) if args.network else None,
                                              'network_input_sha256': network_hash,
                                              'metric': 'case-equal complete mission virtual seconds/source; failures invalidate mean'})
    shutil.copy2(HERE / 'PROTOCOL.md', output / 'PROTOCOL.md')
    for relative in frozen:
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    network = json.loads(args.network.read_text(encoding='utf-8-sig')) if args.network else select_network()
    experiment.save(output / 'network.json', network)
    if network['selected'] is None:
        experiment.save(output / 'summary.json', {'decision': 'no_complete_radio_network_no_performance_runs'})
        return
    stations = network['selected']['stations']
    print('selected_radio_network', {key: value for key, value in network['selected'].items() if key not in ('stations', 'remaining_geometry')}, flush=True)
    scenes = [scene for scene in experiment.make_scenes(103100000, 'development', args.counts) if scene['problem'] == 'q4']
    experiment.save(output / 'scenes_evaluator_only.json', scenes)
    rows = []
    for scene in scenes:
        for mode in args.modes:
            row, trace = evaluate(scene, mode, stations)
            rows.append(row)
            name = scene['id'] + '__' + mode + '.json'
            experiment.save(output / 'records' / name, row)
            experiment.save(output / 'traces' / name, trace)
            print(scene['profile'], scene['count'], mode,
                  round(row['seconds_per_source'], 3) if row['success'] else row['failure'],
                  'initial_suppressed', row['policy_stats'].get('origin_optical_suppressed', 0), flush=True)
    summary = {mode: {'runs': len(selected), 'successes': sum(row['success'] for row in selected),
                      'mean': statistics.mean(row['seconds_per_source'] for row in selected) if all(row['success'] for row in selected) else None}
               for mode in args.modes for selected in [[row for row in rows if row['mode'] == mode]]}
    experiment.save(output / 'summary.json', summary)
    network_unchanged = not args.network or network_hash == hashlib.sha256(args.network.read_bytes()).hexdigest()
    experiment.save(output / 'source_integrity.json', {'unchanged': frozen == hashes(), 'network_unchanged': network_unchanged})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
