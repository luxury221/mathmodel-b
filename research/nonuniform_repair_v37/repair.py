from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'research/network_synthesis_v8'), str(ROOT / 'research/annular_topology_v25')]
import numpy as np
import synthesis
from topology import make_layout


def starting_layouts():
    for inner_count, outer_count in ((7, 12), (8, 11), (6, 12)):
        points = make_layout(inner_count, outer_count, 1000.0,
                             1800 / math.cos(math.pi / outer_count) + 2.0, 0.0)
        yield f'inner{inner_count}_outer{outer_count}', points


def hashes():
    paths = [*HERE.glob('*.py'), HERE / 'PROTOCOL.md', ROOT / 'research/annular_topology_v25/topology.py']
    return {**synthesis.source_hashes(),
            **{str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a fresh D-drive output')
    frozen = hashes()
    synthesis.save(output / 'protocol.json', {'source_hashes': frozen, 'created_local': datetime.now().isoformat(),
                                            'official_calls': 0, 'source_scenes_read': False,
                                            'iterations_per_start': 15, 'trust_radius_meters': 60,
                                            'slack_penalty': 200000})
    shutil.copy2(HERE / 'PROTOCOL.md', output / 'PROTOCOL.md')
    for relative in frozen:
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    baseline_points = synthesis.dual_ring_network()
    baseline_length = synthesis.route_length(synthesis.open_route(baseline_points[1:]))
    baseline_cost = baseline_length + 300 * (len(baseline_points) - 1)
    initial_positions, initial_headings = synthesis.initial_witnesses()
    synthesis.save(output / 'initial_witnesses.json', {'positions': initial_positions.tolist(), 'headings': initial_headings.tolist()})
    records = []
    started = time.perf_counter()
    for name, points in starting_layouts():
        positions, headings = initial_positions.copy(), initial_headings.copy()
        synthesis.save(output / 'starts' / (name + '.json'), {'stations': points.tolist(), 'receivers': len(points)})
        initial_coverage = synthesis.certify(points)
        for violation, position, heading in synthesis.counterexamples(initial_coverage.region, points, limit=64):
            positions = np.vstack((positions, position))
            headings = np.vstack((headings, heading))
        for iteration in range(15):
            proposal, metadata = synthesis.optimize_step(points, positions, headings, penalty=200000, trust_radius=60)
            row = {'layout': name, 'iteration': iteration, 'receivers': len(points), **metadata}
            if proposal is None:
                row['complete'] = False
                records.append(row)
                synthesis.save(output / 'iterations' / f'{name}_{iteration:02d}.json', row)
                break
            points = proposal
            row['stations'] = points.tolist()
            row['witness_count'] = len(positions)
            row['route_length_meters'] = synthesis.route_length(synthesis.open_route(points[1:]))
            row['proxy_cost_meters'] = row['route_length_meters'] + 300 * (len(points) - 1)
            try:
                coverage = synthesis.certify(points)
                row['complete'] = coverage.empty
                row['remaining_area'] = float(coverage.area)
                row['remaining_geometry'] = coverage.region.wkt
                examples = [] if coverage.empty else synthesis.counterexamples(coverage.region, points, limit=64)
                row['counterexamples'] = [{'violation': violation, 'position': position.tolist(), 'heading': heading.tolist()}
                                          for violation, position, heading in examples]
                if examples:
                    positions = np.vstack((positions, [example[1] for example in examples]))
                    headings = np.vstack((headings, [example[2] for example in examples]))
            except Exception as error:
                row.update({'complete': False, 'geometry_error': type(error).__name__ + ': ' + str(error)})
            records.append(row)
            synthesis.save(output / 'iterations' / f'{name}_{iteration:02d}.json', row)
            print(name, iteration, 'slack', row.get('slack_meters'), 'area', row.get('remaining_area'),
                  'complete', row['complete'], 'cost', row['proxy_cost_meters'], flush=True)
            if row['complete'] or row.get('geometry_error'):
                break
    complete = [row for row in records if row['complete']]
    eligible = [row for row in complete if row['proxy_cost_meters'] <= baseline_cost * 0.99]
    selected = min(eligible, key=lambda row: row['proxy_cost_meters']) if eligible else None
    synthesis.save(output / 'selected.json', selected)
    summary = {'starts': 3, 'iterations': len(records), 'complete_layouts': len(complete),
               'eligible_layouts': len(eligible), 'baseline_proxy_cost_meters': baseline_cost,
               'best_remaining_area': min((row['remaining_area'] for row in records if 'remaining_area' in row), default=None),
               'source_integrity': hashes() == frozen, 'wall_seconds': time.perf_counter() - started,
               'official_calls': 0, 'not_a_global_infeasibility_proof': True}
    synthesis.save(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    if not summary['source_integrity']:
        raise ValueError('Repair sources changed during execution')


if __name__ == '__main__':
    main()
