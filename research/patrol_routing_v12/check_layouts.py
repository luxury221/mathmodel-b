from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'research/joint_search_v5'), str(ROOT / 'research/offline_validation')]
from coverage_certificate import TriangleCoverage
from geometry import dual_ring_network, open_route, ring, route_length


def layouts(extended=False, bridge=False):
    if bridge:
        for radius in (1010.5, 1011.0, 1011.5, 1012.0):
            yield f'inner7_{radius:g}_outer13_1860', np.vstack(([0, 0], ring(radius, 7), ring(1860.0, 13, 15)))
        return
    if extended:
        for radius in (1005.0, 1010.0, 1015.0, 1018.0):
            for inner_count, outer_count, outer_radius in ((7, 12, 1870.0), (7, 13, 1860.0), (6, 17, 1833.0), (6, 18, 1830.0)):
                yield f'inner{inner_count}_{radius:g}_outer{outer_count}_{outer_radius:g}', np.vstack(([0, 0], ring(radius, inner_count), ring(outer_radius, outer_count, 15)))
        return
    yield 'original', dual_ring_network()
    for radius in (997.0, 999.0, 1000.0):
        for outer_count, outer_radius in ((12, 1870.0), (13, 1860.0)):
            yield f'inner7_{radius:g}_outer{outer_count}_{outer_radius:g}', np.vstack(([0, 0], ring(radius, 7), ring(outer_radius, outer_count, 15)))
    for outer_count, outer_radius in ((17, 1833.0), (18, 1830.0)):
        yield f'inner6_999_outer{outer_count}_{outer_radius:g}', np.vstack(([0, 0], ring(999.0, 6), ring(outer_radius, outer_count, 15)))


def certify(points, origin_radio=True):
    proof = TriangleCoverage()
    proof.observe_clear_absence([0, 0])
    for point in points if origin_radio else points[1:]:
        proof.observe_absence(point)
    return proof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--extended', action='store_true')
    parser.add_argument('--bridge', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a new D-drive output')
    output.mkdir(parents=True)
    paths = [Path(__file__).resolve(), HERE / 'PROTOCOL.md', ROOT / 'research/joint_search_v5/coverage_certificate.py',
             ROOT / 'research/offline_validation/geometry.py', ROOT / 'research/offline_validation/plan_geometry.py']
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    (output / 'protocol.json').write_text(json.dumps({'created_local': datetime.now().isoformat(), 'source_hashes': hashes,
                                                     'official_calls': 0, 'scene_inputs': False}, indent=2), encoding='utf-8')
    for relative in hashes:
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    rows = []
    for name, points in layouts(args.extended, args.bridge):
        for origin_radio in (True,) if args.extended or args.bridge else (True, False):
            proof = certify(points, origin_radio)
            row = {'name': name, 'origin_radio': origin_radio, 'scan_points': len(points) - int(not origin_radio),
                   'complete': proof.empty, 'remaining_area': proof.area, 'route_meters': route_length(open_route(points)),
                   'stations': points.tolist()}
            rows.append(row)
            print(name, 'origin_radio', origin_radio, 'complete', proof.empty, 'remaining_area', round(proof.area, 6),
                  'route_meters', round(row['route_meters'], 3), flush=True)
            (output / 'layouts.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    integrity = all(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest for relative, digest in hashes.items())
    (output / 'summary.json').write_text(json.dumps({'checked': len(rows), 'complete': sum(row['complete'] for row in rows),
                                                    'source_hashes_unchanged': integrity, 'official_calls': 0}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
