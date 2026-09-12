from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research' / 'offline_validation'))

import numpy as np
from coverage_certificate import TriangleCoverage
from geometry import dual_ring_network, open_route, ring, route_length


def main():
    output = ROOT / 'reports' / 'joint_search_v5' / ('network_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=False)
    results = []
    configurations = [('original', dual_ring_network())]
    for inner_count in (6, 7, 8):
        for inner_radius in (900, 950, 980, 990):
            for outer_count, outer_radius in ((11, 1890), (12, 1870), (13, 1860)):
                stations = np.vstack(([0, 0], ring(inner_radius, inner_count), ring(outer_radius, outer_count, 15)))
                configurations.append((f'{inner_count}x{inner_radius}_{outer_count}x{outer_radius}', stations))
    for name, stations in configurations:
        started = time.perf_counter()
        coverage = TriangleCoverage()
        coverage.observe_clear_absence([0.0, 0.0])
        for station in stations:
            coverage.observe_absence(station)
        result = {'name': name, 'points': len(stations), 'complete': coverage.empty, 'remaining_area': coverage.area,
                  'route_meters': route_length(open_route(stations)), 'seconds': time.perf_counter() - started,
                  'stations': stations.tolist()}
        results.append(result)
        print(name, 'complete', result['complete'], 'area', round(result['remaining_area'], 6),
              'route', round(result['route_meters']), flush=True)
        (output / 'networks.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print('output', output)


if __name__ == '__main__':
    main()
