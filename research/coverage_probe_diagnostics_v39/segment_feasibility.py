from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'research/coverage_probe_v39'))
import numpy as np
import run_coverage as runner
from coverage_certificate import TriangleCoverage
from probe_constraints import certify_probe
from shapely.geometry import Polygon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a fresh D-drive geometry diagnostic output')
    fractions = (0.0, 1 / 64, 1 / 16, 0.25, 0.5, 0.75, 1.0)
    hashes = {**runner.hashes(), str(Path(__file__).resolve().relative_to(ROOT)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    runner.experiment.save(output / 'protocol.json', {'kind': 'post_hoc_geometry_only_not_performance',
                                                    'source_hashes': hashes, 'fractions': fractions,
                                                    'cases': 'the two actually executed station_bind commitments',
                                                    'policy_unchanged': True, 'official_calls': 0})
    records = []
    for path in sorted((args.batch / 'records').glob('*station_bind.json')):
        row = json.loads(path.read_text(encoding='utf-8'))
        trace = json.loads((args.batch / 'traces' / path.name).read_text(encoding='utf-8'))
        reference = json.loads((args.batch / 'records' / path.name.replace('__station_bind', '__previous')).read_text(encoding='utf-8'))
        for event in row['coverage_commitments']:
            anchor = np.asarray(event['anchor'])
            direction = np.asarray(event['direction'])
            mirror = np.asarray(event['candidates'][1])
            station = np.asarray(event['point'])
            displacement = mirror - anchor
            template = anchor + 2 * (displacement @ direction) * direction - displacement
            region = Polygon(event['vertices'])
            bearing = math.degrees(math.atan2(direction[1], direction[0]))
            prefix = [action for action in trace if action['virtual_seconds'] <= event['start_seconds']]
            measured = [np.asarray(action['position']) for action in prefix
                        if action['action'] == 'measure' and action['channel'] == event['channel']]
            positions = [*measured, *(np.asarray(point) for point in event['before_radio'])]
            record = {'profile': row['profile'], 'channel': event['channel'], 'station': event['station'],
                      'station_point': station.tolist(), 'original_same_side_probe': template.tolist(),
                      'first_failed_in_actual_run': event['first_failed'],
                      'predicted_saved_seconds_per_task': event['baseline_proxy_seconds'] - event['proposed_proxy_seconds'],
                      'actual_added_seconds_per_task': row['virtual_seconds'] - reference['virtual_seconds'],
                      'fractions': []}
            for fraction in fractions:
                point = station + fraction * (template - station)
                certificate = certify_probe(region, anchor, point, mirror, bearing)
                coverage = TriangleCoverage()
                for prior in event['before_radio']:
                    coverage.observe_absence(prior)
                for prior in event['before_optical']:
                    coverage.observe_clear_absence(prior)
                coverage.observe_absence(point)
                for index in event['kept']:
                    coverage.observe_absence(event['route'][index])
                fresh = all(np.linalg.norm(point - prior) >= 50 for prior in positions)
                record['fractions'].append({'fraction': fraction, 'point': point.tolist(),
                                            'probe_certified': certificate is not None, 'coverage_complete': coverage.empty,
                                            'remaining_area': coverage.area, 'fresh': fresh,
                                            'distance_from_station': float(np.linalg.norm(point - station))})
            records.append(record)
    complete_nonstation = sum(item['fraction'] > 0 and item['probe_certified'] and item['coverage_complete'] and item['fresh']
                              for record in records for item in record['fractions'])
    result = {'post_hoc_geometry_only': True, 'new_policy_runs': 0, 'official_calls': 0,
              'registered_binding_events': len(records), 'geometry_checks': len(records) * len(fractions),
              'complete_fresh_nonstation_points': complete_nonstation, 'records': records}
    runner.experiment.save(output / 'analysis.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'records'}, indent=2))
    for record in records:
        print(record['profile'], 'proxy_saved', record['predicted_saved_seconds_per_task'],
              'actually_added', record['actual_added_seconds_per_task'])
        for item in record['fractions']:
            print(item['fraction'], item['probe_certified'], item['coverage_complete'], item['fresh'], round(item['remaining_area'], 6))


if __name__ == '__main__':
    main()
