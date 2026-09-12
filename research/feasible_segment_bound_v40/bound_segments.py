from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'research/coverage_probe_v39'), str(ROOT / 'research/paired_scan_diagnostics_v38')]
import numpy as np
import run_coverage as runner
from coverage_certificate import TriangleCoverage
from diagnose_rejection import invisible_heading
from offline_benchmark import OfflineRuleWorld, Source
from probe_constraints import certify_probe
from shapely.geometry import Polygon


def coverage_at(event, point):
    trial = TriangleCoverage()
    for prior in event['before_radio']:
        trial.observe_absence(prior)
    for prior in event['before_optical']:
        trial.observe_clear_absence(prior)
    trial.observe_absence(point)
    for index in event['kept']:
        trial.observe_absence(event['route'][index])
    return trial


def visibility_endpoint(station, displacement, witness):
    source = np.asarray(witness['position'])
    angle = math.radians(witness['heading_deg'])
    heading = np.array([math.cos(angle), math.sin(angle)])
    relative = station - source
    linear_start = float(relative @ heading)
    linear_change = float(displacement @ heading)
    constant = float(relative @ relative - 1000**2)
    if linear_start < 0 or constant > 0:
        return None
    squared = float(displacement @ displacement)
    mixed = float(2 * relative @ displacement)
    radius_end = (-mixed + math.sqrt(max(0.0, mixed**2 - 4 * squared * constant))) / (2 * squared)
    heading_end = linear_start / (-linear_change) if linear_change < 0 else 1.0
    return min(1.0, radius_end, heading_end)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a fresh D-drive output')
    extra = [*HERE.glob('*.py'), HERE / 'PROTOCOL.md', ROOT / 'research/paired_scan_diagnostics_v38/diagnose_rejection.py']
    hashes = {**runner.hashes(), **{str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in extra}}
    runner.experiment.save(output / 'protocol.json', {'source_hashes': hashes, 'kind': 'conditional_segment_bound_not_mission_bound', 'official_calls': 0})
    for relative in hashes:
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    records = []
    for path in sorted((args.batch / 'records').glob('*station_bind.json')):
        row = json.loads(path.read_text(encoding='utf-8'))
        for event in row['coverage_commitments']:
            anchor = np.asarray(event['anchor'])
            direction = np.asarray(event['direction'])
            mirror = np.asarray(event['candidates'][1])
            station = np.asarray(event['point'])
            relative = mirror - anchor
            template = anchor + 2 * (relative @ direction) * direction - relative
            displacement = template - station
            region = Polygon(event['vertices'])
            bearing = math.degrees(math.atan2(direction[1], direction[0]))
            if not coverage_at(event, station).empty:
                raise ValueError('Registered feasible starting station lost its certificate')
            lower, upper = 0.0, 1.0
            checks = []
            for _iteration in range(12):
                fraction = (lower + upper) / 2
                point = station + fraction * displacement
                trial = coverage_at(event, point)
                certificate = certify_probe(region, anchor, point, mirror, bearing)
                valid = certificate is not None and trial.empty
                checks.append({'fraction': fraction, 'complete': valid, 'remaining_area': trial.area})
                if valid:
                    lower = fraction
                else:
                    upper = fraction
            others = [*event['before_radio'], *(event['route'][index] for index in event['kept'])]
            witness = None
            physical_upper = None
            failed_fraction = None
            for fraction in (0.0625, 0.25, 0.5, 1.0):
                point = station + fraction * displacement
                trial = coverage_at(event, point)
                parts = list(trial.region.geoms) if hasattr(trial.region, 'geoms') else [trial.region]
                for part in sorted(parts, key=lambda part: part.area, reverse=True):
                    if part.is_empty:
                        continue
                    proposal = invisible_heading(np.asarray(part.representative_point().coords[0]), [*others, point], event['before_optical'])
                    if proposal is None:
                        continue
                    bound = visibility_endpoint(station, displacement, proposal)
                    if bound is not None and lower <= bound < fraction:
                        witness, physical_upper, failed_fraction = proposal, bound, fraction
                        break
                if witness is not None:
                    break
            record = {'profile': row['profile'], 'channel': event['channel'], 'station': event['station'],
                      'line_length': float(np.linalg.norm(displacement)), 'certified_fraction': lower,
                      'binary_checks': checks, 'physical_fraction_upper_bound': physical_upper, 'physical_witness': witness,
                      'conditional_only': 'other scan points, visit order and action schedule fixed; not an adaptive mission bound'}
            if witness is not None:
                world = OfflineRuleWorld([Source(20, tuple(witness['position']), 1000.0, witness['heading_deg'])], 401739251)
                for point in others:
                    if world.measure(np.asarray(point), 20)['result'] != 'no_signal':
                        raise ValueError('Remaining station detects the proposed witness')
                for point in event['before_optical']:
                    if world.clear(np.asarray(point), 20)['result'] != 'no_target_in_range':
                        raise ValueError('Existing optical evidence excludes the witness')
                if world.measure(station, 20)['result'] == 'no_signal':
                    raise ValueError('Original station does not detect the witness')
                if world.measure(station + failed_fraction * displacement, 20)['result'] != 'no_signal':
                    raise ValueError('Claimed failed point detects the witness')
                delta = physical_upper * record['line_length']
                record['movement_upper_meters'] = delta
                record['fixed_order_two_edge_saving_upper_seconds'] = 2 * delta / 5
                runner.experiment.save(output / (row['profile'] + '_physical_trace.json'), world.trace)
            records.append(record)
    runner.experiment.save(output / 'analysis.json', {'records': records, 'policy_runs': 0, 'official_calls': 0, 'formal_calls': 0,
                                                    'new_holdout': False, 'unit_seed_reused': 401739251})
    for record in records:
        print(json.dumps({key: value for key, value in record.items() if key not in ('binary_checks', 'physical_witness')}, indent=2))


if __name__ == '__main__':
    main()
