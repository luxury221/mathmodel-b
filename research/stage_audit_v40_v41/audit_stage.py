from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'research/q3_action_scheduler_v41'), str(ROOT / 'research/coverage_probe_v39')]
import run_action as baseline
import run_coverage
import numpy as np
from coverage_certificate import TriangleCoverage
from offline_benchmark import OfflineRuleWorld, Source
from probe_constraints import certify_probe
from shapely.geometry import Polygon


BOUND = ROOT / 'reports/feasible_segment_bound_v40/bound_20260912_070654'
ACTION = ROOT / 'reports/q3_action_scheduler_v41/dev1_20260912_071723'
COVERAGE = ROOT / 'reports/coverage_probe_v39/dev1_20260912_064358'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def visible(point, source, heading):
    relative = np.asarray(point) - source
    return float(relative @ relative) <= 1000**2 and float(relative @ heading) >= 0


def audit_bound():
    records = []
    for row in load(BOUND / 'analysis.json')['records']:
        parent = load(next((COVERAGE / 'records').glob('*_' + row['profile'] + '_13_*__station_bind.json')))
        event = next(event for event in parent['coverage_commitments'] if event['channel'] == row['channel'])
        station = np.asarray(event['point'])
        anchor = np.asarray(event['anchor'])
        direction = np.asarray(event['direction'])
        relative = np.asarray(event['candidates'][1]) - anchor
        endpoint = anchor + 2 * (relative @ direction) * direction - relative
        displacement = endpoint - station
        witness = row['physical_witness']
        source = np.asarray(witness['position'])
        angle = math.radians(witness['heading_deg'])
        heading = np.array([math.cos(angle), math.sin(angle)])
        if np.linalg.norm(source) > 1800 or not visible(station, source, heading):
            raise ValueError('Illegal constructed source or invisible original station')
        others = [*event['before_radio'], *(event['route'][index] for index in event['kept'])]
        if any(visible(point, source, heading) for point in others):
            raise ValueError('A kept station detects the witness')
        if any(np.linalg.norm(np.asarray(point) - source) <= 20 for point in event['before_optical']):
            raise ValueError('Past optical observation excludes the witness')
        lower, upper = 0.0, 1.0
        if visible(endpoint, source, heading):
            raise ValueError('Witness does not bound this segment')
        for _iteration in range(70):
            middle = (lower + upper) / 2
            if visible(station + middle * displacement, source, heading):
                lower = middle
            else:
                upper = middle
        if not math.isclose(upper, row['physical_fraction_upper_bound'], abs_tol=1e-12):
            raise ValueError('Independent visibility bisection disagrees with analytic root')
        checks = [*row['binary_checks'], {'fraction': row['certified_fraction'], 'complete': True}]
        for check in checks:
            point = station + check['fraction'] * displacement
            trial = TriangleCoverage()
            for prior in event['before_radio']:
                trial.observe_absence(prior)
            for prior in event['before_optical']:
                trial.observe_clear_absence(prior)
            trial.observe_absence(point)
            for index in event['kept']:
                trial.observe_absence(event['route'][index])
            proof = certify_probe(Polygon(event['vertices']), anchor, point, np.asarray(event['candidates'][1]),
                                  math.degrees(math.atan2(direction[1], direction[0])))
            if check['complete'] != (trial.empty and proof is not None):
                raise ValueError('Stored feasible-segment check does not reproduce')
        trace = load(BOUND / (row['profile'] + '_physical_trace.json'))
        world = OfflineRuleWorld([Source(20, tuple(source), 1000.0, witness['heading_deg'])], 401739251)
        for action in trace:
            getattr(world, action['action'])(np.asarray(action['position']), action['channel'])
            if world.trace[-1] != action:
                raise ValueError('Constructed-source physical replay differs')
        delta = upper * float(np.linalg.norm(displacement))
        if not math.isclose(delta, row['movement_upper_meters'], abs_tol=1e-8):
            raise ValueError('Displacement bound differs')
        if not math.isclose(2 * delta / 5, row['fixed_order_two_edge_saving_upper_seconds'], abs_tol=1e-8):
            raise ValueError('Fixed-order two-edge saving formula differs')
        records.append({'profile': row['profile'], 'coverage_checks': len(checks), 'physical_actions': len(trace),
                        'bisection_fraction': upper, 'movement_bound_meters': delta, 'two_edge_bound_seconds': 2 * delta / 5})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a fresh D-drive audit output')
    files = [Path(__file__), BOUND / 'analysis.json', ACTION / 'analysis.json']
    manifests = (*baseline.MANIFESTS, str((BOUND / 'protocol.json').relative_to(ROOT)),
                 str((ACTION / 'protocol.json').relative_to(ROOT)))
    verified = {}
    for relative in manifests:
        path = ROOT / relative
        files.append(path)
        frozen = load(path)['source_hashes']
        for name, digest in frozen.items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
                raise ValueError('Frozen source changed: ' + name)
            snapshot = path.parent / 'source_snapshot' / name
            if snapshot.exists() and hashlib.sha256(snapshot.read_bytes()).hexdigest() != digest:
                raise ValueError('Frozen snapshot changed: ' + name)
        verified[relative] = len(frozen)
    runner = load(ACTION / 'analysis.json')
    components = {}
    for mode in runner['summary']:
        rows = [row for row in runner['paired_cases'] if row['mode'] == mode]
        components[mode] = {key: statistics.mean(row[key] for row in rows) for key in ('movement', 'radio', 'optical')}
        if not math.isclose(sum(components[mode].values()), runner['summary'][mode]['mean'], abs_tol=1e-9):
            raise ValueError('Q3 action cost components do not sum to the task metric')
    result = {'v40': audit_bound(), 'v41_components': components, 'frozen_manifests': verified,
              'input_hashes': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
              'new_policy_runs': 0, 'new_holdout': False, 'official_calls': 0, 'formal_calls': 0,
              'bound_scope': 'single visit, visit order and all other actions fixed; not an adaptive mission or universal bound'}
    baseline.experiment.save(output, result)
    print(json.dumps({key: value for key, value in result.items() if key != 'input_hashes'}, indent=2))


if __name__ == '__main__':
    main()
