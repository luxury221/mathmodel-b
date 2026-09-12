from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'research/q3_scan_economy_v13'), str(ROOT / 'research/radial_patrol_v7'),
                str(ROOT / 'research/joint_search_v5'), str(ROOT / 'research/shaped_probe_v11')]
import run_scan_benchmark as baseline
from scan_policy import ScanEconomyPolicy
from shaped_policy import ShapedProbePolicy


experiment = baseline.experiment
EVALUATOR_SPEC = importlib.util.spec_from_file_location('publication_frozen_evaluator', ROOT / 'research/joint_search_validation_v7/validate.py')
frozen_evaluator = importlib.util.module_from_spec(EVALUATOR_SPEC)
EVALUATOR_SPEC.loader.exec_module(frozen_evaluator)
EvaluationWorld = frozen_evaluator.EvaluationWorld
PROFILES = ('random', 'boundary', 'clustered', 'near_origin', 'max_radius', 'adversarial_heading', 'boundary_noise')
ERROR_MODES = ('spatial_uniform', 'constant_extreme', 'spatial_extreme', 'spatial_correlated')


def policy_factory(port, problem, stations, variant, network, mode):
    arguments = (port, problem, stations, variant, network)
    if problem == 'q3':
        policy = ScanEconomyPolicy(*arguments, mode='station_only')
        if policy.options['target_radius'] != 2000:
            raise ValueError('Frozen Q3 admission radius changed')
        return policy
    if problem == 'q4':
        return ShapedProbePolicy(*arguments, mode='shaped_cost')
    raise ValueError('Only Q3 and Q4 offline baselines are supported')


def main():
    parser = argparse.ArgumentParser(description='Local simulations only; never calls the official simulator')
    parser.add_argument('--problem', choices=('q3', 'q4', 'all'), default='all')
    parser.add_argument('--counts', nargs='+', type=int, choices=(10, 13, 16), default=[13])
    parser.add_argument('--profiles', nargs='+', choices=PROFILES, default=list(PROFILES))
    parser.add_argument('--case-file', type=Path)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', required=True, type=Path)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    if output.exists() or (os.name == 'nt' and output.drive.upper() != 'D:'):
        raise ValueError('Use a fresh output directory; Windows outputs must remain on D:')
    if arguments.limit is not None and arguments.limit <= 0:
        raise ValueError('A subset limit must be positive')
    if arguments.case_file:
        scenes = json.loads(arguments.case_file.read_text(encoding='utf-8'))
        data_source = 'historical_case_file_not_fresh_holdout'
    else:
        scenes = experiment.make_scenes(103100000, 'development', arguments.counts)
        data_source = 'previously_used_development_base_103100000'
    problems = ('q3', 'q4') if arguments.problem == 'all' else (arguments.problem,)
    selected = [scene for scene in scenes if scene['problem'] in problems and scene['profile'] in arguments.profiles]
    available_count = len(selected)
    if arguments.limit is not None:
        selected = selected[:arguments.limit]
    if not selected:
        raise ValueError('No cases match the requested selection')
    if any(scene['error_mode'] not in ERROR_MODES for scene in selected):
        raise ValueError('Unsupported historical noise model')
    experiment.save(output / 'protocol.json', {
        'evidence_type': 'local_offline_simulation', 'official_calls': 0, 'formal_calls': 0,
        'data_source': data_source, 'available_cases': available_count, 'selected_cases': len(selected),
        'subset_only': len(selected) != available_count,
        'world_factory': 'research/joint_search_validation_v7/validate.py:EvaluationWorld',
        'metric': 'case-equal complete mission virtual seconds per source',
        'q3': 'v13 station_only; target_radius=2000', 'q4': 'v11 shaped_cost',
    })
    experiment.save(output / 'scenes_evaluator_only.json', selected)
    records = []
    for scene in selected:
        mode = 'station_only' if scene['problem'] == 'q3' else 'shaped_cost'
        row, trace = experiment.evaluate(scene, mode, policy_factory=policy_factory, world_factory=EvaluationWorld)
        baseline.audit_trace(trace, row['virtual_seconds'])
        name = scene['id'] + '__' + mode + '.json'
        experiment.save(output / 'records' / name, row)
        experiment.save(output / 'traces' / name, trace)
        records.append(row)
        print(scene['problem'], scene['profile'], scene['count'],
              row['seconds_per_source'] if row['success'] else row['failure'], flush=True)
    summary = {'evidence_type': 'local_offline_simulation', 'official_calls': 0, 'formal_calls': 0,
               'subset_only': len(selected) != available_count, 'problems': {}}
    for problem in problems:
        rows = [row for row in records if row['problem'] == problem]
        if rows:
            summary['problems'][problem] = {
                'runs': len(rows), 'successes': sum(row['success'] for row in rows),
                'mean_seconds_per_source': statistics.mean(row['seconds_per_source'] for row in rows)
                if all(row['success'] for row in rows) else None,
            }
    experiment.save(output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not all(row['success'] for row in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
