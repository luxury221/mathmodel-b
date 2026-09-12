from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / 'research/q3_scan_economy_v13'), str(ROOT / 'research/shaped_probe_validation_v11'),
                str(ROOT / 'research/joint_search_v5'), str(ROOT / 'research/radial_patrol_v7')]
import experiment_v5 as experiment
from radial_policy import RadialPolicy
from scan_policy import ScanEconomyPolicy
from run_scan_benchmark import source_hashes as scan_hashes
from validate_shaped import frozen_verifier, prior_seed_audit, promotion_gate


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def source_hashes():
    return {**scan_hashes(), **{str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in HERE.glob('*.py')}}


def select_candidate(rows):
    reference = {row['scene_id'] for row in rows if row['mode'] == 'previous'}
    if len(reference) != 18:
        raise ValueError('Need all 18 registered Q3 development cases')
    means = {}
    for mode in ('previous', 'station_only', 'certified_reuse'):
        selected = [row for row in rows if row['mode'] == mode]
        if len(selected) != 18 or {row['scene_id'] for row in selected} != reference or not all(row['success'] for row in selected):
            raise ValueError('Incomplete or failed development comparison')
        means[mode] = statistics.mean(row['seconds_per_source'] for row in selected)
    candidate = 'station_only' if means['station_only'] <= means['certified_reuse'] + 1 else 'certified_reuse'
    return candidate, means


def make_scenes():
    scenes = [scene for scene in experiment.make_scenes(261100000, 'holdout', (10, 12, 14, 16)) if scene['problem'] == 'q3']
    for scene in scenes:
        scene['id'] = scene['id'].replace('jointv4_', 'scanv13_')
    for count in (10, 16):
        for repetition in range(4):
            seed = 267100000 + count * 100 + repetition
            generator = np.random.default_rng(seed)
            rotation = generator.uniform(0, 2 * np.pi)
            channels = generator.choice(np.arange(1, 21), count, replace=False)
            sources = []
            for index, channel in enumerate(channels):
                distance = (0.001, 5.000001, 20.000001, 999.999999, 1000.000001, 1799.999999)[index % 6]
                angle = rotation + index * 2 * np.pi / count
                point = distance * np.array([np.cos(angle), np.sin(angle)])
                sources.append({'channel': int(channel), 'position': point.tolist(), 'radius': 1000.0, 'heading_deg': None})
            scenes.append({'id': f'scanv13_stress_q3_boundary_noise_{count}_{repetition}', 'problem': 'q3',
                           'count': count, 'seed': seed, 'profile': 'boundary_noise', 'phase': 'stress',
                           'error_mode': 'constant_extreme' if repetition < 2 else 'spatial_correlated', 'sources': sources})
    return scenes


def evaluate(scene, mode):
    def factory(port, problem, stations, variant, network, mode):
        return RadialPolicy(port, problem, stations, variant, network, mode='pilot1600') if mode == 'previous' else ScanEconomyPolicy(port, problem, stations, variant, network, mode=mode)
    return experiment.evaluate(scene, mode, policy_factory=factory, world_factory=frozen_verifier.EvaluationWorld)


def local_http(output, scenes, records, selected):
    def factory(client, problem, q4_network='dual21'):
        original, metadata = frozen_verifier.make_policy(client, problem, q4_network)
        adapted = type('ScanV13LocalAdapter', (ScanEconomyPolicy, type(original)), {})
        policy = adapted(original.port, problem, original.stations, 'E_joint', 'grid7', mode=selected)
        if adapted.account is not type(original).account:
            raise ValueError('Local HTTP adapter lost authoritative accounting')
        metadata.update({'local_candidate_only': True, 'variant': selected})
        return policy, metadata
    with patch.object(frozen_verifier, 'SELECTED', {'q3': selected}), patch.object(frozen_verifier, 'http_factory', factory):
        return frozen_verifier.http_checks(output, scenes, records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--development', nargs='+', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a new D-drive output')
    development = []
    for directory in args.development:
        if not load(directory / 'source_integrity.json')['unchanged']:
            raise ValueError('Development sources changed during the run')
        for relative, expected in load(directory / 'protocol.json')['source_hashes'].items():
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
                raise ValueError('Development code changed before selection')
        development.extend(load(path) for path in (directory / 'records').glob('*.json'))
    selected, means = select_candidate(development)
    if (1 - means[selected] / means['previous']) * 100 < 1:
        raise ValueError('Insufficient preregistered development gain')
    frozen = source_hashes()
    experiment.save(output / 'selection.json', {'selected': selected, 'development_means': means,
                                               'source_hashes': frozen, 'created_local': datetime.now().isoformat(),
                                               'holdout_seed_base': 261100000, 'stress_seed_base': 267100000,
                                               'official_calls': 0, 'no_post_holdout_tuning': True})
    for relative in frozen:
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    shutil.copy2(HERE / 'PROTOCOL.md', output / 'PROTOCOL.md')
    scenes = make_scenes()
    experiment.save(output / 'seed_audit.json', prior_seed_audit(scenes, ROOT / 'reports'))
    experiment.save(output / 'scenes_evaluator_only.json', scenes)
    records, audits = [], []
    for scene in scenes:
        for mode in ('previous', selected):
            row, trace = evaluate(scene, mode)
            records.append(row)
            name = scene['id'] + '__' + mode + '.json'
            experiment.save(output / 'records' / name, row)
            experiment.save(output / 'traces' / name, trace)
            audits.append(frozen_verifier.audit_trace(trace, row['virtual_seconds']))
            print(scene['phase'], scene['profile'], scene['count'], mode,
                  round(row['seconds_per_source'], 3) if row['success'] else row['failure'], flush=True)
    with patch.object(frozen_verifier, 'SELECTED', {'q3': selected}):
        summary = frozen_verifier.summarize(records)['q3']
    experiment.save(output / 'summary.json', summary)
    replays = []
    for profile, count in (('random', 10), ('near_origin', 16)):
        scene = next(scene for scene in scenes if scene['profile'] == profile and scene['count'] == count)
        direct = next(row for row in records if row['scene_id'] == scene['id'] and row['mode'] == selected)
        replay, _trace = evaluate(scene, selected)
        replays.append({'scene_id': scene['id'], 'passed': direct['success'] and replay['success'] and direct['trace_sha256'] == replay['trace_sha256']})
    experiment.save(output / 'replays.json', replays)
    performance_passed = promotion_gate(summary)
    http = local_http(output, scenes, records, selected) if performance_passed else []
    verification = {'source_hashes_unchanged': source_hashes() == frozen, 'runs': len(records),
                    'successes': sum(row['success'] for row in records), 'actions_audited': sum(row['actions'] for row in audits),
                    'max_clock_error': max(row['max_clock_error'] for row in audits),
                    'replays_passed': sum(row['passed'] for row in replays), 'replays_total': len(replays),
                    'http_passed': sum(row['passed'] for row in http), 'http_total': len(http),
                    'performance_gate_passed': performance_passed, 'official_calls': 0,
                    'q3_target_met': summary['combined'].get('target_met_on_this_phase', False),
                    'q4_changed': False, 'goal_achieved': False}
    verification['eligible_as_next_offline_candidate'] = (performance_passed and verification['source_hashes_unchanged']
                                                         and all(row['passed'] for row in replays)
                                                         and len(http) == 3 and all(row['passed'] for row in http))
    experiment.save(output / 'verification.json', verification)
    print(json.dumps({'selected': selected, 'summary': summary, 'verification': verification}, indent=2), flush=True)


if __name__ == '__main__':
    main()
