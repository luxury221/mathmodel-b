from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
sys.path[:0] = [str(ROOT / 'research/shaped_probe_v11'), str(ROOT / 'research/joint_search_v5')]
import experiment_v5 as experiment
from policy_v5 import ContinuousPolicy
from shaped_policy import MODES, ShapedProbePolicy
from geometry import dual_ring_network

SPEC = importlib.util.spec_from_file_location('frozen_v7_verifier', ROOT / 'research/joint_search_validation_v7/validate.py')
frozen_verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(frozen_verifier)


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def hashes():
    paths = [*HERE.glob('*.py'), *(ROOT / 'research/shaped_probe_v11').glob('*.py'),
             ROOT / 'scripts/env.ps1', ROOT / 'requirements.lock.txt']
    return {**frozen_verifier.hashes(), **{str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}


def select_candidate(rows):
    scores = {}
    reference = {row['scene_id'] for row in rows if row['mode'] == 'previous'}
    if len(reference) != 18:
        raise ValueError('Selection requires all 18 registered development scenes')
    for mode in ('previous', *MODES):
        selected = [row for row in rows if row['mode'] == mode]
        if len(selected) != 18 or {row['scene_id'] for row in selected} != reference or not all(row['success'] for row in selected):
            raise ValueError('Development comparison is incomplete or includes failures')
        scores[mode] = statistics.mean(row['seconds_per_source'] for row in selected)
    best = min(scores[mode] for mode in MODES)
    chosen = next(mode for mode in ('narrow85', 'narrow70', 'shaped_cost') if scores[mode] <= best + 1)
    return chosen, scores


def prior_seed_audit(scenes, reports_root):
    prior_paths = sorted(reports_root.rglob('*scenes*.json'))
    seeds = set()
    for path in prior_paths:
        content = load(path)
        if isinstance(content, list):
            seeds.update(scene['seed'] for scene in content if isinstance(scene, dict) and 'seed' in scene)
    proposed = [scene['seed'] for scene in scenes]
    if len(set(proposed)) != len(proposed) or set(proposed) & seeds:
        raise ValueError('New validation seeds overlap existing scene seeds')
    return {'unique': len(proposed), 'prior_count': len(seeds), 'overlap': [],
            'prior_files': {str(path.relative_to(reports_root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in prior_paths}}


def make_validation_scenes():
    scenes = [scene for scene in experiment.make_scenes(231100000, 'holdout', (10, 12, 14, 16)) if scene['problem'] == 'q4']
    for scene in scenes:
        scene['id'] = scene['id'].replace('jointv4_', 'shapedv11_')
    for count in (10, 16):
        for repetition in range(4):
            seed = 237100000 + count * 100 + repetition
            generator = np.random.default_rng(seed)
            rotation = generator.uniform(0, 2 * np.pi)
            channels = generator.choice(np.arange(1, 21), count, replace=False)
            sources = []
            for index, channel in enumerate(channels):
                distance = (0.001, 5.000001, 20.000001, 999.999999, 1000.000001, 1799.999999)[index % 6]
                angle = rotation + index * 2 * np.pi / count
                position = distance * np.array([np.cos(angle), np.sin(angle)])
                angles = np.deg2rad(np.arange(720) / 2)
                headings = np.column_stack((np.cos(angles), np.sin(angles)))
                vectors = dual_ring_network() - position
                visible = (headings @ vectors.T >= 0) & (np.linalg.norm(vectors, axis=1) <= 1000)
                heading = float(np.argmin(visible.sum(axis=1)) / 2)
                sources.append({'channel': int(channel), 'position': position.tolist(), 'radius': 1000.0, 'heading_deg': heading})
            scenes.append({'id': f'shapedv11_stress_q4_boundary_noise_{count}_{repetition}', 'problem': 'q4',
                           'count': count, 'seed': seed, 'profile': 'boundary_noise', 'phase': 'stress',
                           'error_mode': 'constant_extreme' if repetition < 2 else 'spatial_correlated', 'sources': sources})
    return scenes


def evaluate(scene, mode):
    def factory(port, problem, stations, variant, network, mode):
        return ContinuousPolicy(port, problem, stations, variant, network, mode='certified_fixed') if mode == 'previous' else ShapedProbePolicy(port, problem, stations, variant, network, mode=mode)
    return experiment.evaluate(scene, mode, policy_factory=factory, world_factory=frozen_verifier.EvaluationWorld)


def promotion_gate(summary):
    combined = summary['combined']
    return (combined.get('all_passed', False) and combined.get('improvement_percent', -1) >= 1
            and combined.get('worst_regression_percent', 100) <= 15
            and all(summary[phase].get('improvement_percent', -1) >= 0 for phase in ('holdout', 'stress')))


def local_http(output, scenes, records, chosen):
    def factory(client, problem, q4_network='dual21'):
        original, metadata = frozen_verifier.make_policy(client, problem, q4_network)
        adapted = type('ShapedV11LocalAdapter', (ShapedProbePolicy, type(original)), {})
        policy = adapted(original.port, problem, original.stations, 'F_route', 'dual21', mode=chosen)
        if adapted.account is not type(original).account:
            raise ValueError('Local HTTP adapter lost authoritative accounting')
        metadata.update({'local_candidate_only': True, 'variant': chosen})
        return policy, metadata
    with patch.object(frozen_verifier, 'SELECTED', {'q4': chosen}), patch.object(frozen_verifier, 'http_factory', factory):
        return frozen_verifier.http_checks(output, scenes, records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--development', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a new D-drive validation directory')
    rows = []
    for directory in args.development:
        if not load(directory / 'source_integrity.json')['unchanged']:
            raise ValueError('Development sources changed during execution')
        manifest = load(directory / 'protocol.json')['source_hashes']
        for relative, expected in manifest.items():
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
                raise ValueError('Development source changed before validation: ' + relative)
        rows.extend(load(path) for path in (directory / 'records').glob('*.json'))
    chosen, scores = select_candidate(rows)
    if (1 - scores[chosen] / scores['previous']) * 100 < 1:
        raise ValueError('Development gain is below the preregistered validation threshold')
    frozen = hashes()
    old_manifest = load(ROOT / 'reports/joint_search_validation_v7/validation_20260911_205854/selection.json')['source_hashes']
    if any(frozen[relative] != expected for relative, expected in old_manifest.items()):
        raise ValueError('Previously frozen code changed')
    experiment.save(output / 'selection.json', {'selected': chosen, 'development_means': scores,
                                               'source_hashes': frozen, 'created_local': datetime.now().isoformat(),
                                               'development_directories': [str(path.resolve()) for path in args.development],
                                               'holdout_seed_base': 231100000, 'stress_seed_base': 237100000,
                                               'official_calls': 0, 'no_post_holdout_tuning': True})
    for relative in frozen:
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    shutil.copy2(HERE / 'PROTOCOL.md', output / 'PROTOCOL.md')
    scenes = make_validation_scenes()
    experiment.save(output / 'seed_audit.json', prior_seed_audit(scenes, ROOT / 'reports'))
    experiment.save(output / 'scenes_evaluator_only.json', scenes)
    records = []
    audits = []
    for scene in scenes:
        for mode in ('previous', chosen):
            row, trace = evaluate(scene, mode)
            records.append(row)
            name = scene['id'] + '__' + mode + '.json'
            experiment.save(output / 'records' / name, row)
            experiment.save(output / 'traces' / name, trace)
            audits.append(frozen_verifier.audit_trace(trace, row['virtual_seconds']))
            print(scene['phase'], scene['profile'], scene['count'], mode,
                  round(row['seconds_per_source'], 3) if row['success'] else row['failure'], flush=True)
    with patch.object(frozen_verifier, 'SELECTED', {'q4': chosen}):
        summary = frozen_verifier.summarize(records)['q4']
    experiment.save(output / 'summary.json', summary)
    replays = []
    for profile, count in (('random', 10), ('near_origin', 16)):
        scene = next(scene for scene in scenes if scene['profile'] == profile and scene['count'] == count)
        direct = next(row for row in records if row['scene_id'] == scene['id'] and row['mode'] == chosen)
        replay, _trace = evaluate(scene, chosen)
        replays.append({'scene_id': scene['id'], 'passed': replay['success'] and direct['success'] and replay['trace_sha256'] == direct['trace_sha256']})
    experiment.save(output / 'replays.json', replays)
    performance_passed = promotion_gate(summary)
    http = local_http(output, scenes, records, chosen) if performance_passed else []
    verification = {'source_hashes_unchanged': hashes() == frozen, 'old_frozen_files': len(old_manifest),
                    'runs': len(records), 'successes': sum(row['success'] for row in records),
                    'actions_audited': sum(audit['actions'] for audit in audits),
                    'max_clock_error': max(audit['max_clock_error'] for audit in audits),
                    'replays_passed': sum(row['passed'] for row in replays), 'replays_total': len(replays),
                    'http_passed': sum(row['passed'] for row in http), 'http_total': len(http),
                    'performance_gate_passed': performance_passed, 'official_calls': 0,
                    'goal_achieved': False}
    verification['eligible_as_next_offline_candidate'] = (performance_passed and verification['source_hashes_unchanged']
                                                         and all(row['passed'] for row in replays)
                                                         and len(http) == 3 and all(row['passed'] for row in http))
    experiment.save(output / 'verification.json', verification)
    print(json.dumps({'selected': chosen, 'summary': summary, 'verification': verification}, indent=2), flush=True)


if __name__ == '__main__':
    main()
