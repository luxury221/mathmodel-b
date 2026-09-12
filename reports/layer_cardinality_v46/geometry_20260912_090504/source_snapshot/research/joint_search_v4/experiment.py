from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(ROOT / 'research' / 'offline_validation'),
                str(ROOT / 'research' / 'policy_optimization_v3'), str(ROOT / 'research' / 'q4_optimization_v2')]

import numpy as np
from candidate import CandidatePolicy
from candidate_v2 import Q4OptimizationPolicy
from belief_policy import BeliefPreviousPolicy, SweepPreviousPolicy
from geometry import dual_ring_network, seven_network
from offline_benchmark import OfflineRuleWorld, Source
from plan_experiments import make_scene
from plan_geometry import conservative_cover_check, geometry_contains
from plan_policy import PlanPolicy
from policy import JointSearchPolicy, MODES


PROFILES = ('random', 'boundary', 'clustered', 'near_origin', 'max_radius', 'adversarial_heading')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def hashes():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in (HERE, ROOT / 'research' / 'offline_validation', ROOT / 'research' / 'policy_optimization_v3',
                           ROOT / 'research' / 'q4_optimization_v2') for path in folder.glob('*.py')}


def make_scenes(base_seed=83100000, phase='development', counts=(10, 13, 16)):
    scenes = []
    for problem_index, problem in enumerate(('q3', 'q4')):
        for profile_index, profile in enumerate(PROFILES):
            for count in counts:
                seed = base_seed + problem_index * 1000000 + profile_index * 10000 + count * 100
                scene = make_scene(problem, count, profile, seed, phase, 0)
                scene['id'] = 'jointv4_' + scene['id']
                scenes.append(scene)
    return scenes


def evaluate(scene, mode):
    world = OfflineRuleWorld([Source(**source) for source in scene['sources']], scene['seed'], scene['error_mode'])
    problem = scene['problem']
    arguments = (world.port(), problem, seven_network() if problem == 'q3' else dual_ring_network(),
                 'E_joint' if problem == 'q3' else 'F_route', 'grid7' if problem == 'q3' else 'dual21')
    if mode == 'baseline':
        policy = PlanPolicy(*arguments)
    elif mode == 'previous':
        policy = CandidatePolicy(*arguments, mode='combined') if problem == 'q3' else Q4OptimizationPolicy(*arguments, mode='probe_scan')
    elif mode == 'belief_previous':
        policy = CandidatePolicy(*arguments, mode='combined') if problem == 'q3' else BeliefPreviousPolicy(*arguments)
    elif mode == 'sweep_previous':
        policy = CandidatePolicy(*arguments, mode='combined') if problem == 'q3' else SweepPreviousPolicy(*arguments)
    else:
        policy = JointSearchPolicy(*arguments, mode=mode)
    started = time.perf_counter()
    failure = None
    checks = {'regions': 0, 'covers': 0, 'absence_events': 0}
    try:
        result = policy.run()
        if abs(world.virtual_seconds - result['virtual_seconds']) > 1e-6:
            raise ValueError('Independent clocks disagree')
        for channel, region in result['snapshots']:
            if channel in world.sources:
                if not geometry_contains(region, world.sources[channel].position):
                    raise ValueError('True source excluded')
                checks['regions'] += 1
        for region, centers, kind in result['certificates']:
            if not conservative_cover_check(region, centers):
                raise ValueError('Invalid clearance certificate: ' + kind)
            checks['covers'] += 1
        if world.remaining or set(result['declared_absent']) & set(world.sources):
            raise ValueError('Sources remain or are declared absent')
        if any(np.linalg.norm(action['position']) >= 5000 for action in world.trace):
            raise ValueError('Outside tested movement envelope')
        checks['absence_events'] = len(getattr(policy, 'coverage_events', []))
    except Exception as error:
        failure = type(error).__name__ + ': ' + str(error)
    success = failure is None and not world.remaining
    return {'scene_id': scene['id'], 'problem': problem, 'profile': scene['profile'], 'phase': scene['phase'],
            'seed': scene['seed'], 'count': scene['count'], 'mode': mode, 'success': success, 'failure': failure,
            'cleared_count': scene['count'] - len(world.remaining), 'virtual_seconds': world.virtual_seconds,
            'seconds_per_source': world.virtual_seconds / scene['count'] if success else None,
            'world_stats': world.stats, 'policy_stats': policy.stats, 'checks': checks,
            'wall_seconds': time.perf_counter() - started,
            'trace_sha256': hashlib.sha256(json.dumps(world.trace, sort_keys=True).encode()).hexdigest()}, world.trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--modes', nargs='+', default=['baseline', 'previous', *MODES])
    parser.add_argument('--problems', nargs='+', default=['q3', 'q4'])
    parser.add_argument('--counts', nargs='+', type=int, default=[10, 13, 16])
    parser.add_argument('--base-seed', type=int, default=83100000)
    parser.add_argument('--phase', default='development')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a new output directory on D drive')
    scenes = [scene for scene in make_scenes(args.base_seed, args.phase, args.counts) if scene['problem'] in args.problems]
    save(output / 'protocol.json', {'created_at': datetime.now().isoformat(), 'source_hashes': hashes(),
                                  'modes': args.modes, 'phase': args.phase, 'official_calls': 0,
                                  'metric': 'Case-equal mean of total virtual seconds / source count; all cases must succeed.'})
    save(output / 'scenes_evaluator_only.json', scenes)
    for path in HERE.glob('*'):
        if path.is_file():
            destination = output / 'source_snapshot' / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    records = []
    for scene in scenes:
        for mode in args.modes:
            record, trace = evaluate(scene, mode)
            records.append(record)
            name = scene['id'] + '__' + mode + '.json'
            save(output / 'records' / name, record)
            save(output / 'traces' / name, trace)
            print(scene['problem'], scene['profile'], scene['count'], mode,
                  round(record['seconds_per_source'], 3) if record['success'] else record['failure'], flush=True)
    summary = {}
    for problem in args.problems:
        summary[problem] = {}
        for mode in args.modes:
            selected = [record for record in records if record['mode'] == mode and record['problem'] == problem]
            valid = [record for record in selected if record['success']]
            summary[problem][mode] = {'runs': len(selected), 'successes': len(valid),
                                      'mean': statistics.mean(record['seconds_per_source'] for record in valid) if len(valid) == len(selected) else None,
                                      'profiles': {profile: statistics.mean(record['seconds_per_source'] for record in selected if record['profile'] == profile)
                                                   for profile in PROFILES} if len(valid) == len(selected) else {}}
    save(output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
