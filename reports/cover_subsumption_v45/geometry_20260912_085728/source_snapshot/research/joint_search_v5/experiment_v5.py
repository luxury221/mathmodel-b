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
sys.path[:0] = [str(HERE), str(ROOT / 'research' / 'joint_search_v4'), str(ROOT / 'research' / 'offline_validation'),
                str(ROOT / 'research' / 'policy_optimization_v3'), str(ROOT / 'research' / 'q4_optimization_v2')]

import numpy as np
from belief_policy import SweepPreviousPolicy
from experiment import make_scenes, save
from geometry import dual_ring_network, seven_network
from offline_benchmark import OfflineRuleWorld, Source
from plan_geometry import conservative_cover_check, geometry_contains
from policy import JointSearchPolicy
from policy_v5 import ContinuousPolicy, MODES
from deferred_policy import DeferredPolicy, DEFERRED_MODES


def source_hashes():
    folders = (HERE, ROOT / 'research' / 'joint_search_v4', ROOT / 'research' / 'offline_validation',
               ROOT / 'research' / 'policy_optimization_v3', ROOT / 'research' / 'q4_optimization_v2')
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in folders for path in folder.glob('*.py')}


def evaluate(scene, mode, policy_factory=None, world_factory=OfflineRuleWorld):
    problem = scene['problem']
    world = world_factory([Source(**source) for source in scene['sources']], scene['seed'], scene['error_mode'])
    args = (world.port(), problem, seven_network() if problem == 'q3' else dual_ring_network(),
            'E_joint' if problem == 'q3' else 'F_route', 'grid7' if problem == 'q3' else 'dual21')
    if policy_factory is not None:
        policy = policy_factory(*args, mode=mode)
    elif mode == 'previous':
        policy = JointSearchPolicy(*args, mode='adaptive') if problem == 'q3' else SweepPreviousPolicy(*args)
    elif mode in DEFERRED_MODES:
        policy = DeferredPolicy(*args, mode=mode)
    else:
        policy = ContinuousPolicy(*args, mode=mode)
    started = time.perf_counter()
    failure = None
    checks = {'regions': 0, 'clear_certificates': 0, 'optical_events': 0}
    try:
        result = policy.run()
        if abs(result['virtual_seconds'] - world.virtual_seconds) > 1e-6:
            raise ValueError('Policy/world clocks disagree')
        for channel, region in result['snapshots']:
            if channel in world.sources:
                if not geometry_contains(region, world.sources[channel].position):
                    raise ValueError('True source excluded from region')
                checks['regions'] += 1
        for region, centers, kind in result['certificates']:
            if not conservative_cover_check(region, centers):
                raise ValueError('Clearance certificate failed: ' + kind)
            checks['clear_certificates'] += 1
        if world.remaining or set(result['declared_absent']) & set(world.sources):
            raise ValueError('Real source remains or declared absent')
        if any(np.linalg.norm(action['position']) >= 5000 for action in world.trace):
            raise ValueError('Action outside safety envelope')
        for channels, point in getattr(policy, 'clear_event_log', []):
            for channel in channels:
                matching = [action for action in world.trace if action['action'] == 'clear' and action['channel'] == channel
                            and action['position'] == point.tolist() and action['response']['result'] == 'no_target_in_range']
                if not matching:
                    raise ValueError('Optical absence proof lacks real observation')
            checks['optical_events'] += 1
    except Exception as error:
        failure = type(error).__name__ + ': ' + str(error)
    success = failure is None and not world.remaining
    return {'scene_id': scene['id'], 'phase': scene['phase'], 'problem': problem, 'profile': scene['profile'],
            'count': scene['count'], 'seed': scene['seed'], 'mode': mode, 'success': success, 'failure': failure,
            'virtual_seconds': world.virtual_seconds, 'seconds_per_source': world.virtual_seconds / scene['count'] if success else None,
            'cleared_count': scene['count'] - len(world.remaining), 'checks': checks,
            'world_stats': world.stats, 'policy_stats': policy.stats, 'wall_seconds': time.perf_counter() - started,
            'layout_proofs': getattr(getattr(policy, 'patrol', None), 'layout_log', []),
            'trace_sha256': hashlib.sha256(json.dumps(world.trace, sort_keys=True).encode()).hexdigest()}, world.trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--modes', nargs='+', choices=['previous', *MODES, *DEFERRED_MODES], default=['previous', *MODES])
    parser.add_argument('--counts', nargs='+', type=int, default=[13])
    parser.add_argument('--problems', nargs='+', default=['q3', 'q4'])
    parser.add_argument('--base-seed', type=int, default=103100000)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or output.exists():
        raise ValueError('Use a new D-drive output')
    scenes = [scene for scene in make_scenes(args.base_seed, 'development', args.counts) if scene['problem'] in args.problems]
    for scene in scenes:
        scene['id'] = scene['id'].replace('jointv4_', 'jointv5_')
    save(output / 'protocol.json', {'source_hashes': source_hashes(), 'modes': args.modes,
                                  'created_local': datetime.now().isoformat(), 'official_calls': 0,
                                  'metric': 'Case-equal virtual seconds/source; all failures retained.'})
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
            name = scene['id'] + '__' + mode + '.json'
            save(output / 'records' / name, record)
            save(output / 'traces' / name, trace)
            records.append(record)
            print(scene['problem'], scene['profile'], scene['count'], mode,
                  round(record['seconds_per_source'], 3) if record['success'] else record['failure'], flush=True)
    summary = {}
    for problem in args.problems:
        summary[problem] = {}
        for mode in args.modes:
            rows = [record for record in records if record['problem'] == problem and record['mode'] == mode]
            summary[problem][mode] = {'runs': len(rows), 'successes': sum(record['success'] for record in rows),
                                      'mean': statistics.mean(record['seconds_per_source'] for record in rows) if all(record['success'] for record in rows) else None,
                                      'profiles': {profile: statistics.mean(record['seconds_per_source'] for record in rows if record['profile'] == profile)
                                                   for profile in sorted({record['profile'] for record in rows})} if all(record['success'] for record in rows) else {}}
    save(output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
