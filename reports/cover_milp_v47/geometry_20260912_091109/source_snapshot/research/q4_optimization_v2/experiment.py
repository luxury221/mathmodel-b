from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
sys.path[:0] = [str(ROOT / 'research' / folder) for folder in
                ('q4_validation_v1', 'policy_optimization_v3', 'offline_validation', 'interface_validation')]

import numpy as np
from candidate import CandidatePolicy
from candidate_v2 import MODES, Q4OptimizationPolicy
from geometry import dual_ring_network
from offline_benchmark import OfflineRuleWorld, Source
from plan_experiments import make_scene
from plan_geometry import conservative_cover_check, geometry_contains
from plan_policy import PlanPolicy
from run_validation import (
    action_hash,
    file_hash,
    load,
    paired_summary,
    save,
    worst_heading,
)
from run_validation import checked_hashes as previous_hashes
from shapely.errors import GEOSException

ALL_MODES = ('baseline', 'joint', *MODES)
PROFILES = ('random', 'boundary', 'clustered', 'near_origin', 'max_radius', 'adversarial_heading')


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def hashes():
    result = previous_hashes()
    for path in SCRIPT.parent.glob('*.py'):
        result[str(path.relative_to(ROOT))] = file_hash(path)
    return result


def critical_scene(count, seed, phase, repetition):
    generator = np.random.default_rng(seed)
    channels = generator.choice(np.arange(1, 21), count, replace=False)
    rotation = float(generator.uniform(0, 360))
    radii = (0.001, 5.000001, 999.999999, 1000.000001, 1799.999999, 1790.0)
    sources = []
    stations = dual_ring_network()
    for index, channel in enumerate(channels):
        angle = math.radians((rotation + index * 360.0 / count) % 360.0)
        radius = radii[index % len(radii)]
        position = (radius * math.cos(angle), radius * math.sin(angle))
        sources.append({'channel': int(channel), 'position': position, 'radius': 1000.0,
                        'heading_deg': None if index == 0 else worst_heading(position, stations)})
    return {'id': f'q4v2_{phase}_heading_sweep_{count}_{repetition}', 'phase': phase, 'problem': 'q4',
            'count': count, 'profile': 'heading_sweep', 'seed': seed, 'rotation_deg': rotation,
            'error_mode': 'spatial_extreme', 'sources': sources}


def build_scenes():
    scenes = []
    for phase, counts, repetitions, base_seed in (('development', (10, 13, 16), 1, 73100000),
                                                 ('holdout', (10, 12, 14, 16), 2, 76100000)):
        for profile_index, profile in enumerate(PROFILES):
            for count in counts:
                for repetition in range(repetitions):
                    scene = make_scene('q4', count, profile, base_seed + profile_index * 10000 + count * 100 + repetition,
                                       phase, repetition)
                    scene['id'] = 'q4v2_' + scene['id']
                    scenes.append(scene)
    for phase, repetitions, base_seed in (('development', 3, 74100000), ('stress', 8, 78100000)):
        for count in (10, 16):
            for repetition in range(repetitions):
                scenes.append(critical_scene(count, base_seed + count * 100 + repetition, phase, repetition))
    return scenes


def prepare(output):
    if output.exists():
        raise ValueError('Use a new experiment directory')
    scenes = build_scenes()
    prior_paths = [ROOT / 'reports' / 'offline_review' / 'offline_scenes.json']
    prior_paths += [ROOT / 'reports' / folder / 'scenes.json'
                   for folder in ('plan_trials', 'plan_trials_v2', 'policy_optimization_v3')]
    prior_paths.append(ROOT / 'reports' / 'q4_validation_v1' / 'run_20260911_165940' / 'scenes_evaluator_only.json')
    prior_seeds = {scene['seed'] for path in prior_paths for scene in load(path)}
    assert len(scenes) == len({scene['seed'] for scene in scenes}) == 88
    assert not prior_seeds.intersection(scene['seed'] for scene in scenes)
    save(output / 'scenes_evaluator_only.json', scenes)
    save(output / 'protocol.json', {
        'created_local': now(), 'source_hashes': hashes(), 'scenes_sha256': file_hash(output / 'scenes_evaluator_only.json'),
        'prior_scene_hashes': {str(path.relative_to(ROOT)): file_hash(path) for path in prior_paths},
        'phase_counts': {phase: sum(scene['phase'] == phase for scene in scenes) for phase in ('development', 'holdout', 'stress')},
        'development_modes': ALL_MODES, 'worker_timeout_s': 180, 'official_calls': 0,
        'selection_rule': 'All 24 development cases must fully clear; maximum per-case slowdown versus original <=20%; each profile mean <=105% of original. Among eligible modes choose lowest case-level mean seconds/source; modes within 1 second/source of best prefer baseline,joint,scan,tour_scan,outer_scan,probe_scan in that order.',
        'holdout_rule': 'Freeze selection before reading holdout or stress results; compare original, prior joint, and the unique selected mode. No tuning on holdout/stress; retain every failure.',
        'safety': 'Exact original 21 coordinates retained; no early absence shortcut added; negative q4 responses do not cut position regions; original certified clearance fallback remains.',
        'replay_rule': 'Replay the greatest-total-virtual-time successful case and fixed random10 holdout case for each compared mode.',
        'http_rule': 'Selected policy only, local capability-protected MockArena. Four fixed holdout/stress scenes plus boundary and heading stress five-fault recoveries, six total. No official entry point.',
        'statistics': '88 synthetic scenes including 24 development and 64 fresh evaluation scenes; executions/actions/replays are not independent additional samples. Artificial distribution is not official score prediction.',
    })
    print('Frozen 24 development + 48 holdout + 16 stress scenes; 6 development modes; no official calls', flush=True)


def verify(output):
    protocol = load(output / 'protocol.json')
    if protocol['source_hashes'] != hashes() or protocol['scenes_sha256'] != file_hash(output / 'scenes_evaluator_only.json'):
        raise ValueError('Frozen experiment sources or scenes changed')
    return protocol, load(output / 'scenes_evaluator_only.json')


def evaluation_modes(output):
    selection = load(output / 'selection.json')
    if selection['source_hashes'] != hashes():
        raise ValueError('Selected code changed')
    return tuple(dict.fromkeys(('baseline', 'joint', selection['selected'])))


class PhaseCosts:
    def __init__(self, *args, **kwargs):
        self.cost_phase = 'discovery'
        self.phase_seconds = {'discovery': 0.0, 'cleanup': 0.0}
        super().__init__(*args, **kwargs)

    def account(self, *args, **kwargs):
        before = self.virtual_seconds
        super().account(*args, **kwargs)
        self.phase_seconds[self.cost_phase] += self.virtual_seconds - before

    def localize(self, state):
        self.cost_phase = 'cleanup'
        return super().localize(state)


def evaluate(scene, mode):
    world = OfflineRuleWorld([Source(**source) for source in scene['sources']], scene['seed'], scene['error_mode'])
    base_class = PlanPolicy if mode == 'baseline' else CandidatePolicy if mode == 'joint' else Q4OptimizationPolicy
    policy_class = type('ObservedPolicy', (PhaseCosts, base_class), {})
    arguments = (world.port(), 'q4', dual_ring_network(), 'F_route', 'dual21')
    policy = policy_class(*arguments) if mode == 'baseline' else policy_class(*arguments, mode=mode)
    started = time.perf_counter()
    failure = None
    region_checks, cover_checks = 0, 0
    try:
        result = policy.run()
        if abs(result['virtual_seconds'] - world.virtual_seconds) > 1e-6:
            raise ValueError('Independent policy and world clocks disagree')
        for channel, region in result['snapshots']:
            if channel in world.sources:
                if not geometry_contains(region, world.sources[channel].position):
                    raise ValueError('True source excluded from a conservative region')
                region_checks += 1
        for region, centers, kind in result['certificates']:
            if not conservative_cover_check(region, centers):
                raise ValueError(f'Continuous cover certificate failed: {kind}')
            cover_checks += 1
        if set(result['declared_absent']) & set(world.sources) or world.remaining:
            raise ValueError(f'Real source absent or uncleared: {sorted(world.remaining)}')
        if any(np.linalg.norm(action['position']) >= 5000 for action in world.trace):
            raise ValueError('Action outside the established safety envelope')
    except (AssertionError, RuntimeError, ValueError, ArithmeticError, GEOSException) as error:
        failure = f'{type(error).__name__}: {error}'
    cleared = len(world.sources) - len(world.remaining)
    success = failure is None and cleared == scene['count']
    record = {'scene_id': scene['id'], 'phase': scene['phase'], 'profile': scene['profile'], 'seed': scene['seed'],
              'source_count': scene['count'], 'mode': mode, 'network': 'dual21', 'all_cleared': success,
              'cleared_count': cleared, 'failure': failure, 'virtual_seconds': world.virtual_seconds,
              'seconds_per_source': world.virtual_seconds / scene['count'] if success else None,
              'wall_seconds': time.perf_counter() - started, 'action_count': len(world.trace),
              'world_stats': world.stats, 'policy_stats': policy.stats, 'phase_seconds': policy.phase_seconds,
              'snapshots_checked': region_checks, 'certificates_checked': cover_checks,
              'action_sha256': action_hash(world.trace),
              'trace_sha256': hashlib.sha256(json.dumps(world.trace, sort_keys=True, allow_nan=False).encode()).hexdigest(),
              'harness_completed': True}
    return record, world.trace


def record_path(output, scene, mode):
    return output / 'records' / f'{scene["id"]}__{mode}.json'


def worker(output, scene_id):
    _, scenes = verify(output)
    scene = next(scene for scene in scenes if scene['id'] == scene_id)
    modes = ALL_MODES if scene['phase'] == 'development' else evaluation_modes(output)
    for mode in modes:
        path = record_path(output, scene, mode)
        if path.exists():
            raise ValueError('Do not overwrite an existing run')
        record, trace = evaluate(scene, mode)
        trace_path = output / 'traces' / f'{scene_id}__{mode}.json'
        save(trace_path, trace)
        record.update({'trace_file': str(trace_path.relative_to(output)), 'trace_file_sha256': file_hash(trace_path)})
        save(path, record)


def run_phase(output, phase):
    protocol, scenes = verify(output)
    scenes = [scene for scene in scenes if scene['phase'] == phase]
    modes = ALL_MODES if phase == 'development' else evaluation_modes(output)
    marker = output / f'{phase}_started.json'
    if marker.exists():
        raise ValueError('This phase was already started; preserve incomplete evidence')
    save(marker, {'started_local': now(), 'scenes': len(scenes), 'modes': modes})
    for index, scene in enumerate(scenes, 1):
        attempt_path = output / 'attempts' / f'{scene["id"]}.json'
        attempt = {'scene_id': scene['id'], 'phase': phase, 'started_local': now(), 'state': 'running'}
        save(attempt_path, attempt)
        log_path = output / 'worker_logs' / f'{scene["id"]}.log'
        log_path.parent.mkdir(exist_ok=True)
        failure = None
        started = time.perf_counter()
        with log_path.open('w', encoding='utf-8') as stream:
            try:
                completed = subprocess.run([sys.executable, str(SCRIPT), 'worker', '--output', str(output),
                                            '--scene-id', scene['id']], stdout=stream, stderr=subprocess.STDOUT,
                                           timeout=protocol['worker_timeout_s'], check=False)
                if completed.returncode:
                    failure = f'Worker exit {completed.returncode}; see {log_path.name}'
            except subprocess.TimeoutExpired:
                failure = 'Worker exceeded predeclared 180-second wall limit'
        for mode in modes:
            path = record_path(output, scene, mode)
            if not path.exists():
                save(path, {'scene_id': scene['id'], 'phase': phase, 'profile': scene['profile'], 'seed': scene['seed'],
                            'source_count': scene['count'], 'mode': mode, 'network': 'dual21', 'all_cleared': False,
                            'cleared_count': None, 'seconds_per_source': None, 'virtual_seconds': None,
                            'failure': failure or 'Worker produced no result', 'harness_completed': False})
        attempt.update({'state': 'harness_failed' if failure else 'completed', 'error': failure,
                        'finished_local': now(), 'wall_s': time.perf_counter() - started})
        save(attempt_path, attempt)
        rows = [load(record_path(output, scene, mode)) for mode in modes]
        print(f'{phase} {index}/{len(scenes)} {scene["profile"]} n={scene["count"]}: '
              + ' | '.join(f'{row["mode"]}='
                           + (f'{row["seconds_per_source"]:.2f}' if row['all_cleared'] else f'FAIL {row["failure"]}')
                           for row in rows), flush=True)
    verify(output)
    save(output / f'{phase}_complete.json', {'runs': len(scenes) * len(modes), 'completed_local': now()})


def records(output, phases):
    _, scenes = verify(output)
    return [load(record_path(output, scene, mode)) for scene in scenes if scene['phase'] in phases
            for mode in (ALL_MODES if scene['phase'] == 'development' else evaluation_modes(output))]


def select(output):
    verify(output)
    if (output / 'selection.json').exists():
        raise ValueError('Selection already frozen')
    rows = records(output, ('development',))
    scores = []
    for mode in ALL_MODES:
        paired = paired_summary(rows, 'baseline_dual21', f'{mode}_dual21')
        group_pairs = {profile: paired_summary([row for row in rows if row['profile'] == profile],
                                              'baseline_dual21', f'{mode}_dual21')
                       for profile in (*PROFILES, 'heading_sweep')}
        eligible = (paired['candidate_full_clear'] == 24 and paired['reference_full_clear'] == 24
                    and paired.get('worst_slowdown_pct', math.inf) <= 20.0 + 1e-8
                    and all(pair.get('mean_reduction_pct', -math.inf) >= -5.0 - 1e-8 for pair in group_pairs.values()))
        scores.append({'mode': mode, 'eligible': eligible, 'paired': paired, 'profiles': group_pairs})
    eligible = [score for score in scores if score['eligible']]
    if not eligible:
        raise ValueError('Even the original failed the development checks; stop without selecting')
    best_mean = min(score['paired']['candidate_mean_s_per_source'] for score in eligible)
    winner = next(score['mode'] for score in eligible if score['paired']['candidate_mean_s_per_source'] <= best_mean + 1.0)
    save(output / 'selection.json', {'selected': winner, 'created_local': now(), 'source_hashes': hashes(),
                                     'development_only': True, 'scores': scores,
                                     'development_records_sha256': {path.name: file_hash(path)
                                         for path in sorted((output / 'records').glob('q4v2_development*.json'))}})
    print('Development-only selected mode: ' + winner, flush=True)
    for score in scores:
        print(score['mode'], 'eligible=', score['eligible'], 'mean=', score['paired'].get('candidate_mean_s_per_source'),
              'worst_slowdown_pct=', score['paired'].get('worst_slowdown_pct'), flush=True)


def replay(output):
    _, scenes = verify(output)
    rows = records(output, ('holdout', 'stress'))
    results = []
    for mode in evaluation_modes(output):
        successful = [row for row in rows if row['mode'] == mode and row['all_cleared']]
        selected_ids = {'q4v2_holdout_q4_random_10_0'}
        if successful:
            selected_ids.add(max(successful, key=lambda row: (row['virtual_seconds'], row['scene_id']))['scene_id'])
        for scene_id in sorted(selected_ids):
            scene = next(scene for scene in scenes if scene['id'] == scene_id)
            original = load(record_path(output, scene, mode))
            repeated, _ = evaluate(scene, mode)
            checks = {key: repeated.get(key) == original.get(key)
                      for key in ('all_cleared', 'virtual_seconds', 'trace_sha256', 'policy_stats', 'phase_seconds')}
            results.append({'scene_id': scene_id, 'mode': mode, 'checks': checks, 'passed': all(checks.values())})
            save(output / 'replay.json', results)
            print(f'Replay {mode} {scene_id}: {results[-1]["passed"]}', flush=True)
    if not all(row['passed'] for row in results):
        raise AssertionError('Replay mismatch; preserve all evidence')


def report(output):
    rows = records(output, ('development', 'holdout', 'stress'))
    winner = load(output / 'selection.json')['selected']
    groups = {}
    for phase in ('development', 'holdout', 'stress', 'evaluation'):
        subset = [row for row in rows if row['phase'] == phase or (phase == 'evaluation' and row['phase'] != 'development')]
        groups[phase] = {}
        for profile in ('all', *sorted({row['profile'] for row in subset})):
            selected = subset if profile == 'all' else [row for row in subset if row['profile'] == profile]
            groups[phase][profile] = {mode: paired_summary(selected, 'baseline_dual21', f'{mode}_dual21')
                                     for mode in (ALL_MODES if phase == 'development' else evaluation_modes(output))}
    evaluation = [row for row in rows if row['phase'] != 'development']
    diagnostics = {}
    for mode in evaluation_modes(output):
        successful = [row for row in evaluation if row['mode'] == mode and row['all_cleared']]
        if successful:
            diagnostics[mode] = {
                'mean_discovery_s_per_source': statistics.mean(row['phase_seconds']['discovery'] / row['source_count'] for row in successful),
                'mean_cleanup_s_per_source': statistics.mean(row['phase_seconds']['cleanup'] / row['source_count'] for row in successful),
                'mean_move_m': statistics.mean(row['world_stats']['move_meters'] for row in successful),
                'mean_measures': statistics.mean(row['world_stats']['measure_count'] for row in successful),
                'mean_failed_clears': statistics.mean(row['world_stats']['failed_clear_count'] for row in successful),
                'mean_reuse_measures': statistics.mean(row['policy_stats'].get('reuse_measures', 0) for row in successful),
                'mean_bounded_probes': statistics.mean(row['policy_stats'].get('bounded_probes', 0) for row in successful),
            }
    summary = {'created_local': now(), 'official_calls': 0, 'selected': winner, 'unique_scenes': 88,
               'evaluation_scenes': 64, 'runs': len(rows), 'all_cleared_runs': sum(row['all_cleared'] for row in rows),
               'failures': [row for row in rows if not row['all_cleared']], 'paired_results': groups, 'diagnostics': diagnostics}
    save(output / 'summary.json', summary)
    print(json.dumps({'selected': winner, 'runs': len(rows), 'all_cleared': summary['all_cleared_runs'],
                      'evaluation': groups['evaluation']['all'], 'diagnostics': diagnostics}, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description='Independent Q4 optimization; local synthetic tests only')
    parser.add_argument('command', choices=('prepare', 'run', 'worker', 'select', 'replay', 'report'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--phase', choices=('development', 'holdout', 'stress'))
    parser.add_argument('--scene-id')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.drive.upper() != 'D:' or not output.is_relative_to(ROOT / 'reports' / 'q4_optimization_v2'):
        raise ValueError('All evidence must remain in the dedicated D-drive directory')
    if args.command == 'run':
        if not args.phase:
            parser.error('--phase is required for run')
        run_phase(output, args.phase)
    elif args.command == 'worker':
        worker(output, args.scene_id)
    else:
        globals()[args.command](output)


if __name__ == '__main__':
    main()
