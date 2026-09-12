from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from geometry import dual_ring_network, seven_network, triangular_network
from offline_benchmark import (
    OfflineRuleWorld,
    Source,
    generate_scene,
    protocol_checks,
    run_policy,
)
from plan_geometry import conservative_cover_check, geometry_contains
from plan_policy import VARIANTS, PlanPolicy
from shapely.errors import GEOSException
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'reports' / 'plan_trials_v2'
SCRIPT_DIR = Path(__file__).resolve().parent


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save(name, content):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / name).write_text(
        json.dumps(content, ensure_ascii=False, indent=2, default=json_default), encoding='utf-8',
    )


def load(name):
    return json.loads((OUTPUT / name).read_text(encoding='utf-8'))


def source_hashes():
    names = ('geometry.py', 'offline_benchmark.py', 'plan_geometry.py', 'plan_policy.py', 'plan_experiments.py')
    return {name: hashlib.sha256((SCRIPT_DIR / name).read_bytes()).hexdigest() for name in names}


def make_scene(problem, count, profile, seed, phase, repetition):
    generator_profile = 'random' if profile == 'max_radius' else 'boundary' if profile == 'adversarial_heading' else profile
    sources, error_mode = generate_scene(problem, count, generator_profile, seed)
    if profile == 'max_radius':
        sources = [Source(source.channel, source.position, 1500.0, source.heading_deg) for source in sources]
        error_mode = 'spatial_extreme'
    if profile == 'adversarial_heading':
        stations = dual_ring_network()
        headings = np.arange(720) * math.pi / 360
        directions = np.column_stack((np.cos(headings), np.sin(headings)))
        revised = []
        for index, source in enumerate(sources):
            vectors = stations - np.asarray(source.position)
            in_range = np.linalg.norm(vectors, axis=1) <= 1000
            visibility = (directions @ vectors.T >= 0) & in_range
            selected_heading = float(np.argmin(visibility.sum(axis=1)) / 2)
            heading = selected_heading if problem == 'q4' and index else None
            revised.append(Source(source.channel, source.position, 1000.0, heading))
        sources = revised
        error_mode = 'spatial_extreme'
    return {
        'id': f'{phase}_{problem}_{profile}_{count}_{repetition}', 'phase': phase,
        'problem': problem, 'count': count, 'profile': profile, 'seed': seed,
        'error_mode': error_mode, 'sources': [source.__dict__ for source in sources],
    }


def prepare():
    if (OUTPUT / 'protocol.json').exists():
        protocol = load('protocol.json')
        if protocol['source_hashes'] != source_hashes():
            raise RuntimeError('Code changed after protocol freeze; do not silently reuse earlier results')
        return protocol
    scenes = []
    specifications = [
        ('development', (10, 13, 16), 1, ('random', 'boundary', 'clustered', 'near_origin'), 910000),
        ('holdout', tuple(range(10, 17)), 2, ('random', 'boundary', 'clustered', 'near_origin'), 6900000),
        ('stress', (10, 16), 3, ('max_radius', 'adversarial_heading'), 8900000),
    ]
    for phase, counts, repetitions, profiles, base_seed in specifications:
        for problem_index, problem in enumerate(('q3', 'q4')):
            for profile_index, profile in enumerate(profiles):
                for count in counts:
                    for repetition in range(repetitions):
                        seed = base_seed + problem_index * 100000 + profile_index * 1000 + count * 10 + repetition
                        scenes.append(make_scene(problem, count, profile, seed, phase, repetition))
    save('scenes.json', scenes)
    protocol = {
        'date': '2026-09-11', 'kind': 'offline_plan_ablation_and_frozen_holdout',
        'official_simulator_calls': 0, 'source_hashes': source_hashes(),
        'scene_sha256': hashlib.sha256((OUTPUT / 'scenes.json').read_bytes()).hexdigest(),
        'scenario_counts': {phase: sum(scene['phase'] == phase for scene in scenes) for phase in ('development', 'holdout', 'stress')},
        'variants': VARIANTS,
        'selection_rule': 'Among variants clearing every development scene, select lowest mean seconds/source independently for q3 and q4; ties within 1 second/source prefer the lower stage.',
        'holdout_rule': 'No tuning after selection freeze. Compare A_cover, frozen winner, F_route, and legacy best; also compare grid25 A_cover and frozen q4 winner.',
        'legacy_rule': 'Previous-turn q3 active 7-node prototype and q4 cover-only 21-node prototype; fixed before these runs.',
        'performance_scope': 'Synthetic distributions only; no official score, HTTP cost, or exact continuous minimax optimality claim.',
        'smoke_test_disclosure': 'Before freezing, one separate random scene per problem with seed 810001 checked engineering behavior; not included in development or holdout.',
        'repair_disclosure': 'v1 stopped at holdout task 350 on a GEOS topology error. Its inputs, logs, and source snapshot remain in reports/plan_trials. v2 builds fresh convex constraints from their vertex hull, catches GEOS failures as failed runs, and uses entirely new holdout/stress seeds. Development scenes are intentionally reused.',
        'protocol_checks': protocol_checks(),
    }
    save('protocol.json', protocol)
    return protocol


def execute(scene, variant, network, keep_failure=True):
    sources = [Source(**source) for source in scene['sources']]
    stations = {
        'grid7': seven_network,
        'dual21': dual_ring_network,
        'grid25': lambda: triangular_network()[0],
    }[network]()
    world = OfflineRuleWorld(sources, scene['seed'], scene['error_mode'])
    started = time.perf_counter()
    failure = None
    statistics = {}
    snapshots_checked = 0
    certificates_checked = 0
    try:
        if variant == 'legacy_best':
            result = run_policy(world.port(), stations, scene['problem'] == 'q3')
            snapshots = [(channel, Polygon(region)) for channel, region in result['region_snapshots']]
            certificates = []
            declared_absent = result['declared_absent']
        else:
            result = PlanPolicy(world.port(), scene['problem'], stations, variant, network).run()
            snapshots = result['snapshots']
            certificates = result['certificates']
            declared_absent = result['declared_absent']
            statistics = result['stats']
            assert abs(world.virtual_seconds - result['virtual_seconds']) < 1e-6
        for channel, region in snapshots:
            if channel in world.sources:
                assert geometry_contains(region, world.sources[channel].position), f'True source excluded: channel {channel}'
                snapshots_checked += 1
        for region, centers, kind in certificates:
            assert conservative_cover_check(region, centers), f'Continuous cover rejected: {kind}'
            certificates_checked += 1
        assert not set(declared_absent) & set(world.sources)
        assert not world.remaining, f'Uncleared channels: {sorted(world.remaining)}'
        assert all(np.linalg.norm(action['position']) < 5000 for action in world.trace)
    except (AssertionError, RuntimeError, ValueError, ArithmeticError, GEOSException) as exception:
        failure = f'{type(exception).__name__}: {exception}'
        if keep_failure:
            save(f'failure_{scene["id"]}_{network}_{variant}.json', {'scene': scene, 'failure': failure, 'trace': world.trace})
    cleared = len(sources) - len(world.remaining)
    record = {
        'scene_id': scene['id'], 'phase': scene['phase'], 'problem': scene['problem'],
        'profile': scene['profile'], 'source_count': len(sources), 'seed': scene['seed'],
        'variant': variant, 'network': network, 'cleared_count': cleared,
        'all_cleared': cleared == len(sources) and failure is None,
        'seconds_per_source': world.virtual_seconds / cleared if cleared else None,
        'virtual_seconds': world.virtual_seconds, 'wall_seconds': time.perf_counter() - started,
        'action_count': len(world.trace), 'snapshots_checked': snapshots_checked,
        'certificates_checked': certificates_checked, 'failure': failure,
        **world.stats, **statistics,
    }
    return record, world.trace


def tasks_for_phase(phase, scenes, selection=None):
    tasks = []
    for scene in scenes:
        if scene['phase'] != phase:
            continue
        network = 'grid7' if scene['problem'] == 'q3' else 'dual21'
        if phase == 'development':
            variants = list(VARIANTS)
        elif phase == 'holdout':
            variants = sorted({'A_cover', selection[scene['problem']], 'F_route', 'legacy_best'})
        else:
            variants = sorted({'A_cover', selection[scene['problem']]})
        tasks.extend((scene, variant, network) for variant in variants)
        if scene['problem'] == 'q4' and phase != 'stress':
            extra_variants = ('A_cover', 'F_route') if phase == 'development' else sorted({'A_cover', selection['q4']})
            tasks.extend((scene, variant, 'grid25') for variant in extra_variants)
    return tasks


def run_phase(phase):
    prepare()
    filename = f'{phase}_runs.json'
    if (OUTPUT / filename).exists():
        records = load(filename)
        if (OUTPUT / f'{phase}_complete.json').exists():
            print(phase, 'already complete:', len(records), 'runs', flush=True)
            return records
    scenes = load('scenes.json')
    selection = load('selection.json')['selected'] if phase != 'development' else None
    tasks = tasks_for_phase(phase, scenes, selection)
    records = []
    worst = {}
    started = time.perf_counter()
    for index, (scene, variant, network) in enumerate(tasks):
        record, trace = execute(scene, variant, network)
        records.append(record)
        key = f'{scene["problem"]}_{network}_{variant}'
        if key not in worst or record['seconds_per_source'] > worst[key]['record']['seconds_per_source']:
            worst[key] = {'record': record, 'trace': trace}
        if record['failure']:
            print('FAIL', scene['id'], network, variant, record['failure'], flush=True)
        if (index + 1) % 20 == 0 or index + 1 == len(tasks):
            save(filename, records)
            print(phase, f'{index + 1}/{len(tasks)}', 'runs;', round(time.perf_counter() - started, 1), 'seconds', flush=True)
    save(f'{phase}_worst_traces.json', worst)
    save(f'{phase}_complete.json', {
        'runs': len(records), 'all_cleared': sum(record['all_cleared'] for record in records),
        'elapsed_seconds': time.perf_counter() - started,
    })
    return records


def select():
    records = load('development_runs.json')
    selected = {}
    evidence = {}
    for problem, network in (('q3', 'grid7'), ('q4', 'dual21')):
        relevant = pd.DataFrame([record for record in records if record['problem'] == problem and record['network'] == network])
        summary = relevant.groupby('variant').agg(
            runs=('all_cleared', 'size'), successes=('all_cleared', 'sum'), mean=('seconds_per_source', 'mean'),
        )
        eligible = summary[summary.runs == summary.successes]
        if eligible.empty:
            raise RuntimeError('No safe development configuration; holdout must not begin')
        best_mean = float(eligible['mean'].min())
        tied = eligible[eligible['mean'] <= best_mean + 1.0]
        selected[problem] = min(tied.index, key=lambda variant: VARIANTS[variant])
        evidence[problem] = summary.reset_index().to_dict('records')
    result = {'selected': selected, 'development_evidence': evidence,
              'frozen_before_holdout': True, 'source_hashes': source_hashes()}
    if (OUTPUT / 'selection.json').exists() and load('selection.json') != result:
        raise RuntimeError('Attempt to change an already-frozen selection')
    save('selection.json', result)
    print('Frozen selection:', selected, flush=True)
    return result


def summarize():
    records = []
    for phase in ('development', 'holdout', 'stress'):
        if (OUTPUT / f'{phase}_runs.json').exists():
            records.extend(load(f'{phase}_runs.json'))
    frame = pd.DataFrame(records)
    frame.to_csv(OUTPUT / 'all_runs.csv', index=False, encoding='utf-8-sig')
    grouped = frame.groupby(['phase', 'problem', 'network', 'variant']).agg(
        runs=('scene_id', 'size'), successes=('all_cleared', 'sum'),
        mean_seconds_per_source=('seconds_per_source', 'mean'),
        median_seconds_per_source=('seconds_per_source', 'median'),
        p90_seconds_per_source=('seconds_per_source', lambda values: values.quantile(0.9)),
        worst_seconds_per_source=('seconds_per_source', 'max'),
        mean_move_meters=('move_meters', 'mean'), mean_measures=('measure_count', 'mean'),
        max_wall_seconds=('wall_seconds', 'max'), max_actions=('action_count', 'max'),
    ).reset_index()
    grouped.to_csv(OUTPUT / 'summary.csv', index=False, encoding='utf-8-sig')
    stratified = frame.groupby(['phase', 'problem', 'network', 'variant', 'profile']).agg(
        runs=('scene_id', 'size'), successes=('all_cleared', 'sum'),
        mean_seconds_per_source=('seconds_per_source', 'mean'),
    ).reset_index()
    stratified.to_csv(OUTPUT / 'by_profile.csv', index=False, encoding='utf-8-sig')
    paired = []
    selection = load('selection.json')['selected']
    for problem, network in (('q3', 'grid7'), ('q4', 'dual21')):
        group = frame[(frame.phase == 'holdout') & (frame.problem == problem) & (frame.network == network)]
        chosen = group[group.variant == selection[problem]].set_index('scene_id').seconds_per_source
        for baseline_name in ('A_cover', 'legacy_best'):
            baseline = group[group.variant == baseline_name].set_index('scene_id').seconds_per_source
            differences = baseline - chosen
            paired.append({
                'problem': problem, 'network': network, 'selected': selection[problem], 'baseline': baseline_name,
                'paired_cases': len(differences), 'faster_cases': int((differences > 1e-6).sum()),
                'slower_cases': int((differences < -1e-6).sum()),
                'mean_saved_seconds_per_source': float(differences.mean()),
                'relative_mean_reduction': float(1 - chosen.mean() / baseline.mean()),
            })
    result = {
        'official_simulator_calls': 0, 'unique_scenarios': frame.scene_id.nunique(),
        'run_count': len(frame), 'all_cleared_runs': int(frame.all_cleared.sum()),
        'feasible_region_checks': int(frame.snapshots_checked.sum()),
        'continuous_cover_checks': int(frame.certificates_checked.sum()),
        'summary': grouped.to_dict('records'), 'paired_holdout': paired,
        'failures': frame[~frame.all_cleared].to_dict('records'),
        'selected': selection,
        'limitations': [
            'Sampled minimax and hypothesis-bank scores are approximations, not continuous global optima.',
            'Conservative geometry and finite coverage, not hypothesis survival, certify safety.',
            'No official simulator, API, hidden cases, formal tests, or official latency evaluated.',
            'No two-step rollout/POMDP or globally optimized detection-node placement implemented.',
            'Near-origin and clustered q4 fixtures are all-directional stress cases.',
            'Adversarial heading fixtures intentionally target the dual21 geometry; not an official distribution.',
        ],
    }
    save('results.json', result)
    print(grouped.to_string(index=False), flush=True)
    return result


def replay():
    prepare()
    scenes = {scene['id']: scene for scene in load('scenes.json')}
    checked = 0
    for phase in ('development', 'holdout', 'stress'):
        records = load(f'{phase}_runs.json')
        for original in records:
            repeated, _trace = execute(scenes[original['scene_id']], original['variant'], original['network'], keep_failure=False)
            for name, value in original.items():
                if name != 'wall_seconds':
                    assert repeated[name] == value, (original['scene_id'], original['variant'], name)
            checked += 1
        print('Replayed', phase, 'cumulative', checked, flush=True)
    result = {'replayed_runs': checked, 'all_non_wall_metrics_exact': True, 'official_simulator_calls': 0}
    save('replay.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['prepare', 'development', 'holdout', 'stress', 'summarize', 'replay', 'all'], default='all')
    arguments = parser.parse_args()
    prepare()
    if arguments.phase in ('all', 'development'):
        run_phase('development')
        select()
    if arguments.phase in ('all', 'holdout'):
        run_phase('holdout')
    if arguments.phase in ('all', 'stress'):
        run_phase('stress')
    if arguments.phase in ('all', 'summarize'):
        result = summarize()
        if result['failures']:
            raise SystemExit(1)
    if arguments.phase == 'replay':
        replay()


if __name__ == '__main__':
    main()
