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
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'research' / 'policy_optimization_v3'),
                str(ROOT / 'research' / 'interface_validation'), str(ROOT / 'research' / 'practice_baseline')]

import numpy as np
from baseline import verify_hashes
from bootstrap import OfflineRuleWorld, Source, client_module, session_module, storage
from experiments import evaluate, make_scene
from experiments import verify as verify_v3
from geometry import dual_ring_network
from mock_server import Fault, MockArena

from b2026_candidate.validation import require_validation, source_hashes

VARIANTS = (('baseline', 'dual21'), ('joint', 'dual21'), ('baseline', 'grid25'))
PROFILES = ('random', 'boundary', 'clustered', 'near_origin', 'max_radius', 'adversarial_heading')
OFFSETS = (0.0, 0.0000001, 22.4999999, 22.5, 22.5000001, 30.0, 45.0, 359.9999999)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def action_hash(trace):
    actions = [{key: action[key] for key in ('action', 'channel', 'position', 'response')} for action in trace]
    return hashlib.sha256(json.dumps(actions, sort_keys=True, allow_nan=False).encode()).hexdigest()


def checked_hashes():
    verify_hashes()
    verify_v3()
    require_validation()
    hashes = source_hashes()
    paths = list(SCRIPT.parent.glob('*.py'))
    paths += [ROOT / 'research' / 'offline_validation' / name
              for name in load(ROOT / 'reports' / 'plan_trials_v2' / 'protocol.json')['source_hashes']]
    for path in paths:
        hashes[str(path.relative_to(ROOT))] = file_hash(path)
    return hashes


def worst_heading(position, stations):
    vectors = stations - np.asarray(position)
    vectors = vectors[np.linalg.norm(vectors, axis=1) <= 1000.0]
    if not len(vectors):
        return 0.0
    bearings = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))
    edges = np.mod(np.r_[bearings - 90.0, bearings + 90.0], 360.0)
    headings = np.unique(np.mod(np.r_[np.arange(720) / 2, edges, edges - 1e-7, edges + 1e-7], 360.0))
    directions = np.column_stack((np.cos(np.radians(headings)), np.sin(np.radians(headings))))
    counts = (directions @ vectors.T >= 0.0).sum(axis=1)
    return float(headings[int(np.argmin(counts))])


def build_scenes():
    scenes = []
    for profile_index, profile in enumerate(PROFILES):
        for count in (10, 12, 14, 16):
            for repetition in range(2):
                seed = 63100000 + profile_index * 10000 + count * 100 + repetition
                scenes.append(make_scene('q4', count, profile, seed, 'q4v1', repetition))
    stations = dual_ring_network()
    radii = (0.001, 5.000001, 999.999999, 1000.000001, 1799.999999, 1790.0)
    for count in (10, 16):
        for repetition, offset in enumerate(OFFSETS):
            seed = 64100000 + count * 100 + repetition
            generator = np.random.default_rng(seed)
            channels = generator.choice(np.arange(1, 21), count, replace=False)
            sources = []
            for index, channel in enumerate(channels):
                angle = math.radians((offset + index * 360.0 / count) % 360.0)
                radius = radii[index % len(radii)]
                position = (radius * math.cos(angle), radius * math.sin(angle))
                sources.append({'channel': int(channel), 'position': position, 'radius': 1000.0,
                                'heading_deg': None if index == 0 else worst_heading(position, stations)})
            scenes.append({'id': f'q4v1_q4_heading_sweep_{count}_{repetition}', 'phase': 'q4v1',
                           'problem': 'q4', 'count': count, 'profile': 'heading_sweep', 'seed': seed,
                           'offset_deg': offset, 'error_mode': 'spatial_extreme', 'sources': sources})
    return scenes


def variant_name(mode, network):
    return f'{mode}_{network}'


def record_path(output, scene, mode, network):
    return output / 'records' / f'{scene["id"]}_{variant_name(mode, network)}.json'


def prepare(output):
    if output.exists():
        raise ValueError('Use a new output directory; previous evidence is immutable')
    scenes = build_scenes()
    previous_seeds = set()
    previous_paths = [ROOT / 'reports' / 'offline_review' / 'offline_scenes.json']
    previous_paths += [ROOT / 'reports' / name / 'scenes.json'
                      for name in ('plan_trials', 'plan_trials_v2', 'policy_optimization_v3')]
    for path in previous_paths:
        previous_seeds.update(scene['seed'] for scene in load(path))
    assert len(scenes) == len({scene['seed'] for scene in scenes}) == 64
    assert not previous_seeds.intersection(scene['seed'] for scene in scenes)
    hashes = checked_hashes()
    save(output / 'scenes_evaluator_only.json', scenes)
    save(output / 'protocol.json', {
        'created_local': now(), 'source_hashes': hashes, 'scenes_sha256': file_hash(output / 'scenes_evaluator_only.json'),
        'previous_scene_hashes': {str(path.relative_to(ROOT)): file_hash(path) for path in previous_paths},
        'fresh_scenes': 64, 'variants': VARIANTS, 'planned_offline_runs': 192,
        'profiles': {profile: sum(scene['profile'] == profile for scene in scenes) for profile in (*PROFILES, 'heading_sweep')},
        'scene_worker_timeout_s': 120, 'official_calls': 0,
        'scope': 'Local synthetic validation only; no simulator login, official practice, or formal requests.',
        'truth_boundary': 'Only evaluator constructs and checks sources; every policy receives only an observation port.',
        'selection_rule': 'No tuning or automatic deployment; frozen q4 original, joint candidate, and original grid25 backup.',
        'analysis_rule': 'Case-level seconds/source; paired comparisons only for full-clear pairs, all failures retained. Report each profile separately.',
        'heading_rule': 'Fixed position scales and rotations; evaluator chooses least-visible heading from a finite grid and half-plane endpoints.',
        'replay_rule': 'For each of three variants replay the slowest successful case and random10 repetition0; duplicates removed.',
        'http_rule': 'Original F_route only: four predeclared scenes on both networks, plus two five-fault recovery runs; candidate q4 not connected.',
    })
    print(f'Frozen: {len(scenes)} fresh scenes, {len(scenes) * len(VARIANTS)} offline runs; official calls=0', flush=True)


def verify(output):
    protocol = load(output / 'protocol.json')
    if protocol['source_hashes'] != checked_hashes():
        raise ValueError('Source changed after freeze')
    if protocol['scenes_sha256'] != file_hash(output / 'scenes_evaluator_only.json'):
        raise ValueError('Scene set changed after freeze')
    return protocol, load(output / 'scenes_evaluator_only.json')


def worker(output, scene_id):
    _, scenes = verify(output)
    scene = next(scene for scene in scenes if scene['id'] == scene_id)
    for mode, network in VARIANTS:
        path = record_path(output, scene, mode, network)
        if path.exists():
            raise ValueError('Refusing to overwrite an evaluated run')
        record, trace = evaluate(scene, mode, network)
        trace_path = output / 'traces' / f'{scene_id}_{variant_name(mode, network)}.json'
        save(trace_path, trace)
        record.update({'action_sha256': action_hash(trace), 'trace_file': str(trace_path.relative_to(output)),
                       'trace_file_sha256': file_hash(trace_path), 'harness_completed': True})
        save(path, record)


def offline(output):
    protocol, scenes = verify(output)
    (output / 'records').mkdir(exist_ok=False)
    for index, scene in enumerate(scenes, 1):
        log_path = output / 'worker_logs' / f'{scene["id"]}.log'
        log_path.parent.mkdir(exist_ok=True)
        status = {'scene_id': scene['id'], 'started_local': now(), 'state': 'running'}
        status_path = output / 'attempts' / f'{scene["id"]}.json'
        save(status_path, status)
        failure = None
        started = time.perf_counter()
        with log_path.open('w', encoding='utf-8') as stream:
            try:
                result = subprocess.run([sys.executable, str(SCRIPT), 'worker', '--output', str(output),
                                         '--scene-id', scene['id']], stdout=stream, stderr=subprocess.STDOUT,
                                        timeout=protocol['scene_worker_timeout_s'], check=False)
                if result.returncode:
                    failure = f'Worker return code {result.returncode}; see {log_path.name}'
            except subprocess.TimeoutExpired:
                failure = 'Scene worker exceeded predeclared 120-second wall limit'
        for mode, network in VARIANTS:
            path = record_path(output, scene, mode, network)
            if not path.exists():
                save(path, {'scene_id': scene['id'], 'profile': scene['profile'], 'seed': scene['seed'],
                            'source_count': scene['count'], 'mode': mode, 'network': network,
                            'all_cleared': False, 'cleared_count': None, 'seconds_per_source': None,
                            'virtual_seconds': None, 'failure': failure or 'Worker produced no result',
                            'harness_completed': False})
        status.update({'state': 'harness_failed' if failure else 'completed', 'error': failure,
                       'finished_local': now(), 'wall_s': time.perf_counter() - started})
        save(status_path, status)
        records = [load(record_path(output, scene, mode, network)) for mode, network in VARIANTS]
        print(f'{index:02d}/64 {scene["profile"]} n={scene["count"]}: '
              + ' | '.join(f'{row["mode"]}/{row["network"]}: '
                           + (f'{row["seconds_per_source"]:.2f}s/source' if row['all_cleared'] else f'FAIL {row["failure"]}')
                           for row in records), flush=True)
    verify(output)
    save(output / 'offline_complete.json', {'completed_runs': 192, 'completed_local': now(), 'official_calls': 0})


def all_records(output):
    _, scenes = verify(output)
    return [load(record_path(output, scene, mode, network)) for scene in scenes for mode, network in VARIANTS]


def paired_summary(rows, reference, candidate):
    original = {row['scene_id']: row for row in rows if variant_name(row['mode'], row['network']) == reference}
    revised = {row['scene_id']: row for row in rows if variant_name(row['mode'], row['network']) == candidate}
    scene_ids = sorted(set(original) & set(revised))
    successful = [scene_id for scene_id in scene_ids if original[scene_id]['all_cleared'] and revised[scene_id]['all_cleared']]
    result = {'cases': len(scene_ids), 'both_full_clear': len(successful),
              'reference_full_clear': sum(original[scene_id]['all_cleared'] for scene_id in scene_ids),
              'candidate_full_clear': sum(revised[scene_id]['all_cleared'] for scene_id in scene_ids)}
    if not successful:
        return result
    original_values = [original[scene_id]['seconds_per_source'] for scene_id in successful]
    revised_values = [revised[scene_id]['seconds_per_source'] for scene_id in successful]
    differences = [first - second for first, second in zip(original_values, revised_values, strict=True)]
    result.update({'reference_mean_s_per_source': statistics.mean(original_values),
                   'candidate_mean_s_per_source': statistics.mean(revised_values),
                   'mean_reduction_pct': 100 * (1 - statistics.mean(revised_values) / statistics.mean(original_values)),
                   'median_paired_s_saved': statistics.median(differences),
                   'wins': sum(value > 1e-6 for value in differences),
                   'ties': sum(abs(value) <= 1e-6 for value in differences),
                   'losses': sum(value < -1e-6 for value in differences),
                   'worst_slowdown_pct': max(100 * (second / first - 1) for first, second
                                             in zip(original_values, revised_values, strict=True))})
    return result


def replay(output):
    _, scenes = verify(output)
    rows = all_records(output)
    results = []
    for mode, network in VARIANTS:
        group = [row for row in rows if row['mode'] == mode and row['network'] == network and row['all_cleared']]
        selected = {'q4v1_q4_random_10_0'}
        if group:
            selected.add(max(group, key=lambda row: (row['virtual_seconds'], row['scene_id']))['scene_id'])
        for scene_id in sorted(selected):
            scene = next(scene for scene in scenes if scene['id'] == scene_id)
            original = load(record_path(output, scene, mode, network))
            record, trace = evaluate(scene, mode, network)
            checks = {'same_clearance': record['all_cleared'] == original['all_cleared'],
                      'same_complete_trace': record['trace_sha256'] == original.get('trace_sha256'),
                      'same_virtual_time': record['virtual_seconds'] == original['virtual_seconds'],
                      'same_policy_stats': record['policy_stats'] == original.get('policy_stats'),
                      'same_actions': action_hash(trace) == original.get('action_sha256')}
            results.append({'scene_id': scene_id, 'mode': mode, 'network': network,
                            'checks': checks, 'passed': all(checks.values())})
            save(output / 'replay.json', results)
            print(f'Replay {scene_id} {mode}/{network}: {all(checks.values())}', flush=True)
    if not all(row['passed'] for row in results):
        raise AssertionError('Deterministic replay mismatch; keep evidence')


def http(output):
    _, scenes = verify(output)
    selected_ids = ('q4v1_q4_random_10_0', 'q4v1_q4_boundary_16_0',
                    'q4v1_q4_clustered_16_1', 'q4v1_q4_heading_sweep_16_3')
    recovery = {('q4v1_q4_boundary_16_0', 'dual21'), ('q4v1_q4_heading_sweep_16_3', 'grid25')}
    results = []
    for scene_id in selected_ids:
        scene = next(scene for scene in scenes if scene['id'] == scene_id)
        for network in ('dual21', 'grid25'):
            direct = load(record_path(output, scene, 'baseline', network))
            for run_mode in (('normal', 'recovery') if (scene_id, network) in recovery else ('normal',)):
                trial = f'{scene_id}_{network}_{run_mode}'
                folder = output / 'http' / trial
                folder.mkdir(parents=True, exist_ok=False)
                if not direct['all_cleared']:
                    results.append({'trial': trial, 'passed': False, 'error': 'Direct policy failed; no HTTP actions sent'})
                    save(output / 'http_results.json', results)
                    continue
                faults = [] if run_mode == 'normal' else [Fault('/enter', 'drop_after'), Fault('/measure', 'drop_after'),
                    Fault('/measure', 'truncate_after'), Fault('/clear', 'drop_after'), Fault('/exit', 'drop_after')]
                world = OfflineRuleWorld([Source(**source) for source in scene['sources']], scene['seed'], scene['error_mode'])
                with MockArena(world, faults=faults) as arena, storage.Journal(folder / 'requests.jsonl') as journal:
                    client = client_module.RobotClient(arena.base_url, arena.robot_id, journal, mock_token=arena.token)
                    summary = session_module.run_session(client, 'q4', network)
                    save(folder / 'summary.json', summary)
                    requests = {}
                    for request in arena.requests:
                        request_id = json.loads(request['body_utf8'])['request_id']
                        requests.setdefault(request_id, set()).add((request['path'], request['body_utf8']))
                    checks = {
                        'normal_exit': summary['status'] == 'completed' and client.exited and not summary['error'],
                        'all_cleared': not world.remaining and len(client.cleared_channels) == scene['count'],
                        'same_actions': action_hash(world.trace) == direct['action_sha256'],
                        'same_clock': abs(client.virtual_seconds - direct['virtual_seconds']) <= (len(world.trace) + 1) * 1e-6,
                        'authoritative_clock_exact': client.virtual_seconds == world.virtual_seconds,
                        'position_exact': client.position == tuple(world.position),
                        'receiver_exact': client.receiver_channel == world.receiver_channel,
                        'same_policy_stats': summary.get('policy_stats') == direct['policy_stats'],
                        'one_execution_per_action': len(arena.executions) == len(world.trace) + 2,
                        'identical_retry_bodies': all(len(bodies) == 1 for bodies in requests.values()),
                        'expected_retries': client.retry_count == (5 if run_mode == 'recovery' else 0),
                        'all_faults_exercised': not arena.faults,
                        'no_uncertainty': not any(summary.get(key) for key in ('uncertain', 'pending_request', 'watchdog_errors')),
                    }
                    save(folder / 'checks.json', checks)
                    results.append({'trial': trial, 'scene_id': scene_id, 'network': network, 'passed': all(checks.values()),
                                    'checks': checks, 'cleared': len(client.cleared_channels), 'source_count': scene['count'],
                                    'virtual_s': client.virtual_seconds, 'physical_actions': len(world.trace),
                                    'retry_count': client.retry_count, 'request_log_sha256': file_hash(folder / 'requests.jsonl')})
                save(output / 'http_results.json', results)
                print(f'HTTP {trial}: passed={results[-1]["passed"]}', flush=True)
    verify(output)
    if len(results) != 10 or not all(row['passed'] for row in results):
        raise AssertionError('Q4 local HTTP validation incomplete or failed; preserve evidence')


def report(output):
    rows = all_records(output)
    grouped = {}
    for profile in ('all', *PROFILES, 'heading_sweep'):
        group = rows if profile == 'all' else [row for row in rows if row['profile'] == profile]
        grouped[profile] = {
            'joint_vs_original': paired_summary(group, 'baseline_dual21', 'joint_dual21'),
            'grid25_vs_dual21': paired_summary(group, 'baseline_dual21', 'baseline_grid25'),
        }
    variants = {}
    for mode, network in VARIANTS:
        group = [row for row in rows if row['mode'] == mode and row['network'] == network]
        success = [row for row in group if row['all_cleared']]
        variants[variant_name(mode, network)] = {'runs': len(group), 'all_cleared': len(success)}
        if success:
            variants[variant_name(mode, network)].update({
                'mean_s_per_source': statistics.mean(row['seconds_per_source'] for row in success),
                'median_s_per_source': statistics.median(row['seconds_per_source'] for row in success),
                'min_s_per_source': min(row['seconds_per_source'] for row in success),
                'max_s_per_source': max(row['seconds_per_source'] for row in success),
                'mean_move_fraction': statistics.mean(row['world_stats']['move_meters'] / 5 / row['virtual_seconds'] for row in success),
                'failed_clear_count': sum(row['world_stats']['failed_clear_count'] for row in success),
                'slowest_scene': max(success, key=lambda row: row['seconds_per_source'])['scene_id'],
            })
    http_rows = load(output / 'http_results.json') if (output / 'http_results.json').exists() else []
    replay_rows = load(output / 'replay.json') if (output / 'replay.json').exists() else []
    summary = {'created_local': now(), 'official_calls': 0, 'independent_synthetic_scenes': 64, 'offline_runs': len(rows),
               'all_cleared_runs': sum(row['all_cleared'] for row in rows), 'variants': variants,
               'paired_by_profile': grouped, 'failures': [row for row in rows if not row['all_cleared']],
               'http_runs': len(http_rows), 'http_passed': sum(row['passed'] for row in http_rows),
               'http_physical_actions': sum(row.get('physical_actions', 0) for row in http_rows),
               'http_recovered_retries': sum(row.get('retry_count', 0) for row in http_rows),
               'replay_runs': len(replay_rows), 'replay_passed': sum(row['passed'] for row in replay_rows)}
    save(output / 'summary.json', summary)
    print(json.dumps({key: summary[key] for key in ('offline_runs', 'all_cleared_runs', 'variants', 'http_runs', 'http_passed',
                                                  'http_physical_actions', 'http_recovered_retries', 'replay_runs', 'replay_passed')},
                     ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description='Q4 local-only frozen-policy validation; no official simulator endpoints')
    parser.add_argument('command', choices=('prepare', 'offline', 'worker', 'replay', 'http', 'report'))
    parser.add_argument('--output', required=True, type=storage.d_path)
    parser.add_argument('--scene-id')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'reports' / 'q4_validation_v1'):
        raise ValueError('Q4 evidence must remain inside the dedicated D-drive report directory')
    if args.command == 'worker':
        worker(output, args.scene_id)
    else:
        globals()[args.command](output)


if __name__ == '__main__':
    main()
