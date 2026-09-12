from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from candidate import MODES, ROOT, CandidatePolicy
from geometry import dual_ring_network, seven_network, triangular_network
from offline_benchmark import OfflineRuleWorld, Source
from plan_experiments import make_scene
from plan_geometry import conservative_cover_check, geometry_contains
from plan_policy import PlanPolicy
from shapely.errors import GEOSException

OUTPUT = ROOT / 'reports' / 'policy_optimization_v3'
SCRIPT = Path(__file__).resolve().parent
BASE_VARIANTS = {'q3': 'E_joint', 'q4': 'F_route'}
BASE_NETWORKS = {'q3': 'grid7', 'q4': 'dual21'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode('utf-8')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded(value))


def hashes():
    baseline = read(ROOT / 'reports' / 'plan_trials_v2' / 'protocol.json')['source_hashes']
    for name, expected in baseline.items():
        path = ROOT / 'research' / 'offline_validation' / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen baseline changed: {name}')
    return {
        'baseline': baseline,
        'candidate': {name: hashlib.sha256((SCRIPT / name).read_bytes()).hexdigest()
                      for name in ('candidate.py', 'experiments.py')},
    }


def verify():
    protocol = read(OUTPUT / 'protocol.json')
    if protocol['hashes'] != hashes():
        raise ValueError('Code changed after experiment freeze; use a new version')
    if protocol['scenes_sha256'] != hashlib.sha256((OUTPUT / 'scenes.json').read_bytes()).hexdigest():
        raise ValueError('Frozen scenes changed')


def prepare():
    if (OUTPUT / 'protocol.json').exists():
        verify()
        return
    scenes = []
    specifications = (
        ('development', (10, 13, 16), 1, ('random', 'boundary', 'clustered', 'near_origin'), 21100000),
        ('holdout', (10, 12, 14, 16), 2, ('random', 'boundary', 'clustered', 'near_origin'), 26100000),
        ('stress', (10, 16), 2, ('max_radius', 'adversarial_heading'), 28100000),
    )
    for phase, counts, repetitions, profiles, base_seed in specifications:
        for problem_index, problem in enumerate(('q3', 'q4')):
            for profile_index, profile in enumerate(profiles):
                for count in counts:
                    for repetition in range(repetitions):
                        seed = base_seed + problem_index * 100000 + profile_index * 1000 + count * 10 + repetition
                        scenes.append(make_scene(problem, count, profile, seed, phase, repetition))
    old_seeds = {scene['seed'] for scene in read(ROOT / 'reports' / 'plan_trials_v2' / 'scenes.json')}
    if old_seeds.intersection(scene['seed'] for scene in scenes):
        raise ValueError('Fresh evaluation seeds overlap earlier experiments')
    save(OUTPUT / 'scenes.json', scenes)
    save(OUTPUT / 'protocol.json', {
        'created_local': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'kind': 'fresh_synthetic_paired_candidate_validation', 'official_calls': 0,
        'hashes': hashes(), 'scenes_sha256': hashlib.sha256((OUTPUT / 'scenes.json').read_bytes()).hexdigest(),
        'modes': MODES, 'baseline_variants': BASE_VARIANTS, 'baseline_networks': BASE_NETWORKS,
        'phase_scene_counts': {phase: sum(scene['phase'] == phase for scene in scenes)
                               for phase in ('development', 'holdout', 'stress')},
        'selection_rule': 'Per problem: all 12 development scenes must clear and no case may exceed 1.25x baseline time; choose minimum mean case-level seconds/source, ties within 1 second/source favor baseline, prune, joint, combined in that order.',
        'holdout_rule': 'Freeze selection before holdout/stress; no tuning on either. Paired baseline versus winner on fresh scenes. Retain all failures, with no efficiency reward for partial clearance.',
        'stress_backup': 'Also pair the q4 winner and baseline on the unchanged analytical grid25 backup.',
        'engineering_checks': 'Unit tests and separate smoke seeds 18100103/18100104 are not counted as evaluation scenes.',
        'limitations': 'Synthetic stress mix is not an estimate of the official case distribution. No formal-test performance or general all-clear guarantee is inferred.',
    })
    print(json.dumps({'prepared_scenes': len(scenes)}, ensure_ascii=False), flush=True)


def evaluate(scene, mode, network):
    world = OfflineRuleWorld([Source(**source) for source in scene['sources']], scene['seed'], scene['error_mode'])
    stations = {'grid7': seven_network, 'dual21': dual_ring_network,
                'grid25': lambda: triangular_network()[0]}[network]()
    started = time.perf_counter()
    failure = None
    snapshots_checked, certificates_checked = 0, 0
    policy_class = PlanPolicy if mode == 'baseline' else CandidatePolicy
    args = (world.port(), scene['problem'], stations, BASE_VARIANTS[scene['problem']], network)
    policy = policy_class(*args) if mode == 'baseline' else policy_class(*args, mode=mode)
    try:
        result = policy.run()
        if abs(result['virtual_seconds'] - world.virtual_seconds) > 1e-6:
            raise ValueError('Independent policy and world clocks disagree')
        for channel, region in result['snapshots']:
            if channel in world.sources:
                if not geometry_contains(region, world.sources[channel].position):
                    raise ValueError(f'True source excluded from conservative region: {channel}')
                snapshots_checked += 1
        for region, centers, kind in result['certificates']:
            if not conservative_cover_check(region, centers):
                raise ValueError(f'Continuous clearance certificate rejected: {kind}')
            certificates_checked += 1
        if set(result['declared_absent']) & set(world.sources):
            raise ValueError('A real source was declared absent')
        if world.remaining:
            raise ValueError(f'Uncleared sources: {sorted(world.remaining)}')
        if any(np.linalg.norm(action['position']) >= 5000 for action in world.trace):
            raise ValueError('Movement left the experiment safety envelope')
    except (AssertionError, RuntimeError, ValueError, ArithmeticError, GEOSException) as exception:
        failure = f'{type(exception).__name__}: {exception}'
    cleared = len(world.sources) - len(world.remaining)
    all_cleared = failure is None and cleared == len(world.sources)
    record = {
        'scene_id': scene['id'], 'phase': scene['phase'], 'problem': scene['problem'], 'profile': scene['profile'],
        'source_count': len(world.sources), 'seed': scene['seed'], 'mode': mode, 'network': network,
        'all_cleared': all_cleared, 'cleared_count': cleared, 'failure': failure,
        'virtual_seconds': world.virtual_seconds,
        'seconds_per_source': world.virtual_seconds / len(world.sources) if all_cleared else None,
        'wall_seconds': time.perf_counter() - started, 'action_count': len(world.trace),
        'snapshots_checked': snapshots_checked, 'certificates_checked': certificates_checked,
        'world_stats': world.stats, 'policy_stats': policy.stats,
        'trace_sha256': hashlib.sha256(encoded(world.trace)).hexdigest(),
    }
    return record, world.trace


def records(phase):
    path = OUTPUT / f'{phase}.jsonl'
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []


def tasks(phase):
    scenes = [scene for scene in read(OUTPUT / 'scenes.json') if scene['phase'] == phase]
    selection = read(OUTPUT / 'selection.json')['selected'] if phase != 'development' else None
    result = []
    for scene in scenes:
        modes = MODES if selection is None else list(dict.fromkeys(('baseline', selection[scene['problem']])))
        networks = [BASE_NETWORKS[scene['problem']]]
        if phase == 'stress' and scene['problem'] == 'q4':
            networks.append('grid25')
        result.extend((scene, mode, network) for network in networks for mode in modes)
    return result


def run_phase(phase):
    verify()
    if (OUTPUT / f'{phase}.jsonl').exists():
        raise ValueError('Phase already started; preserve results and inspect rather than silently rerunning')
    planned = tasks(phase)
    with (OUTPUT / f'{phase}.jsonl').open('x', encoding='utf-8') as handle:
        for index, (scene, mode, network) in enumerate(planned, start=1):
            save(OUTPUT / 'running_task.json', {'phase': phase, 'index': index, 'total': len(planned),
                                               'scene_id': scene['id'], 'mode': mode, 'network': network})
            record, trace = evaluate(scene, mode, network)
            save(OUTPUT / 'traces' / f'{scene["id"]}_{network}_{mode}.json', trace)
            handle.write(encoded(record).decode('utf-8') + '\n')
            handle.flush()
            if index % 8 == 0 or record['failure'] is not None or index == len(planned):
                print(json.dumps({'phase': phase, 'done': index, 'total': len(planned),
                                  'latest_failure': record['failure']}, ensure_ascii=False), flush=True)
    save(OUTPUT / f'{phase}_complete.json', {'completed': len(planned)})


def select():
    verify()
    if (OUTPUT / 'selection.json').exists():
        raise ValueError('Selection is already frozen')
    development = records('development')
    if len(development) != len(tasks('development')):
        raise ValueError('Development is incomplete')
    selected, assessment = {}, {}
    for problem in ('q3', 'q4'):
        subset = [record for record in development if record['problem'] == problem]
        baseline = {record['scene_id']: record for record in subset if record['mode'] == 'baseline'}
        if not all(record['all_cleared'] for record in baseline.values()):
            raise ValueError('Baseline failed; diagnose before candidate selection')
        assessments = {}
        for mode in MODES:
            group = [record for record in subset if record['mode'] == mode]
            eligible = all(record['all_cleared'] for record in group)
            mean = statistics.mean(record['seconds_per_source'] for record in group) if eligible else None
            worst_ratio = max(record['virtual_seconds'] / baseline[record['scene_id']]['virtual_seconds']
                              for record in group) if eligible else None
            assessments[mode] = {'all_cleared': eligible, 'mean_s_per_source': mean,
                                 'max_time_ratio': worst_ratio, 'eligible': eligible and worst_ratio <= 1.25}
        best = min(value['mean_s_per_source'] for value in assessments.values() if value['eligible'])
        selected[problem] = next(mode for mode in MODES if assessments[mode]['eligible']
                                 and assessments[mode]['mean_s_per_source'] <= best + 1.0)
        assessment[problem] = assessments
    save(OUTPUT / 'selection.json', {'selected': selected, 'development_assessment': assessment, 'hashes': hashes()})
    print(json.dumps({'selected': selected, 'development_assessment': assessment}, ensure_ascii=False, indent=2))


def paired_summary(group, chosen):
    baseline = {record['scene_id']: record for record in group if record['mode'] == 'baseline'}
    candidate = {record['scene_id']: record for record in group if record['mode'] == chosen}
    complete = [scene_id for scene_id in baseline if scene_id in candidate
                and baseline[scene_id]['all_cleared'] and candidate[scene_id]['all_cleared']]
    result = {'cases': len(baseline), 'baseline_success': sum(record['all_cleared'] for record in baseline.values()),
              'candidate_success': sum(record['all_cleared'] for record in candidate.values()), 'chosen': chosen}
    if len(complete) != len(baseline):
        return result | {'efficiency_comparison': 'withheld: not every paired case cleared'}
    before = [baseline[scene_id]['seconds_per_source'] for scene_id in complete]
    after = [candidate[scene_id]['seconds_per_source'] for scene_id in complete]
    deltas = [new - old for old, new in zip(before, after, strict=True)]
    result.update({
        'baseline_mean': statistics.mean(before), 'candidate_mean': statistics.mean(after),
        'mean_paired_delta': statistics.mean(deltas), 'median_paired_delta': statistics.median(deltas),
        'relative_mean_reduction': 1 - statistics.mean(after) / statistics.mean(before),
        'wins': sum(delta < -1e-6 for delta in deltas), 'ties': sum(abs(delta) <= 1e-6 for delta in deltas),
        'losses': sum(delta > 1e-6 for delta in deltas),
        'worst_slowdown_fraction': max(new / old - 1 for old, new in zip(before, after, strict=True)),
        'baseline_mean_move_m': statistics.mean(baseline[scene_id]['world_stats']['move_meters'] for scene_id in complete),
        'candidate_mean_move_m': statistics.mean(candidate[scene_id]['world_stats']['move_meters'] for scene_id in complete),
    })
    return result


def report():
    verify()
    selection = read(OUTPUT / 'selection.json')['selected']
    summaries = []
    for phase in ('holdout', 'stress'):
        rows = records(phase)
        if len(rows) != len(tasks(phase)):
            raise ValueError(f'{phase} is incomplete')
        for problem in ('q3', 'q4'):
            networks = sorted({record['network'] for record in rows if record['problem'] == problem})
            for network in networks:
                group = [record for record in rows if record['problem'] == problem and record['network'] == network]
                for profile in ['all'] + sorted({record['profile'] for record in group}):
                    subset = group if profile == 'all' else [record for record in group if record['profile'] == profile]
                    summaries.append({'phase': phase, 'problem': problem, 'network': network, 'profile': profile,
                                      **paired_summary(subset, selection[problem])})
    all_rows = [record for phase in ('development', 'holdout', 'stress') for record in records(phase)]
    save(OUTPUT / 'summary.json', {'groups': summaries, 'total_runs': len(all_rows),
                                  'successful_runs': sum(record['all_cleared'] for record in all_rows)})
    lines = ['# 独立候选优化 v3：离线配对验证', '',
             '## 实验边界',
             '- 原v2策略与接口文件未修改；候选策略只接收观测端口，不读取合成源真值。',
             '- 不调用官方模拟器，不消耗演练或正式次数；这不是官方成绩提升证明。',
             '- 24个开发场景选型，64个新留出场景验证，16个极端场景压力测试；q4额外验证grid25备用网。',
             '- 全清除、可行域包含真源、连续清除覆盖证书、独立时间核算均为硬检查；失败保留，不计作高效率。',
             f'- 开发集预先固定规则选择：{selection}；留出和压力结果不再用于调参。', '',
             '## 候选差异',
             '- joint：保留原测量和站网；允许1—4个有连续覆盖证书的清除点沿途执行，逐成功前缀核算返主路线后的增量，门限120秒；清理阶段也复用顺手测量/清除。',
             '- prune：矩形兜底保留原连续覆盖证书，只剔除到当前可行域距离大于20.000001米的空清除圆；其余中心就近执行，候选集每次严格缩小。',
             '- combined：两者组合。baseline：原策略，不添加优化。', '',
             '| 阶段 | 问题/站网 | 分布 | 全清（原/新） | 原均值秒/源 | 新均值秒/源 | 均值降幅 | 配对胜/平/负 | 最坏变慢 |',
             '|---|---|---|---|---:|---:|---:|---|---:|']
    for group in summaries:
        if 'baseline_mean' not in group:
            lines.append(f'| {group["phase"]} | {group["problem"]}/{group["network"]} | {group["profile"]} | '
                         f'{group["baseline_success"]}/{group["candidate_success"]}，共{group["cases"]} | - | - | - | - | - |')
            continue
        lines.append(f'| {group["phase"]} | {group["problem"]}/{group["network"]} | {group["profile"]} | '
                     f'{group["baseline_success"]}/{group["candidate_success"]}，共{group["cases"]} | '
                     f'{group["baseline_mean"]:.3f} | {group["candidate_mean"]:.3f} | {group["relative_mean_reduction"]:.1%} | '
                     f'{group["wins"]}/{group["ties"]}/{group["losses"]} | {group["worst_slowdown_fraction"]:.1%} |')
    lines.extend(['', '## 解释与完整性',
                  '- 主指标为先逐场景计算总虚拟秒/源，再对场景等权求均值；同时分布分层报告，不与官方随机案例均值混算。',
                  '- 已检查11类统计推断风险：分层反转逐项保留；以场景而非动作为单位；人工分布限制外推；未按成功筛选、未加入碰撞变量；不宣称总体成功率。',
                  '- 使用同场景配对而非极慢官方个案前后比较，保留所有失败；开发选型与留出分离；无反复显著性检验或事后阈值修改；不将相关性或场景统计当作普遍因果证据。',
                  '- 少量固定种子和人工分布不足以证明鲁棒性。dual21仍有原有坐标敏感性边界；不删去grid25备用方案。',
                  '- 不自动接入正式机器人。官方接口一致性检查和新演练外部验证完成前，候选只能视为离线实验版本。'])
    (OUTPUT / '候选优化_离线配对报告.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'total_runs': len(all_rows), 'successful_runs': sum(record['all_cleared'] for record in all_rows),
                      'main': [group for group in summaries if group['profile'] == 'all']}, ensure_ascii=False, indent=2))


def replay():
    verify()
    scenes = {scene['id']: scene for scene in read(OUTPUT / 'scenes.json')}
    rows = [record for phase in ('development', 'holdout', 'stress') for record in records(phase)]
    comparisons = []
    for index, original in enumerate(rows, start=1):
        repeated, _trace = evaluate(scenes[original['scene_id']], original['mode'], original['network'])
        matched = (original['trace_sha256'] == repeated['trace_sha256']
                   and original['all_cleared'] == repeated['all_cleared']
                   and original['failure'] == repeated['failure'])
        comparisons.append({'scene_id': original['scene_id'], 'mode': original['mode'],
                            'network': original['network'], 'matched': matched})
        if index % 24 == 0 or not matched:
            print(json.dumps({'replayed': index, 'total': len(rows), 'latest_matched': matched}), flush=True)
    save(OUTPUT / 'replay.json', {'runs': len(rows), 'matched': sum(row['matched'] for row in comparisons),
                                 'comparisons': comparisons})
    if not all(row['matched'] for row in comparisons):
        raise ValueError('Deterministic replay differs')
    print(json.dumps({'replayed': len(rows), 'all_matched': True}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run', 'select', 'report', 'replay'))
    parser.add_argument('--phase', choices=('development', 'holdout', 'stress'))
    args = parser.parse_args()
    if args.command == 'run':
        if args.phase is None:
            parser.error('--phase is required')
        run_phase(args.phase)
    else:
        {'prepare': prepare, 'select': select, 'report': report, 'replay': replay}[args.command]()


if __name__ == '__main__':
    main()
