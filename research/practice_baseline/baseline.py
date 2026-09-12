from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'reports' / 'practice_baseline_v1'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def verify_hashes():
    frozen = load(ROOT / 'reports' / 'plan_trials_v2' / 'protocol.json')['source_hashes']
    latest = Path((ROOT / 'reports' / 'interface_validation_v1' / 'LATEST_SUCCESS.txt').read_text(encoding='utf-8-sig').strip())
    interface = load(latest / 'protocol.json')['interface_source_hashes']
    for name, expected in frozen.items():
        if hashlib.sha256((ROOT / 'research' / 'offline_validation' / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Frozen algorithm changed: {name}')
    for name, expected in interface.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Validated interface changed: {name}')
    return {'frozen_hashes': frozen, 'interface_hashes': interface}


def freeze():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / 'protocol.json').exists():
        verify_hashes()
        return
    save(OUTPUT / 'protocol.json', {
        'created_local': datetime.now(timezone(timedelta(hours=8))).isoformat(), 'additional_runs_per_problem': 5,
        'selected': {'q3': 'E_joint', 'q4': 'F_route'}, 'networks': {'q3': 'grid7', 'q4': 'dual21'},
        'rule': 'Include every initiated run, including incomplete or failed runs; no tuning during the baseline batch.',
        'pilot_rule': 'Previously observed first q3/q4 practices are labeled pilot; report new batch and all six separately.',
        'official_mode': 'practice only, verified on the visible UI before /enter; never formal',
        'truth_boundary': 'Only post-exit public aggregate counts and self-generated request logs are read.',
        'comparison_boundary': 'Official cases cannot be replayed here; paired optimization comparisons use fresh synthetic scenes.',
        **verify_hashes(),
    })
    save(OUTPUT / 'runs.json', [])


def audit_log(folder):
    folder = Path(folder).resolve()
    if not folder.is_relative_to(ROOT / 'logs' / 'practice'):
        raise ValueError('Only the project practice request logs are permitted')
    summary = load(folder / 'summary.json')
    entries = [json.loads(line) for line in (folder / 'requests.jsonl').read_text(encoding='utf-8').splitlines()]
    requests = {entry['payload']['request_id']: entry for entry in entries if entry['event'] == 'request'}
    responses = {entry['request_id']: json.loads(entry['body_utf8']) for entry in entries if entry['event'] == 'response'}
    commits = [entry for entry in entries if entry['event'] == 'committed']
    position, receiver = (0.0, 0.0), 1
    movement, switches, measures = 0.0, 0, 0
    cleared, clear_results, failed_by_channel = set(), Counter(), Counter()
    movement_by_action = Counter()
    operation_counts = Counter()
    for entry in commits:
        request = requests[entry['request_id']]
        payload, path = request['payload'], request['path']
        response = responses[entry['request_id']]
        if not response['accepted']:
            raise ValueError('Commit references a rejected response')
        operation_counts[path] += 1
        if path not in ('/measure', '/clear'):
            continue
        destination = (payload['position']['x'], payload['position']['y'])
        distance = math.dist(position, destination)
        movement += distance
        movement_by_action[path] += distance
        position = destination
        if path == '/measure':
            measures += 1
            switches += payload['channel'] != receiver
            receiver = payload['channel']
        else:
            clear_results[response['clear_result']] += 1
            if response['clear_result'] == 'success':
                cleared.add(payload['channel'])
            else:
                failed_by_channel[payload['channel']] += 1
    components = {'movement_s': movement / 5, 'measure_s': measures * 5, 'switch_s': switches,
                  'successful_clear_s': clear_results['success'] * 5, 'failed_clear_s': clear_results['no_target_in_range'] * 3}
    residual = summary['virtual_time_s'] - sum(components.values())
    if abs(residual) > (len(commits) + 1) * 1e-6:
        raise ValueError(f'Independent timing audit failed: {residual}')
    acknowledged = [responses[entry['request_id']] for entry in commits]
    return {
        'run_id': folder.name, 'problem': summary['problem'], 'variant': summary['variant'], 'network': summary['network'],
        'client_status': summary['status'], 'exited': summary['exited'], 'error': summary['error'],
        'cleared': len(cleared), 'cleared_channels': sorted(cleared), 'virtual_s': summary['virtual_time_s'],
        'real_response_s': (acknowledged[-1]['real_timestamp_ms'] - acknowledged[0]['real_timestamp_ms']) / 1000,
        'measure_count': measures, 'clear_failures': clear_results['no_target_in_range'],
        'move_m': movement, 'move_by_destination_action_m': dict(movement_by_action),
        'components': components, 'rounding_residual_s': residual, 'accepted_counts': dict(operation_counts),
        'failed_by_channel': dict(failed_by_channel), 'retry_count': summary['retry_count'],
        'policy_stats': summary.get('policy_stats', {}),
        'log_directory': str(folder), 'log_sha256': hashlib.sha256((folder / 'requests.jsonl').read_bytes()).hexdigest(),
    }


def register(args):
    verify_hashes()
    protocol, rows = load(OUTPUT / 'protocol.json'), load(OUTPUT / 'runs.json')
    row = audit_log(args.run_dir)
    evidence = Path(args.evidence).resolve()
    if not evidence.is_relative_to(ROOT / 'outputs') or not evidence.is_file():
        raise ValueError('Require a saved public result screenshot inside the project outputs')
    if not 10 <= args.total <= 16 or args.omni < 0 or args.directional < 0 or args.omni + args.directional != args.total:
        raise ValueError('Invalid public source counts')
    if row['variant'] != protocol['selected'][row['problem']] or row['network'] != protocol['networks'][row['problem']]:
        raise ValueError('Baseline configuration mismatch')
    if any(previous['run_id'] == row['run_id'] for previous in rows):
        raise ValueError('This run is already registered; do not duplicate it')
    if args.kind == 'batch' and sum(previous['problem'] == row['problem'] and previous['kind'] == 'batch' for previous in rows) >= 5:
        raise ValueError('The authorized five additional runs for this problem are already registered')
    row.update({
        'kind': args.kind, 'source_count': args.total, 'omni': args.omni, 'directional': args.directional,
        'case_code': args.case_code, 'evidence': str(evidence), 'evidence_sha256': hashlib.sha256(evidence.read_bytes()).hexdigest(),
        'all_cleared': row['cleared'] == args.total and row['client_status'] == 'completed' and row['exited'],
        'seconds_per_cleared_source': row['virtual_s'] / row['cleared'] if row['cleared'] else None,
    })
    if row['cleared'] > args.total:
        raise ValueError('Cleared count exceeds public total; inspect the evidence')
    rows.append(row)
    save(OUTPUT / 'runs.json', rows)
    print(json.dumps({key: row[key] for key in ('run_id', 'problem', 'kind', 'cleared', 'source_count', 'all_cleared', 'seconds_per_cleared_source')}, ensure_ascii=False))


def report():
    rows = load(OUTPUT / 'runs.json')
    statistics_rows = []
    for problem in ('q3', 'q4'):
        for kind in ('batch', 'all'):
            group = [row for row in rows if row['problem'] == problem and (kind == 'all' or row['kind'] == kind)]
            if not group:
                continue
            times = [row['seconds_per_cleared_source'] for row in group if row['seconds_per_cleared_source'] is not None]
            statistics_rows.append({
                'problem': problem, 'scope': kind, 'runs': len(group), 'successes': sum(row['all_cleared'] for row in group),
                'mean_s_per_cleared_source': statistics.mean(times), 'median_s_per_cleared_source': statistics.median(times),
                'min_s_per_cleared_source': min(times), 'max_s_per_cleared_source': max(times),
                'mean_move_fraction': statistics.mean(row['components']['movement_s'] / row['virtual_s'] for row in group),
                'total_failed_clears': sum(row['clear_failures'] for row in group),
                'total_opportunistic_clears': sum(row['policy_stats'].get('opportunistic_clears', 0) for row in group),
            })
    save(OUTPUT / 'summary.json', statistics_rows)
    flat = [{key: row[key] for key in ('run_id', 'problem', 'kind', 'source_count', 'omni', 'directional', 'cleared',
                                      'all_cleared', 'virtual_s', 'seconds_per_cleared_source', 'move_m', 'measure_count', 'clear_failures')} for row in rows]
    with (OUTPUT / 'runs.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    lines = ['# 冻结策略官方演练基线', '', '## Material Passport',
             '- 类型：官方演练，所有正式测试次数保持未使用。',
             '- 状态：执行与公开总数交叉核对；统计值为描述性分析，不是总体成功概率或官方排名。',
             '- 冻结配置：问题3 E_joint + grid7；问题4 F_route + dual21。',
             '- 每题计划新增5次；之前各1次标记pilot，不混淆新增批次与全部案例。', '',
             '| 问题 | 范围 | 全清除/案例 | 均值 秒/源 | 中位数 | 最小 | 最大 | 移动占比均值 |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for item in statistics_rows:
        lines.append(f'| {item["problem"]} | {item["scope"]} | {item["successes"]}/{item["runs"]} | '
                     f'{item["mean_s_per_cleared_source"]:.3f} | {item["median_s_per_cleared_source"]:.3f} | '
                     f'{item["min_s_per_cleared_source"]:.3f} | {item["max_s_per_cleared_source"]:.3f} | '
                     f'{100 * item["mean_move_fraction"]:.1f}% |')
    lines += ['', '## 解释边界',
              '- 六次全清除不等于总体失败概率为零；只有少量官方随机案例，不报告精确尾部分位或显著性结论。',
              '- 单位源时间是每例总虚拟时间除以实际清除数，再对案例等权平均；与汇总总时间除以所有源数不是同一统计量。',
              '- 问题3、4的案例和类型组合不同，不能直接拿两题的均值比较策略优劣。',
              '- 时间包含移动、检测、切频、成功及未命中清除；失败清除次数不能单独代表总时间。',
              '- 不对官方隐藏案例构造可读真值；后续算法成对比较使用独立合成场景，官方新案例仅作外部验证。',
              '- runs.json保存所有已登记案例及截图、日志哈希；本批不删去慢例或失败例。']
    (OUTPUT / '冻结策略_官方演练基线报告.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(statistics_rows, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description='Read-only practice-log auditing; never sends simulator requests')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('freeze')
    commands.add_parser('report')
    command = commands.add_parser('register')
    command.add_argument('--run-dir', required=True, type=Path)
    command.add_argument('--total', required=True, type=int)
    command.add_argument('--omni', required=True, type=int)
    command.add_argument('--directional', required=True, type=int)
    command.add_argument('--evidence', required=True, type=Path)
    command.add_argument('--case-code', required=True)
    command.add_argument('--kind', choices=('pilot', 'batch'), default='batch')
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze()
    elif args.command == 'register':
        register(args)
    else:
        report()


if __name__ == '__main__':
    main()
