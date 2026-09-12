from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'reports' / 'candidate_practice_v3'
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'research' / 'practice_baseline'))

from baseline import audit_log, verify_hashes

from b2026_candidate.validation import (
    require_validation,
    source_hashes,
    verify_candidate,
)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def evidence(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT / 'outputs') or not path.is_file():
        raise ValueError('Evidence must be a captured public screenshot under the project outputs directory')
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def verify():
    protocol = load(OUTPUT / 'protocol.json')
    if str(require_validation()) != protocol['validation_evidence'] or source_hashes() != protocol['interface_hashes']:
        raise ValueError('Validated adapter changed during the fixed practice batch')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != protocol['recorder_hash']:
        raise ValueError('Practice recorder changed after freeze')
    verify_hashes()
    verify_candidate()
    return protocol


def freeze():
    if (OUTPUT / 'protocol.json').exists():
        verify()
        return
    validated = require_validation()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save(OUTPUT / 'protocol.json', {
        'created_local': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'problem': 'q3', 'variant': 'E_joint+v3_combined', 'network': 'grid7', 'planned_practices': 5,
        'validation_evidence': str(validated), 'interface_hashes': source_hashes(),
        'recorder_hash': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'primary_checks': 'Normal exit, no uncertainty, official public source total equals successful clear count, independent timing audit.',
        'stopping_rule': 'Fixed five practices; any incomplete/failed/audit-failed attempt is retained and blocks automatic continuation.',
        'comparison': 'Different official random cases from the old baseline; descriptive comparison only, no paired effect or significance claim.',
        'truth_boundary': 'Only own robot requests and public post-exit aggregate counts; no hidden source data or encrypted-log inspection.',
        'official_mode': 'practice only; inspect visible q3 practice readiness before each enter; never formal',
        'algorithm_hashes': verify_candidate(), 'original_hashes': verify_hashes(),
    })
    save(OUTPUT / 'runs.json', [])


def begin(args):
    protocol = verify()
    rows = load(OUTPUT / 'runs.json')
    if len(rows) >= protocol['planned_practices'] or any(row['state'] != 'completed' for row in rows):
        raise ValueError('Batch limit reached or a previous attempt needs attention')
    if any(row['case_code'] == args.case_code for row in rows):
        raise ValueError('Duplicate case')
    row = {'attempt': len(rows) + 1, 'state': 'pending', 'case_code': args.case_code,
           'ready_evidence': evidence(args.evidence), 'created_local': datetime.now(timezone(timedelta(hours=8))).isoformat()}
    rows.append(row)
    save(OUTPUT / 'runs.json', rows)
    print(json.dumps(row, ensure_ascii=False))


def successful(audit, total):
    return (audit['client_status'] == 'completed' and audit['exited'] and not audit['error']
            and audit['cleared'] == total)


def register(args):
    verify()
    rows = load(OUTPUT / 'runs.json')
    row = rows[args.attempt - 1]
    if row['attempt'] != args.attempt or row['state'] != 'pending':
        raise ValueError('Only the pending attempt can be completed')
    if not 10 <= args.total <= 16:
        raise ValueError('Public source total outside the q3 bounds')
    folder = Path(args.run_dir).resolve()
    if not folder.is_relative_to(ROOT / 'logs' / 'practice'):
        raise ValueError('Only self-generated practice robot logs are allowed')
    row.update({'source_count': args.total, 'omni': args.total, 'directional': 0,
                'result_evidence': evidence(args.evidence), 'log_directory': str(folder)})
    try:
        summary = load(folder / 'summary.json')
        if summary.get('variant') != 'E_joint+v3_combined' or summary.get('candidate_mode') != 'combined':
            raise ValueError('Run does not use the frozen candidate')
        if summary.get('uncertain') or summary.get('pending_request') or summary.get('watchdog_errors'):
            raise ValueError('Run outcome is uncertain or watchdog failed')
        audit = audit_log(folder)
        row.update(audit)
        row['all_cleared'] = successful(audit, args.total)
        row['seconds_per_source'] = audit['virtual_s'] / args.total if row['all_cleared'] else None
        row['state'] = 'completed' if row['all_cleared'] else 'failed'
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        row.update({'state': 'failed_audit', 'all_cleared': False, 'audit_error': f'{type(error).__name__}: {error}'})
    save(OUTPUT / 'runs.json', rows)
    print(json.dumps({key: row.get(key) for key in ('attempt', 'state', 'source_count', 'cleared', 'virtual_s',
                                                  'seconds_per_source', 'retry_count', 'audit_error')}, ensure_ascii=False))
    if row['state'] != 'completed':
        raise RuntimeError('Practice failed or audit incomplete; preserve evidence and stop this batch')


def failure(args):
    rows = load(OUTPUT / 'runs.json')
    row = rows[args.attempt - 1]
    if row['state'] != 'pending':
        raise ValueError('Only a pending attempt may be marked failed')
    row.update({'state': 'failed', 'all_cleared': False, 'note': args.note,
                'result_evidence': evidence(args.evidence)})
    save(OUTPUT / 'runs.json', rows)


def report():
    verify()
    rows = load(OUTPUT / 'runs.json')
    previous = [row for row in load(ROOT / 'reports' / 'practice_baseline_v1' / 'runs.json')
                if row['problem'] == 'q3' and row['kind'] == 'batch']
    passed = sum(row['state'] == 'completed' for row in rows)
    summary = {'attempts': len(rows), 'all_cleared': passed, 'planned': 5, 'formal_calls': 0,
               'baseline_cases': len(previous), 'baseline_mean_s_per_source': statistics.mean(
                   row['seconds_per_cleared_source'] for row in previous)}
    if rows and passed == len(rows):
        values = [row['seconds_per_source'] for row in rows]
        summary.update({'mean_s_per_source': statistics.mean(values), 'median_s_per_source': statistics.median(values),
                        'min_s_per_source': min(values), 'max_s_per_source': max(values),
                        'mean_move_fraction': statistics.mean(row['components']['movement_s'] / row['virtual_s'] for row in rows),
                        'retries': sum(row['retry_count'] for row in rows),
                        'clear_failures': sum(row['clear_failures'] for row in rows)})
    save(OUTPUT / 'summary.json', summary)
    lines = ['# 问题3候选固定配置官方演练', '',
             '## 边界',
             '- 冻结v3 combined / grid7；每例之前核对问题3演练等待机器人进入页面，没有正式测试。',
             '- 运行前登记意图，失败保留并停止自动继续，不中途换参数或重置同一个案例。',
             '- 使用公开结束总数核对全清，自己的请求日志独立核对计时；不读取隐藏案例或加密日志。', '',
             '| 次序 | 官方案例编号 | 清除/公开总数 | 总虚拟秒 | 虚拟秒/源 | 重试 | 状态 |',
             '|---|---|---|---:|---:|---:|---|']
    for row in rows:
        if row['state'] == 'completed':
            lines.append(f'| {row["attempt"]} | {row["case_code"]} | {row["cleared"]}/{row["source_count"]} | '
                         f'{row["virtual_s"]:.3f} | {row["seconds_per_source"]:.3f} | {row["retry_count"]} | 正常全清 |')
        else:
            lines.append(f'| {row["attempt"]} | {row["case_code"]} | 未核实 | - | - | - | {row["state"]} |')
    lines.extend(['', '## 描述性汇总', f'- 已登记{len(rows)}次，核实全清{passed}次；预设最多5次。'])
    if 'mean_s_per_source' in summary:
        lines.append(f'- 候选均值{summary["mean_s_per_source"]:.3f}秒/源，中位数{summary["median_s_per_source"]:.3f}；'
                     f'范围{summary["min_s_per_source"]:.3f}—{summary["max_s_per_source"]:.3f}。')
    lines.extend([f'- 原策略此前固定5次均值{summary["baseline_mean_s_per_source"]:.3f}秒/源，仅作为历史背景。',
                  '- 两批不是同场景配对，源数/位置/半径/误差可能不同；不能据此计算可信的算法改善百分比或显著性。',
                  '- 少量全清记录不等于总体失败概率为零；结合离线留出、压力退化与接口验证判断，不选取单次最好成绩。',
                  '- 原程序和问题4默认策略未更改；本入口只支持问题3候选演练，正式测试仍需另行明确确认。'])
    (OUTPUT / '问题3候选_官方演练报告.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description='Local evidence recording only; never connects to a simulator')
    subcommands = parser.add_subparsers(dest='command', required=True)
    subcommands.add_parser('freeze')
    subcommands.add_parser('report')
    start = subcommands.add_parser('begin')
    start.add_argument('--case-code', required=True)
    start.add_argument('--evidence', required=True)
    finish = subcommands.add_parser('register')
    finish.add_argument('--attempt', type=int, choices=range(1, 6), required=True)
    finish.add_argument('--run-dir', required=True)
    finish.add_argument('--total', type=int, required=True)
    finish.add_argument('--evidence', required=True)
    failed = subcommands.add_parser('failure')
    failed.add_argument('--attempt', type=int, choices=range(1, 6), required=True)
    failed.add_argument('--evidence', required=True)
    failed.add_argument('--note', required=True)
    args = parser.parse_args()
    if args.command in ('freeze', 'report'):
        {'freeze': freeze, 'report': report}[args.command]()
    else:
        {'begin': begin, 'register': register, 'failure': failure}[args.command](args)


if __name__ == '__main__':
    main()
