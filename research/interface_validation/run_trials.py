from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

from bootstrap import (
    FROZEN_HASHES,
    ROOT,
    OfflineRuleWorld,
    client_module,
    frozen,
    session_module,
    storage,
)
from mock_server import Fault, MockArena

CASES = [
    ('q3_random10', 'q3', 10, 'random', 12911001, 'grid7'),
    ('q3_boundary16', 'q3', 16, 'boundary', 12911002, 'grid7'),
    ('q3_clustered16', 'q3', 16, 'clustered', 12911003, 'grid7'),
    ('q4_random10', 'q4', 10, 'random', 12911004, 'dual21'),
    ('q4_boundary16', 'q4', 16, 'boundary', 12911005, 'dual21'),
    ('q4_clustered16', 'q4', 16, 'clustered', 12911006, 'dual21'),
    ('q4_grid25_random12', 'q4', 12, 'random', 12911007, 'grid25'),
    ('q4_grid25_boundary16', 'q4', 16, 'boundary', 12911008, 'grid25'),
]


def json_value(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if hasattr(value, 'item'):
        return value.item()
    raise TypeError(type(value).__name__)


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False, default=json_value), encoding='utf-8')


def action_signature(world):
    records = [{key: entry[key] for key in ('action', 'channel', 'position', 'response')} for entry in world.trace]
    return hashlib.sha256(json.dumps(records, sort_keys=True, default=json_value).encode()).hexdigest()


def code_hashes():
    paths = list((ROOT / 'src' / 'b2026_robot').glob('*.py')) + [ROOT / 'src' / 'run_robot.py']
    paths += list(Path(__file__).parent.glob('*.py'))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def report(output, rows):
    lines = [
        '# 官方接口适配与本地异常验证报告', '',
        '## Material Passport',
        '- Artifact: interface_validation_v1 / 本地协议集成验证。',
        '- Verification Status: VERIFIED，仅限已执行的本地单元测试和下述合成场景。',
        '- Evidence: 附件1、附件2的公开协议，冻结v2策略，自建回环HTTP假接口。',
        '- Official simulator / practice / formal calls: 0 / 0 / 0。',
        '- Ground truth: 仅由本地评估器持有，不进入策略观测端口。', '',
        '## 范围与结论',
        '保持第二轮五个冻结源文件及开发集选出的配置不变。问题3使用E_joint；问题4使用F_route，保留grid25备份。',
        '新增真实HTTP传输、严格字段转换、串行保护、幂等重试、微秒计时核对、现实/虚拟期限保护及D盘JSONL日志。',
        '本批是工程集成回归，不是新的性能调优或官方成绩估计。所有地址由自建假接口绑定127.0.0.1随机空闲端口产生。', '',
        '## 全流程结果',
        '| 案例 | 网络 | 故障恢复 | 清除数 | HTTP虚拟秒 | 动作序列一致 |',
        '|---|---|---|---:|---:|---|',
    ]
    for row in rows:
        lines.append(f'| {row["trial"]} | {row["network"]} | {row["retry_count"]}次重试 | '
                     f'{row["cleared"]}/{row["source_count"]} | {row["virtual_time_s"]:.6f} | 是 |')
    lines += [
        '', (f'共{len(CASES)}个合成场景、{len(rows)}次HTTP全流程，均全清除并确认主动退出；'
             '每次HTTP执行的动作、位置、频道和观测序列SHA-256均与同场景的冻结离线策略一致。'),
        '计时与直接离线计算允许每动作1微秒的累计舍入差；客户端对每次接受响应另设3微秒局部核对容差。',
        ('两次recovery全流程分别注入进入、检测、清除、退出的执行后断线，以及另一次检测响应截断，'
         '重试保持请求ID、路径和UTF-8请求体完全相同，没有重复执行。'), '',
        '## 单元测试与异常边界',
        ('单元测试结果以同目录unit_tests.log、verification.json为准。覆盖题目199秒计时示例、全部结果映射、'
        'accepted=false的零时间哨兵、HTTP错误不盲重试、重试耗尽、关闭连接、畸形响应、幂等冲突、'
         '清除不切频道、短/零剩余时间、虚拟时间预检查、计算期间看门狗退出、并发拒绝、日志写入失败和默认禁止官方连接。'),
        ('对于“动作可能已执行但最终响应未知”，程序标记uncertain并停止新动作；不会改用新ID重发，'
         '也不会在接口结束后用/exit查询状态。日志保留待决请求；本版本不提供跨进程断点续跑。'), '',
        '## 仍需官方演练验证',
        '- 自建假接口不是官方模拟器；不涉及登录、在线校时、25分钟窗口控制、服务器配额、加密日志或上传。',
        '- 假接口仅实现本次客户端用到的合法协议与列明的错误分支，不是完整HTTP服务器安全认证。',
        '- 本地舍入采用逐动作round到微秒；官方内部舍入细节未推断，需通过演练检查局部计时容差。',
        '- /exit成功但应答丢失后，官方接口可能关闭；客户端诚实保留未确认状态，须查看模拟器界面。',
        '- 官方接口没有返回演练/正式模式的字段。启动确认开关不能替代人工确认模拟器处于演练模式。',
        '- 21点网络的坐标敏感性等数学限制仍沿用第二轮报告，HTTP通过不增加几何理论保证。',
        '- 看门狗能在普通Python/释放GIL的计算期间请求退出；不能保证处理操作系统冻结、进程崩溃或原生代码永久持有GIL。', '',
        '## 下一步',
        ('用户确认后，人工打开官方模拟器并选择演练，使用登录参赛队号，在接口就绪后显式运行演练入口。'
         '首先检查实际响应字段、计时、完整清除与退出，再考虑正式测试。'),
    ]
    (output / '接口适配_本地验证报告.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Local mock only: frozen-policy HTTP integration regression')
    parser.add_argument('--output', required=True, type=storage.d_path)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        raise RuntimeError('Choose a new output directory; never overwrite an earlier verification run')
    protocol = {
        'purpose': 'engineering regression, not model tuning or official performance',
        'official_calls': 0, 'cases': CASES, 'frozen_source_hashes': FROZEN_HASHES,
        'interface_source_hashes': code_hashes(), 'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'shapely')},
        'document_sha256': {name: hashlib.sha256((ROOT / '附件' / name).read_bytes()).hexdigest()
                            for name in ('附件1.docx', '附件2.docx')},
        'recovery_cases': ['q3_random10', 'q4_boundary16'],
    }
    save(protocol_path, protocol)
    rows, cases = [], []
    for case_id, problem, count, profile, seed, network in CASES:
        sources, error_mode = frozen['offline_benchmark'].generate_scene(problem, count, profile, seed)
        cases.append({'id': case_id, 'seed': seed, 'error_mode': error_mode,
                      'sources': [source.__dict__ for source in sources]})
        save(output / 'synthetic_cases_evaluator_only.json', cases)
        world = OfflineRuleWorld(sources, seed, error_mode)
        stations = {'grid7': frozen['geometry'].seven_network, 'dual21': frozen['geometry'].dual_ring_network,
                    'grid25': lambda: frozen['geometry'].triangular_network()[0]}[network]()
        variant = 'E_joint' if problem == 'q3' else 'F_route'
        baseline = frozen['plan_policy'].PlanPolicy(world.port(), problem, stations, variant, network).run()
        if world.remaining:
            raise AssertionError(f'Frozen direct baseline failed: {case_id}')
        direct_signature = action_signature(world)
        modes = ('normal', 'recovery') if case_id in protocol['recovery_cases'] else ('normal',)
        for mode in modes:
            trial = f'{case_id}_{mode}'
            trial_dir = output / trial
            trial_dir.mkdir()
            faults = [] if mode == 'normal' else [
                Fault('/enter', 'drop_after'), Fault('/measure', 'drop_after'),
                Fault('/measure', 'truncate_after'), Fault('/clear', 'drop_after'), Fault('/exit', 'drop_after'),
            ]
            remote_world = OfflineRuleWorld(sources, seed, error_mode)
            started = time.perf_counter()
            with MockArena(remote_world, faults=faults) as arena, storage.Journal(trial_dir / 'requests.jsonl') as journal:
                client = client_module.RobotClient(arena.base_url, arena.robot_id, journal, mock_token=arena.token)
                summary = session_module.run_session(client, problem, 'grid25' if network == 'grid25' else 'dual21')
                summary['evaluator_remaining'] = sorted(remote_world.remaining)
                summary['endpoint'] = arena.base_url
                save(trial_dir / 'summary.json', summary)
                tolerance = (len(remote_world.trace) + 1) * 1e-6
                observed_signature = action_signature(remote_world)
                checks = {
                    'completed_and_exited': summary['status'] == 'completed',
                    'all_cleared': not remote_world.remaining,
                    'same_action_signature': observed_signature == direct_signature,
                    'same_declared_absent': summary.get('declared_absent') == baseline['declared_absent'],
                    'same_clock_within_microsecond_accumulation': abs(client.virtual_seconds - world.virtual_seconds) <= tolerance,
                    'client_and_server_clock_equal': client.virtual_seconds == remote_world.virtual_seconds,
                    'client_and_server_position_equal': client.position == tuple(remote_world.position),
                    'receiver_channel_equal': client.receiver_channel == remote_world.receiver_channel,
                    'one_execution_per_action': len(arena.executions) == len(world.trace) + 2,
                    'all_faults_exercised': not arena.faults,
                    'retry_count_expected': client.retry_count == (0 if mode == 'normal' else 5),
                }
                save(trial_dir / 'checks.json', checks)
                if not all(checks.values()):
                    raise AssertionError(f'{trial} failed: {checks}')
                row = {'trial': trial, 'problem': problem, 'network': network, 'source_count': count,
                       'cleared': len(client.cleared_channels), 'virtual_time_s': client.virtual_seconds,
                       'direct_virtual_time_s': world.virtual_seconds, 'retry_count': client.retry_count,
                       'physical_actions': len(remote_world.trace), 'action_sha256': observed_signature,
                       'wall_seconds': time.perf_counter() - started, 'passed': True}
                rows.append(row)
                save(output / 'results.json', rows)
                print(f'{trial}: cleared={count}/{count}; actions={row["physical_actions"]}; '
                      f'retries={client.retry_count}; sequence=identical; wall={row["wall_seconds"]:.2f}s', flush=True)
    if code_hashes() != protocol['interface_source_hashes']:
        raise RuntimeError('Interface source changed during validation')
    with (output / 'results.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report(output, rows)
    save(output / 'integration_complete.json', {'passed': True, 'runs': len(rows), 'scenes': len(CASES), 'official_calls': 0})
    print(f'Local HTTP verification complete: {output}', flush=True)


if __name__ == '__main__':
    main()
