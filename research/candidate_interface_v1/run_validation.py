from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import unittest

from candidate_setup import (
    Fault,
    MockArena,
    OfflineRuleWorld,
    candidate_module,
    candidate_session,
    client_module,
    frozen,
    frozen_hashes,
    storage,
    validation,
)
from test_candidate_interface import CandidateInterfaceTests

CASES = (
    ('random10', 10, 'random', 33111001), ('random16', 16, 'random', 33111002),
    ('boundary10', 10, 'boundary', 33111003), ('boundary16', 16, 'boundary', 33111004),
    ('clustered10', 10, 'clustered', 33111005), ('clustered16', 16, 'clustered', 33111006),
    ('near_origin10', 10, 'near_origin', 33111007), ('near_origin16', 16, 'near_origin', 33111008),
)
RECOVERY = ('random10', 'boundary16')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def signature(world):
    actions = [{key: entry[key] for key in ('action', 'channel', 'position', 'response')} for entry in world.trace]
    return hashlib.sha256(json.dumps(actions, sort_keys=True, allow_nan=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description='Only self-created random-port mock HTTP; no official simulator calls.')
    parser.add_argument('--output', type=storage.d_path, required=True)
    args = parser.parse_args()
    output = args.output
    if not output.is_relative_to(validation.VALIDATION_ROOT) or output == validation.VALIDATION_ROOT:
        parser.error('Output must be a new child of the candidate validation root')
    output.mkdir(parents=True, exist_ok=False)
    os.environ['B2026_CANDIDATE_VALIDATION_OUTPUT'] = str(output)
    protocol = {'problem': 'q3', 'mode': 'combined', 'cases': CASES, 'recovery_cases': RECOVERY,
                'source_hashes': validation.source_hashes(), 'frozen_source_hashes': frozen_hashes,
                'candidate_hashes': validation.verify_candidate(), 'official_calls': 0,
                'purpose': 'Engineering equivalence and fault recovery; not algorithm tuning',
                'stopping_rule': 'Do not authorize practice if any test or integration check fails; retain evidence.'}
    save(output / 'protocol.json', protocol)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CandidateInterfaceTests)
    with (output / 'unit_tests.log').open('w', encoding='utf-8') as stream:
        unit_result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    save(output / 'unit_results.json', {'run': unit_result.testsRun, 'failures': len(unit_result.failures),
                                      'errors': len(unit_result.errors), 'passed': unit_result.wasSuccessful()})
    if not unit_result.wasSuccessful():
        raise RuntimeError('Candidate unit tests failed; evidence preserved, no practice authorization')
    print(f'Candidate unit tests: {unit_result.testsRun} passed', flush=True)
    rows, case_records = [], []
    for case_id, count, profile, seed in CASES:
        sources, error_mode = frozen['offline_benchmark'].generate_scene('q3', count, profile, seed)
        case_records.append({'id': case_id, 'seed': seed, 'error_mode': error_mode,
                             'sources': [source.__dict__ for source in sources]})
        save(output / 'synthetic_cases_evaluator_only.json', case_records)
        direct = OfflineRuleWorld(sources, seed, error_mode)
        policy = candidate_module.CandidatePolicy(direct.port(), 'q3', frozen['geometry'].seven_network(),
                                                 'E_joint', 'grid7', mode='combined')
        result = policy.run()
        if direct.remaining:
            raise AssertionError(f'Direct frozen candidate failed: {case_id}')
        expected_signature = signature(direct)
        for mode in (('normal', 'recovery') if case_id in RECOVERY else ('normal',)):
            trial = f'{case_id}_{mode}'
            trial_dir = output / trial
            faults = [] if mode == 'normal' else [
                Fault('/enter', 'drop_after'), Fault('/measure', 'drop_after'),
                Fault('/measure', 'truncate_after'), Fault('/clear', 'drop_after'), Fault('/exit', 'drop_after'),
            ]
            remote = OfflineRuleWorld(sources, seed, error_mode)
            started = time.perf_counter()
            with MockArena(remote, faults=faults) as arena, storage.Journal(trial_dir / 'requests.jsonl') as journal:
                client = client_module.RobotClient(arena.base_url, arena.robot_id, journal, mock_token=arena.token)
                summary = candidate_session.run_session(client, 'q3')
                save(trial_dir / 'summary.json', summary)
                checks = {
                    'completed_and_exited': summary['status'] == 'completed' and client.exited,
                    'all_cleared': not remote.remaining and len(client.cleared_channels) == count,
                    'same_actions_and_responses': signature(remote) == expected_signature,
                    'same_absent_channels': summary.get('declared_absent') == result['declared_absent'],
                    'same_policy_stats': summary.get('policy_stats') == result['stats'],
                    'same_clock': abs(client.virtual_seconds - direct.virtual_seconds) <= (len(remote.trace) + 1) * 1e-6,
                    'authoritative_clock_exact': client.virtual_seconds == remote.virtual_seconds,
                    'position_exact': client.position == tuple(remote.position),
                    'receiver_exact': client.receiver_channel == remote.receiver_channel,
                    'one_execution_per_unique_action': len(arena.executions) == len(direct.trace) + 2,
                    'all_faults_exercised': not arena.faults,
                    'retries_exact': client.retry_count == (5 if mode == 'recovery' else 0),
                }
                row = {'trial': trial, 'source_count': count, 'cleared': len(client.cleared_channels),
                       'virtual_seconds': client.virtual_seconds, 'retry_count': client.retry_count,
                       'physical_actions': len(remote.trace), 'action_sha256': signature(remote),
                       'wall_seconds': time.perf_counter() - started, 'passed': all(checks.values())}
                save(trial_dir / 'checks.json', checks)
                rows.append(row)
                save(output / 'results.json', rows)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                if not all(checks.values()):
                    raise AssertionError(f'{trial} failed: {checks}')
    if validation.source_hashes() != protocol['source_hashes']:
        raise RuntimeError('Sources changed during validation')
    proof = {'passed': True, 'problem': 'q3', 'candidate_mode': 'combined', 'unit_tests': unit_result.testsRun,
             'synthetic_scenes': len(CASES), 'http_runs': len(rows), 'all_cleared_http_runs': len(rows),
             'matched_physical_actions': sum(row['physical_actions'] for row in rows),
             'recovered_retries': sum(row['retry_count'] for row in rows), 'official_calls': 0}
    save(output / 'verification.json', proof)
    lines = ['# 问题3候选：本地HTTP一致性与故障验证', '',
             '范围：冻结v3 combined / grid7；原v2机器人和候选算法文件均不修改。',
             '传输、权威计时沿用原客户端；新会话控制流通过AST检查与原会话一致。只使用随机回环端口上的自建假接口。', '',
             '| 场景 | 全清 | 虚拟秒 | 重试 | 物理动作 | 序列与离线一致 |',
             '|---|---|---:|---:|---:|---|']
    for row in rows:
        lines.append(f'| {row["trial"]} | {row["cleared"]}/{row["source_count"]} | {row["virtual_seconds"]:.6f} | '
                     f'{row["retry_count"]} | {row["physical_actions"]} | 是 |')
    lines.extend(['', f'通过{unit_result.testsRun}项候选适配单元测试和{len(rows)}次HTTP全流程。',
                  '故障覆盖四端点执行后断线与测量应答截断；请求ID和字节请求体不变，无重复物理执行。',
                  '短/零预算、计时错误、重试耗尽、计算异常与看门狗退出均测试；异常不能被标成成功。',
                  '未验证官方登录、服务器配额、加密日志上传或隐藏案例分布。实际演练必须另行查看公开界面确认问题3演练已就绪。',
                  '该验证只证明所测工程路径一致性，不增加算法泛化或必然全清的理论保证。'])
    (output / '问题3候选_HTTP验证报告.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (validation.VALIDATION_ROOT / 'LATEST_SUCCESS.txt').write_text(str(output), encoding='utf-8')
    print(json.dumps(proof, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
