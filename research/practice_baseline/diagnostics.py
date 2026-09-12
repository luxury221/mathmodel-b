from __future__ import annotations

import json
import math
import sys
from collections import defaultdict

import numpy as np
from baseline import OUTPUT, ROOT, load, save, verify_hashes

sys.path.insert(0, str(ROOT / 'research' / 'offline_validation'))

from geometry import dual_ring_network, seven_network
from offline_benchmark import ObservationPort
from plan_policy import PlanPolicy


def committed_actions(folder):
    entries = [json.loads(line) for line in (folder / 'requests.jsonl').read_text(encoding='utf-8').splitlines()]
    requests = {entry['payload']['request_id']: entry for entry in entries if entry['event'] == 'request'}
    responses = {entry['request_id']: json.loads(entry['body_utf8']) for entry in entries if entry['event'] == 'response'}
    actions = []
    for entry in entries:
        if entry['event'] != 'committed' or entry['path'] not in ('/measure', '/clear'):
            continue
        request = requests[entry['request_id']]
        response = responses[entry['request_id']]
        if not response['accepted']:
            raise ValueError('A committed action was rejected')
        payload = request['payload']
        actions.append({
            'operation': entry['path'][1:], 'channel': payload['channel'],
            'position': [payload['position']['x'], payload['position']['y']], 'response': response,
        })
    return actions


class RecordedPort:
    def __init__(self, actions):
        self.actions = actions
        self.index = 0

    def respond(self, operation, position, channel):
        if self.index >= len(self.actions):
            raise ValueError('Replay attempted an unrecorded action')
        action = self.actions[self.index]
        if (action['operation'] != operation or action['channel'] != channel
                or not np.allclose(position, action['position'], rtol=0, atol=1e-7)):
            raise ValueError(f'Replay diverged at action {self.index}')
        self.index += 1
        response = action['response']
        if operation == 'clear':
            return {'result': response['clear_result']}
        result = {'result': response['measure_result']}
        if result['result'] == 'direction':
            result['bearing_deg'] = response['svd_deg']
        return result

    def measure(self, position, channel):
        return self.respond('measure', position, channel)

    def clear(self, position, channel):
        return self.respond('clear', position, channel)


class TracedPolicy(PlanPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase = 'discovery'
        self.reason = 'discovery'
        self.costs = defaultdict(lambda: {'actions': 0, 'move_m': 0.0, 'virtual_s': 0.0, 'failed_clears': 0})
        self.clear_episodes = []

    def account(self, position, channel, operation, result):
        distance = math.dist(self.position, position)
        before = self.virtual_seconds
        super().account(position, channel, operation, result)
        cost = self.costs[self.reason]
        cost['actions'] += 1
        cost['move_m'] += distance
        cost['virtual_s'] += self.virtual_seconds - before
        cost['failed_clears'] += int(operation == 'clear' and result != 'success')

    def measure(self, channel, position, active=False, opportunistic=False):
        self.reason = 'opportunistic_measure' if opportunistic else 'active_measure' if active else 'discovery'
        return super().measure(channel, position, active, opportunistic)

    def execute_clear_plan(self, state, plan, opportunistic=False):
        self.reason = ('opportunistic' if opportunistic else 'cleanup') + '_' + plan.certificate_kind
        before = self.virtual_seconds
        super().execute_clear_plan(state, plan, opportunistic)
        self.clear_episodes.append({
            'channel': state.channel, 'reason': self.reason, 'planned_centers': len(plan.centers),
            'virtual_s': self.virtual_seconds - before,
        })

    def clear_near(self, state, opportunistic=False):
        self.reason = 'opportunistic_near' if opportunistic else 'cleanup_near'
        return super().clear_near(state, opportunistic)


def diagnose(run):
    folder = ROOT / 'logs' / 'practice' / run['run_id']
    recorded = RecordedPort(committed_actions(folder))
    stations = seven_network() if run['problem'] == 'q3' else dual_ring_network()
    policy = TracedPolicy(ObservationPort(recorded.measure, recorded.clear), run['problem'], stations,
                         run['variant'], run['network'])
    result = policy.run()
    if recorded.index != len(recorded.actions) or set(result['cleared_channels']) != set(run['cleared_channels']):
        raise ValueError('Incomplete replay')
    if abs(result['virtual_seconds'] - run['virtual_s']) > (recorded.index + 1) * 1e-6:
        raise ValueError('Replay timing mismatch')
    return {
        'run_id': run['run_id'], 'problem': run['problem'], 'kind': run['kind'],
        'matched_actions': recorded.index, 'costs': dict(policy.costs), 'clear_episodes': policy.clear_episodes,
    }


def main():
    verify_hashes()
    runs = load(OUTPUT / 'runs.json')
    results = [diagnose(run) for run in runs]
    save(OUTPUT / 'diagnostics.json', results)
    lines = [
        '# 冻结策略耗时诊断', '',
        '仅顺序回放机器人自己已获准取得的测量与清除响应；不连接模拟器，不读取隐藏真值。',
        '全部动作的位置、频道、类型及结束状态与原日志核对。此处是日志重放，不是重跑官方案例。', '',
        '| 问题 | 动作分类（新增批次） | 动作数 | 移动米数 | 虚拟秒数 | 占该题总时间 |',
        '|---|---|---:|---:|---:|---:|',
    ]
    aggregate = {}
    for problem in ('q3', 'q4'):
        costs = defaultdict(lambda: {'actions': 0, 'move_m': 0.0, 'virtual_s': 0.0, 'failed_clears': 0})
        for result in results:
            if result['problem'] != problem or result['kind'] != 'batch':
                continue
            for reason, values in result['costs'].items():
                for name, value in values.items():
                    costs[reason][name] += value
        total = sum(cost['virtual_s'] for cost in costs.values())
        for reason, cost in sorted(costs.items(), key=lambda item: -item[1]['virtual_s']):
            lines.append(f'| {problem} | {reason} | {cost["actions"]} | {cost["move_m"]:.1f} | '
                         f'{cost["virtual_s"]:.1f} | {cost["virtual_s"] / total:.1%} |')
        aggregate[problem] = dict(costs)
    lines.extend(['', '## 解释',
                  '- 移动按下一条动作归属，仅作路径成本分解，不把它解释为该动作的可完全消除成本。',
                  '- 同样的站网，改变访问顺序、提前清除或追加测量会改变后续响应；不能直接用旧官方日志评分新策略。',
                  '- 优化应在新合成场景中配对运行，保留未命中负信息、连续区域清除证书和有限兜底。'])
    (OUTPUT / '冻结策略_耗时诊断.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'matched_actions': sum(result['matched_actions'] for result in results),
                      'runs': len(results), 'aggregate': aggregate}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
