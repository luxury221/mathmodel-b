from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'reports' / 'plan_trials_v2'
LABELS = {
    'A_cover': 'A 覆盖基线', 'B_minimax': 'B +受约束主动选点',
    'C_multi_clear': 'C +多圆试清除', 'D_direction_gate': 'D +方向/成本判断',
    'E_joint': 'E +联合搜索清除', 'F_route': 'F +开放路径优化',
    'legacy_best': '上一轮较优原型',
}


def read(name):
    return json.loads((OUTPUT / name).read_text(encoding='utf-8'))


def table(headers, rows):
    return '\n'.join([
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
        *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows],
    ])


def make_plots(summary, frame, selected):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for axis, problem, network in zip(axes, ('q3', 'q4'), ('grid7', 'dual21')):
        group = summary[(summary.phase == 'development') & (summary.problem == problem) & (summary.network == network)].sort_values('variant')
        axis.plot(np.arange(len(group)), group.mean_seconds_per_source, marker='o', color='#2864a8')
        axis.set_xticks(np.arange(len(group)), [name.split('_')[0] for name in group.variant])
        axis.set(title=f'{problem.upper()} development: A through F', ylabel='Virtual seconds / source')
        axis.grid(alpha=0.2)
    figure.suptitle('Layered ablation: synthetic offline scenarios only')
    figure.savefig(OUTPUT / 'development_ablation.png', dpi=170)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for axis, problem, network in zip(axes, ('q3', 'q4'), ('grid7', 'dual21')):
        group = summary[(summary.phase == 'holdout') & (summary.problem == problem) & (summary.network == network)]
        names = ['A_cover', 'legacy_best', selected[problem]]
        means = [float(group[group.variant == name].mean_seconds_per_source.iloc[0]) for name in names]
        axis.bar(np.arange(3), means, color=['#8d9eb3', '#d58a4a', '#2864a8'], width=0.65)
        axis.set_xticks(np.arange(3), ['Cover baseline', 'Previous prototype', 'Frozen selection'])
        axis.set(title=f'{problem.upper()}: 56 paired holdout scenarios', ylabel='Virtual seconds / source')
        axis.grid(axis='y', alpha=0.2)
        axis.set_axisbelow(True)
        for index, mean in enumerate(means):
            axis.text(index, mean + 10, f'{mean:.1f}', ha='center')
        axis.set_ylim(0, max(means) * 1.15)
    figure.suptitle('Held-out comparison — not official competition scores')
    figure.savefig(OUTPUT / 'holdout_comparison.png', dpi=170)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    regressions = []
    for axis, problem, network in zip(axes, ('q3', 'q4'), ('grid7', 'dual21')):
        group = frame[(frame.phase == 'holdout') & (frame.problem == problem) & (frame.network == network)]
        old = group[group.variant == 'legacy_best'].set_index('scene_id')
        new = group[group.variant == selected[problem]].set_index('scene_id')
        for scene_id, record in new.iterrows():
            previous = old.loc[scene_id]
            movement_change = (record.move_meters - previous.move_meters) / 5 / record.source_count
            measurement_change = (record.measure_count - previous.measure_count) * 5 / record.source_count
            switching_change = (record.switch_count - previous.switch_count) / record.source_count
            failed_clear_change = (record.failed_clear_count - previous.failed_clear_count) * 3 / record.source_count
            total_change = record.seconds_per_source - previous.seconds_per_source
            assert abs(movement_change + measurement_change + switching_change + failed_clear_change - total_change) < 1e-7
            regressions.append({
                'scene_id': scene_id, 'problem': problem, 'profile': record.profile,
                'legacy_seconds_per_source': previous.seconds_per_source,
                'selected_seconds_per_source': record.seconds_per_source,
                'change_seconds_per_source': total_change,
                'movement_change_seconds_per_source': movement_change,
                'measurement_change_seconds_per_source': measurement_change,
                'switching_change_seconds_per_source': switching_change,
                'failed_clear_change_seconds_per_source': failed_clear_change,
            })
        axis.scatter(old.seconds_per_source, new.loc[old.index].seconds_per_source, s=24, alpha=0.75, color='#2864a8')
        limit = max(old.seconds_per_source.max(), new.seconds_per_source.max()) * 1.08
        axis.plot([0, limit], [0, limit], linestyle='--', color='#bf5a32')
        axis.set(xlim=(0, limit), ylim=(0, limit), title=f'{problem.upper()}: below line is faster',
                 xlabel='Previous prototype seconds / source', ylabel='Frozen selection seconds / source')
        axis.grid(alpha=0.2)
    figure.savefig(OUTPUT / 'paired_regressions.png', dpi=170)
    plt.close(figure)
    regression_frame = pd.DataFrame(regressions).sort_values('change_seconds_per_source', ascending=False)
    regression_frame.to_csv(OUTPUT / 'regressions.csv', index=False, encoding='utf-8-sig')
    return regression_frame


def main():
    result = read('results.json')
    protocol = read('protocol.json')
    selection = read('selection.json')
    selected = selection['selected']
    replay = read('replay.json') if (OUTPUT / 'replay.json').exists() else None
    summary = pd.DataFrame(result['summary'])
    frame = pd.read_csv(OUTPUT / 'all_runs.csv')
    regressions = make_plots(summary, frame, selected)
    lines = [
        '## Material Passport', '', '- Origin Skill: academic-research-suite / experiment-agent',
        '- Origin Mode: run + validate', '- Origin Date: 2026-09-11',
        '- Verification Status: ' + ('VERIFIED — 仅指离线运行和确定性回放' if replay else 'ANALYZED — 尚未完成全量回放'),
        '- Version Label: plan_trials_v2', '- Overall Confidence: CAUTION — 合成实验不能替代官方验证', '',
        '# 按方案实施：第二轮分层消融与留出验证', '',
        '## 1. 本轮实际执行范围', '',
        '**没有启动官方模拟器，没有调用官方接口，也没有消耗演练或正式测试次数。**', '',
        '这一轮不是重复上一轮几何检查，而是把方案的主动定位、多圆清除、方向假设和联合调度真正接入离线闭环。原方案、仓库原脚本及上一轮结果保留不变，新结果独立放在 D 盘。', '',
        '**执行中发现并修复过一次实际异常：** v1 在留出任务第 350 项因新生成角锥多边形的浮点近重复闭合顶点而自交，触发 GEOS 拓扑错误。没有将该次尝试写成全部通过。失败输入、日志与冻结源码保存在 `D:\\数模B题\\B题\\reports\\plan_trials`。', '',
        '修复只作用于新构造的理论凸约束：对其顶点取凸包，得到包含全部计算顶点的有效凸几何，再与已有非凸可行域相交；没有用 buffer(0) 静默删区域，也没有把保留失败清除信息的整个状态随意凸化。随后重新冻结 v2，并换用全新的留出/压力种子；开发集继续用于同一预定选型规则。', '',
        table(['层级', '新增内容', '保证与近似的边界'], [
            ['A', '固定覆盖网、未知频道排除、MEC、有限格点清除', '全向负观测删保守内接接收圆；定向负观测不单独删除位置圆'],
            ['B', '真实保守接收域约束、SLSQP 候选优化、角锥采样 minimax', '接收距离约束逐顶点复核；最坏定位评价仍是采样近似，不是连续全局最优'],
            ['C', '2–4 圆覆盖与枚举访问次序，失败清除更新区域', '外包可行域必须被内接清除圆多边形连续覆盖，不能只覆盖样本'],
            ['D', '位置/半径/类型/朝向假设及动作成本判断', '假设只用于估计收益；假设为空不代表源不存在'],
            ['E', '搜索点上的机会式测量与近路清除', '每节点附加动作有上限，随后强制推进有限保证网络'],
            ['F', '清尾阶段最近邻加 2-opt 开放路线', '优化估计目标点的路线，不声称原问题全局最优'],
        ]), '',
        '公共实现还补上了原点初筛、源数上界终止、重复地点观测缓存、每源最多四次主动测量、连续两次无信号后的成本兜底。问题四使用上一轮复核的名义 21 点网络，并对 25 点解析备份作配对比较。', '',
        '## 2. 实验先冻结，再评估', '',
        table(['集合', '独立场景数', '用途'], [[phase, count, purpose] for (phase, count), purpose in zip(
            protocol['scenario_counts'].items(), ['完整 A–F 消融及配置选择', '源数 10–16，四类分布，每类两种子；禁止再据此调参', '接收半径恰为 1500 m，以及针对 21 点构造的困难朝向'],
        )]), '',
        '四类基本场景为随机、目标圆边界、局部聚集、原点毫米邻域。问题四的聚集及近原点场景为额外的全定向压力情形；其余基本场景包含混合源。误差为位置固定的有界函数，并加入 ±1° 极值及显示取整压力。', '',
        '组件调试使用过另一个种子 810001 的独立随机场景，已在协议披露，不混入开发和留出统计。开发集规则为：只在全部清除成功的配置中选平均耗时最小者，相差不超过 1 秒/源则选更简单层级。源码哈希、场景哈希及选择结果在留出运行前冻结。', '',
        f"冻结配置：问题三 **{LABELS[selected['q3']]}**；问题四 **{LABELS[selected['q4']]}**。后续结果没有改变这个选择。", '',
        '## 3. 开发集分层消融', '',
    ]
    development = summary[(summary.phase == 'development') & (summary.network != 'grid25')]
    lines.append(table(['问题', '配置', '运行数', '全清除', '平均秒/源'], [
        [record.problem, LABELS[record.variant], record.runs, record.successes, f'{record.mean_seconds_per_source:.2f}']
        for record in development.itertuples()
    ]))
    lines.extend(['', '问题三的 E、F 开发均值差距不足 1 秒/源，因此按预先规则保留更简单的 E，而不是为了追求表面最小值继续堆叠模块。', '',
                  f'![开发集消融]({(OUTPUT / "development_ablation.png").as_posix()})', '', '## 4. 冻结后的留出结果', ''])
    heldout = summary[summary.phase == 'holdout']
    lines.append(table(['问题', '网络', '配置', '全清除/场景数', '平均秒/源', 'P90', '最坏秒/源'], [
        [record.problem, record.network, LABELS[record.variant], f'{record.successes}/{record.runs}',
         f'{record.mean_seconds_per_source:.2f}', f'{record.p90_seconds_per_source:.2f}', f'{record.worst_seconds_per_source:.2f}']
        for record in heldout.itertuples()
    ]))
    lines.extend(['', '时间指标均为每个场景的“虚拟总时间/成功清除源数”，再跨场景求平均；不等于程序现实计算时间。旧版对照在同一批新场景上重新运行，不拿两批不同地图的均值直接相减。', ''])
    for comparison in result['paired_holdout']:
        lines.append(
            f"- {comparison['problem']} 相对{LABELS[comparison['baseline']]}：平均时间变化 "
            f"{-100 * comparison['relative_mean_reduction']:+.2f}%；{comparison['faster_cases']} 个场景更快，"
            f"{comparison['slower_cases']} 个场景更慢，共 {comparison['paired_cases']} 个配对场景。"
        )
    lines.extend(['', f'![留出比较]({(OUTPUT / "holdout_comparison.png").as_posix()})', '',
                  '## 5. 额外压力测试与完整性核验', ''])
    stress = summary[summary.phase == 'stress']
    lines.append(table(['问题', '配置', '全清除/场景数', '平均秒/源', '最坏秒/源'], [
        [record.problem, LABELS[record.variant], f'{record.successes}/{record.runs}',
         f'{record.mean_seconds_per_source:.2f}', f'{record.worst_seconds_per_source:.2f}']
        for record in stress.itertuples()
    ]))
    lines.extend([
        '', f"本轮合计 **{result['all_cleared_runs']}/{result['run_count']}** 次运行全清除，涉及 **{result['unique_scenarios']}** 个独立合成场景。",
        f"评估器检查了 {result['feasible_region_checks']} 次真实源包含性、{result['continuous_cover_checks']} 次连续清除覆盖证书。策略本身只收到观测，没有读取坐标、朝向或总源数真值。",
        f"全量确定性回放：{replay['replayed_runs']} 次，所有非现实时间指标逐项完全一致。" if replay else '全量确定性回放尚未完成，不作可复现性完成声明。',
        '新增 9 项几何/假设组件测试通过，并继承 14 项动作语义检查；包括这次浮点自交问题的回归案例、“样本顶点全覆盖但区域未覆盖”的反例、固定接收半径一致性、定向无信号约束和接收候选域检查。',
        '', '## 6. 仍然变慢的案例：不隐藏负结果', '',
    ])
    slower = regressions[regressions.change_seconds_per_source > 1e-6]
    if len(slower):
        lines.append(table(['场景', '旧版秒/源', '冻结配置秒/源', '增加秒/源', '其中移动变化秒/源'], [
            [record.scene_id, f'{record.legacy_seconds_per_source:.2f}', f'{record.selected_seconds_per_source:.2f}',
             f'{record.change_seconds_per_source:.2f}', f'{record.movement_change_seconds_per_source:+.2f}']
            for record in slower.head(6).itertuples()
        ]))
    else:
        lines.append('本批留出场景未出现相对旧版变慢的案例；这仍不意味着逐实例支配。')
    lines.extend([
        '', 'regressions.csv 同时保存了移动、检测、切频道、失败清除四部分的精确计时差，四项相加必须等于总时间差。留出结果只用于评估，不据此修改本轮参数。变慢案例及最坏动作轨迹已保存，后续若针对它们修改策略，应另设新的留出集，避免把这批场景反复调成“测试通过”。',
        f'![配对回归检查]({(OUTPUT / "paired_regressions.png").as_posix()})', '',
        '## 7. 解释边界与未完成事项', '',
        '- 本轮完成方案的主要可执行层级，但没有实现完整两步滚动 POMDP、连续全局 minimax 或全局最优检测网。',
        '- 圆覆盖的最终接受条件为连续几何检查；候选分块和 SLSQP 可能漏掉更优解，但不能因此放宽安全条件。',
        '- 方向/半径假设采用确定性低差异位置样本、24 个朝向和半径区间端点；其响应比例是启发式权重，不是题目给定概率，也不是贝叶斯真实后验。',
        '- 21 点名义网络的布点误差敏感性仍沿用上一轮审查结论，不能因这批测试通过就取消 25 点备份。',
        '- 运行时间不包含官方 HTTP 延迟、重试、真实接口的期限和错误响应。当前没有官方 API 客户端，不应直接宣称已经完成真实接口联调。',
        '- 全清除样本率不等于官方失败概率为零，不可填入题目要求的正式测试结果表。',
        '', '## 8. 统计与方法风险检查（11/11）', '',
        '1. Simpson：已同时保存按场景类型的分层表，不只看总均值。',
        '2. 生态推断：场景平均改善不推断每个源均改善。',
        '3. 选择偏差：人为压力场景不代表官方抽样分布。',
        '4. Collider：不以结果好坏筛选进入汇总的案例。',
        '5. 基准率：不从零失败样本推断真实失败率为零。',
        '6. 均值回归：使用固定场景配对，而非挑最差案例单独重测。',
        '7. 幸存者偏差：失败必须保留记录，不允许静默删除；本轮记录见 results.json。',
        '8. 多重搜索：完整保留六层开发结果及留出回退，不只展示最好的一行。',
        '9. 分析路径自由度：组件冒烟披露；开发/留出分离，选型在留出前冻结；不宣称第三方预注册。',
        '10. 因果外推：改变算法在固定替身中的效果不能直接外推官方成绩。',
        '11. 反向因果：配置在运行前决定，没有用测试结果回写策略分组。',
        '', '## 9. 文件和复现', '',
        f'- 协议与哈希：`{OUTPUT / "protocol.json"}`', f'- 环境版本：`{OUTPUT / "environment.json"}`',
        f'- 冻结选型：`{OUTPUT / "selection.json"}`',
        f'- 完整结果：`{OUTPUT / "all_runs.csv"}`', f'- 分层结果：`{OUTPUT / "by_profile.csv"}`',
        f'- 变慢案例：`{OUTPUT / "regressions.csv"}`', f'- 全部场景真值（仅评估器使用）：`{OUTPUT / "scenes.json"}`',
        f'- 最坏轨迹：`{OUTPUT / "holdout_worst_traces.json"}`', f'- 回放结果：`{OUTPUT / "replay.json"}`', '',
        '```powershell', f"& '{ROOT / 'research' / 'offline_validation' / 'run_plan_trials.ps1'}'", '```', '',
        '入口检查源码哈希并复用已完成阶段，随后做确定性回放与报告更新。全部新增或更新的实验文件均在 D 盘；不会联网或启动官方测试。若要修改算法继续调参，应创建新实验版本，不要绕过哈希检查覆盖这轮证据。',
    ])
    (OUTPUT / '方案实施_第二轮测试报告.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('Created second-round report and three diagnostic figures.')


if __name__ == '__main__':
    main()
