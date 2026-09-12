# 候选优化 v3（仅离线）

## 状态与隔离

原 `research/offline_validation` 五个冻结文件、`src/b2026_robot` 接口和官方演练入口均不修改。候选仅从 `ObservationPort` 取得测量/清除反馈。`experiments.py` 在策略返回后由独立评估器读取**自建合成场景**真值，检查全清除、负信息更新、连续覆盖和时间一致性；策略本身不读取真值。这里没有连接官方端口的代码。

## 两项改动

1. **joint：沿途联合测清。** 复用原来的顺手测量，并允许连续可行域可由1—4个圆覆盖时提前清除。对可能在任意第 `k` 个中心成功的情况，检查增量

   `prefix_distance / 5 + 3 * (k - 1) + 5 + (distance(center_k, next) - distance(current, next)) / 5`。

   所有成功前缀的最大增量不超过120秒才加入候选，每个停点最多额外清除4源。清理阶段到达一个源后，也复用原来的顺手观测和邻近清除。这个门限只限制**单次插入**相对于当前下一锚点的增量，不保证整个任务必然更短。

2. **prune：几何剔除空兜底点。** 原矩形兜底中心全部保留其连续覆盖证书。执行前删除到当前可行域的距离大于20.000001米的中心，其余按当前距离就近执行。因为真实源仍在保守可行域内，被删除的清除圆不可能包含真实源。每步执行或删除至少一个中心，保留有限终止性。浮点容差只让剔除更保守，不将内接多边形“不相交”误当成真实圆“不相交”。

`combined` 同时启用两项；`baseline` 使用原策略。原站点集合、访问次序、方向源负测量语义、近场清除、接收距离约束均不变。问题4原本已使用清理阶段2-opt，这里不重复把2-opt当新算法。

## 可复核实验

在项目根目录使用 PowerShell，先运行 `. .\scripts\env.ps1`。它将解释器、缓存、临时文件和结果保持在D盘。

```powershell
& $PythonExe research/policy_optimization_v3/test_candidate.py
& $PythonExe research/policy_optimization_v3/experiments.py prepare
& $PythonExe research/policy_optimization_v3/experiments.py run --phase development
& $PythonExe research/policy_optimization_v3/experiments.py select
& $PythonExe research/policy_optimization_v3/experiments.py run --phase holdout
& $PythonExe research/policy_optimization_v3/experiments.py run --phase stress
& $PythonExe research/policy_optimization_v3/experiments.py report
& $PythonExe research/policy_optimization_v3/experiments.py replay
```

输出：`reports/policy_optimization_v3`。已启动的阶段禁止直接覆盖重跑；代码哈希在首次 `prepare` 时冻结，代码变动必须另开版本。完整复现已有实验使用 `replay`，它保留原记录并比对每条动作的完整轨迹哈希。

开发集包含24场景、4个策略臂。每题12场景全部全清、单例时间不超过原策略的1.25倍才有选型资格；按场景等权的秒/源均值选择，1秒以内依次优先原策略、prune、joint、combined。然后冻结选型，使用64个全新留出场景和16个压力场景，后者另加问题4 `grid25` 备用网络。留出结果和压力结果不用于改门限或重选策略。

压力场景标签沿用合成生成器名称。`adversarial_heading` 在问题4中表示难接收方向角；在问题3中所有源仍为全向，该标签实际表示边界位置、1000米半径与空间固定的极端误差，不表示问题3混入了定向源。

## 使用限制

本文件中的改动是待检验假设，不是已证明的效率提升。真实官方案例不能用改策略后的请求重放旧响应来评分。上线前仍需独立的HTTP一致性、重试幂等性、时间预算验证，以及仅限**演练**的新案例测试；不自动启动正式测试。
