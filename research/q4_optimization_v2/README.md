# 问题四独立优化 v2

本目录研究问题四的低移动成本补测，不修改当前机器人、问题3候选或既有问题4策略。所有运行均为本地合成环境；`http_selected.py`也只创建自建MockArena，没有官方地址或正式测试入口。

## 四个新候选

- `scan`：在既有joint候选上，利用机器人当前所在位置补测最多8个已经发现但尚未定位清除的频道。不新增移动；测量和切频照常计费。优先考虑可行域靠近当前位置的目标，或预测接收概率及交会角达到固定阈值的目标。
- `tour_scan`：保留原21个坐标的逐位表示，额外尝试多起点最近邻与2-opt重排，仅采用静态路径更短的顺序。不能假定改变顺序一定改善总任务时间。
- `outer_scan`：同样保留完整站点集合，先外环再内环。作为发现顺序的消融对照，不默认采用。
- `probe_scan`：大规模认证清除之前，最多每频道增加1次受预算限制的测向；即时成本不超过90虚拟秒，也不超过原清除计划最坏成本的20%。测量后仍使用认证清除计划。

所有候选继承原搜索终止条件和有限清除兜底。定向负信息不直接删除源位置；无“提前假定源数”“无证书清除覆盖”或同位置反复测量取平均。

## 评估约束

先冻结24个开发场景及6个配置（原策略、旧joint和4个新候选），开发集选型后才读取48个留出与16个压力场景结果。所有种子均与已记录的旧实验集合核对无重叠。

开发集候选必须全清，单例相对原策略最坏变慢不超过20%，每个分布组的均值不比原策略慢5%以上。通过后按案例等权均值选择；距最佳不超过1秒/源时优先更简单的配置。这只是选型筛选规则，不是留出集或官方案例的风险保证。

留出和压力集只比较原策略、旧joint和已冻结的获选方案。所有失败、慢例和超时保留，不允许根据留出结果回头调整本版本。重复重放不增加独立样本量。

每个场景由单独子进程执行，墙钟上限180秒。主要效率指标是虚拟秒/源，不是受CPU负载影响的墙钟时间。评估器分别核对真源包含、连续清除证书、全清及虚拟计时。

## 本地执行顺序

```powershell
. .\scripts\env.ps1
$Output = Join-Path $ProjectRoot ('reports\q4_optimization_v2\run_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
& $PythonExe research/q4_optimization_v2/experiment.py prepare --output $Output
& $PythonExe research/q4_optimization_v2/experiment.py run --phase development --output $Output
& $PythonExe research/q4_optimization_v2/experiment.py select --output $Output
& $PythonExe research/q4_optimization_v2/experiment.py run --phase holdout --output $Output
& $PythonExe research/q4_optimization_v2/experiment.py run --phase stress --output $Output
& $PythonExe research/q4_optimization_v2/experiment.py replay --output $Output
& $PythonExe research/q4_optimization_v2/http_selected.py --output $Output
& $PythonExe research/q4_optimization_v2/experiment.py report --output $Output
```

任一步失败都应停止后续命令并保留证据，不能覆盖旧批次。使用相同版本再次完整执行只是重现同一88个场景，不是取得另一批独立样本。

本地候选HTTP测试只临时替换原会话的策略构造函数，并检查继续沿用原权威计时实现；退出、重试和看门狗控制流不改。结束后恢复构造函数。通过该测试也不代表已获得官方演练或正式测试授权。

`protocol.json`保存源码、场景和旧输入哈希；`selection.json`保存开发集选型及逐条结果哈希；`attempts`、`records`、`traces`、`http`保存完整证据。代码、环境缓存、临时文件及产物均在D盘。
