# 问题四专项本地验证

本目录只新增评估程序，不修改冻结算法、机器人入口或问题3候选。没有官方模拟器地址参数，也不会启动登录、官方演练或正式测试。

## 测试设计

- 64个预先固定的新合成场景：6类分布各8个，加16个方向与位置边界场景。
- 三个配置逐场景配对运行，共192次：原 `F_route/dual21`、现有 `joint/dual21` 候选、原 `F_route/grid25` 备用网。
- 专项位置尺度包括近原点、5米附近、1000米附近、1800米边界附近；方向从固定网格与接收半平面边界中选择低可见方向。
- 只使用冻结的真实站网坐标，不把坐标扰动后的方案混入默认策略结果。
- 每个场景在独立子进程中依次运行三个配置，120秒墙钟上限；失败、超时、缺失结果均保留，不当成耗时改善。
- 对每个配置预设重放一个慢例和一个固定随机例，仅检查确定性，不增加样本量。
- 原策略在4个预定场景上分别验证两种站网，再增加2次故障版本，共10次自建回环HTTP全流程。没有把未验证的q4候选接入正式或演练入口。

所有场景真值只用于评估器构造环境和核验结果，策略仅接收观测端口。均值采用逐案例“虚拟秒/源”等权汇总；人工压力分布不代表官方随机分布，不据此推断官方排名或总体失败概率。

## 运行

从项目目录执行，必须先加载D盘环境。每次完整重跑使用新的输出子目录：

```powershell
. .\scripts\env.ps1
$Output = Join-Path $ProjectRoot ('reports\q4_validation_v1\run_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
& $PythonExe research/q4_validation_v1/run_validation.py prepare --output $Output
& $PythonExe research/q4_validation_v1/run_validation.py offline --output $Output
& $PythonExe research/q4_validation_v1/run_validation.py replay --output $Output
& $PythonExe research/q4_validation_v1/run_validation.py http --output $Output
& $PythonExe research/q4_validation_v1/run_validation.py report --output $Output
```

每一步失败时停止后续命令并保留输出，不覆盖已有批次。源码或场景哈希变动会阻止继续执行。再次使用本版本运行的是同一64个场景的复核，不是新的独立验证集；若根据结果优化算法，应另建版本和新测试集。

## 证据

`protocol.json`固定源码哈希、场景哈希、种子无重叠检查依据及分析规则。`attempts`记录每个场景的启动和终态；`records`、`traces`保存逐配置结果与完整轨迹；`http`保存自有HTTP请求、会话摘要和检查；`summary.json`保存分层配对结果。代码、结果、缓存、临时文件均位于D盘。
