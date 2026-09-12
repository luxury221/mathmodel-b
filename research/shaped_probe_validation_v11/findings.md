# v11独立验证

候选在开发集选择后冻结；24留出和8压力的32种子与此前864种子无交集，未用于本候选调参。
64/64配对运行成功；合并均值467.419→461.448，节省5.971秒/源（1.277%）。
留出和压力均值分别改善1.349%、1.068%；逐例18快10慢4平，最坏慢12.500%。

预登记的正确性、均值和最坏退步门槛通过。2次重放与3次本机HTTP通过，故障恢复5项，53项独立进程回归通过。
这只支持保留下一轮离线对照候选，不支持“已达到300”“所有场景更快”或“官方成绩一定提升”。
附加探索性成对案例重采样区间跨零；不按该诊断重新选型或改参数。

完整证据：selection.json、seed_audit.json、summary.json、verification.json、analysis.json、replays.json、http_results.json、records/、traces/和source_snapshot/。
验证目录：D:\数模B题\B题\reports\shaped_probe_validation_v11\validation_20260911_223412。
