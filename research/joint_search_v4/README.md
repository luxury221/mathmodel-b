# 问题三、四联合优化 v4（仅本地）

本目录是独立研究版本，不替换原机器人或正式入口。所有环境、源码副本、缓存、日志和结果留在D盘。
目标仍为问题三均值低于180、问题四均值低于300，单位是案例等权的总虚拟秒/源。
不能把本地合成成绩当作官方成绩，不能通过删除失败或调整源数权重声称达标。

## 代码

- `coverage.py`：连续朝向区间的保守位置域、全向覆盖和已发现源的方向联合约束。
- `policy.py`：主动定位、目标与站点共同排路、条件外环优先、动态覆盖等探索候选。
- `search_planner.py`：仅用于规划选点的离散见证集，不提供搜索结束证书。
- `belief_policy.py`：在旧策略上附加方向位置约束和有上限的原地多频道光学尝试。
- `experiment.py`：开发实验；每次必须写入新目录，保存源码、全部轨迹和失败。
- `final_validation.py`：冻结候选后的新留出/压力集、独立计时审计、重放及纯本地HTTP故障恢复。
- `test_joint_search.py`：方向边界、全向兼容、几何退化、停止条件与原地清除回归。

## 本轮冻结

问题三选择 `adaptive`，问题四选择 `sweep_previous`。冻结证据目录：
`D:\数模B题\B题\reports\joint_search_v4\validation_20260911_184912`。
该目录创建之后不要再修改本目录Python文件并沿用本次验证。下一轮另建研究版本和全新验证种子。

## 复现

在项目根目录先执行 `. .\scripts\env.ps1`。
单元回归可运行 `& $PythonExe -m pytest research/joint_search_v4/test_joint_search.py -q -p no:cacheprovider`。
重复相同留出种子只算重放，不能算新增独立样本。`final_validation.py`是首次冻结新验证的入口，当前已有相同种子证据，再执行会检测重复并停止；
如需复现已冻结结果，使用 `experiment.evaluate` 逐例重放保存的场景，并与轨迹哈希比较，不要改种子冒充原结果。

`final_validation.py`没有接受官方地址的参数。HTTP验证只在 `MockArena` 创建的随机回环端口运行。
旧机器人、Q3演练候选和Q4 v2冻结版本保持不变；下一轮是否部署不由本地研究结果自动决定。

原HTTP验证的累计精度检查应结合 `http_clock_audit.json` 阅读；审计程序位于独立目录
`D:\数模B题\B题\research\joint_search_v4_audit\audit_http_clock.py`，未修改本轮已冻结Python源码。
