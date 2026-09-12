# 按方案分层测试：离线入口

## 运行

```powershell
& 'D:\数模B题\B题\research\offline_validation\run_plan_trials.ps1'
```

默认依次执行组件测试、静态检查、冻结协议检查、尚未完成的实验阶段、全量确定性回放、报告与图表生成。仅想查看既有实验产物、避免再做完整回放时，可传入 `-SkipReplay`。

环境由 `D:\数模B题\B题\scripts\env.ps1` 设置，Python、缓存、临时文件和全部新增结果都在 D 盘。不启动网络服务、不调用官方模拟器、不读取官方案例。

## 文件分工

- `D:\数模B题\B题\research\offline_validation\plan_geometry.py`：保守区域更新、连续多圆覆盖、受约束主动选点、位置/方向/半径假设。
- `D:\数模B题\B题\research\offline_validation\plan_policy.py`：20 频道状态、固定网络进度、主动定位、机会式动作和清尾路线。
- `D:\数模B题\B题\research\offline_validation\plan_experiments.py`：场景隔离、开发/留出/压力阶段、选型冻结、完整性评估、确定性回放。
- `D:\数模B题\B题\research\offline_validation\test_plan_components.py`：组件与拓扑异常回归测试。
- `D:\数模B题\B题\research\offline_validation\plan_report.py`：从实际结果生成报告及诊断图，不生成实验数值。

旧版 `geometry.py`、`offline_benchmark.py` 及上一轮报告保留，作为可复现对照。

## 版本与证据

- v1 执行在新角锥多边形的浮点自交处中断。证据、协议和源码快照保留在 `D:\数模B题\B题\reports\plan_trials`。
- v2 仅对新构造的理论凸约束取顶点凸包，避免非法闭合边；已有包含失败清除信息的区域仍保留非凸形状。
- v2 使用新的留出和压力种子，保存在 `D:\数模B题\B题\reports\plan_trials_v2`。
- 核心源码变化后，冻结协议检查会拒绝复用结果。继续研发应开新实验版本，而不是修改旧哈希来强行通过。

## 安全层和效率层

- 安全层：外包可行域、已证明的发现网络、MEC 清除、多圆连续覆盖、有限格点兜底和严格频道终止条件。
- 效率层：采样 minimax、低差异位置样本、离散方向/半径假设、期望动作成本、2-opt。
- 效率层的样本耗尽或低概率不构成不存在证明；无法找到更优动作时回到安全层。
- 采样 minimax 不等于连续全局 minimax；2-opt 不等于全局最优路线。

## 当前观测接口边界

策略只接收 `measure(position, channel)` 和 `clear(position, channel)` 两个回调。当前回调由本地合成规则替身提供，并非官方 HTTP 客户端。

后续若接入真实接口，必须另外处理：只返回已接受且实际执行的动作；超时重试复用同一请求 ID；拒绝请求不能推进机器人位置或频道；实际剩余时间来自接口；完整记录指令与响应。不能把当前离线回调直接宣称为已经通过接口联调。

正式测试仍需用户明确授权。此次离线运行不能生成或替代官方加密测试日志。
