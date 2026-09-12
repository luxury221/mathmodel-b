# B题：官方协议适配与本地验证

## 当前边界

本阶段只开发接口适配器并执行本地假接口测试，不启动官方模拟器，不连接官方端口，不进行官方演练或正式测试。
所有新增代码、缓存、日志和验证结果均位于 `D:\数模B题\B题`。

冻结算法继续使用第二轮的选择：问题3为 `E_joint + grid7`，问题4为 `F_route + dual21`，问题4可显式选择 `grid25` 备份。
`research\offline_validation` 下五个冻结源文件不修改；创建策略前核对 `reports\plan_trials_v2\protocol.json` 的SHA-256。
本次新增文件位于 `src\b2026_robot` 和 `research\interface_validation`，不会覆盖原有720次离线实验。

## 一键本地验证（现在可以安全运行）

```powershell
& 'D:\数模B题\B题\scripts\test-robot-interface.ps1'
```

仅运行接口单元测试、几何回归、静态检查和编译检查：

```powershell
& 'D:\数模B题\B题\scripts\test-robot-interface.ps1' -UnitOnly
```

脚本会自行启动并关闭测试用HTTP服务器，只绑定 `127.0.0.1` 的随机空闲端口，且在发送动作前验证假接口的随机能力凭据。
它不会读取或启动 `simulator` 中的任何官方程序，也不需要参赛队号、登录、联网校时或正式次数。
每次产生独立目录 `D:\数模B题\B题\reports\interface_validation_v1\run_时间_随机ID`，不覆盖旧结果。
最近一次全部成功的结果目录记录在 `D:\数模B题\B题\reports\interface_validation_v1\LATEST_SUCCESS.txt`。

输出包括：

- `unit_tests.log`：接口与故障注入单元测试。
- `frozen_component_tests.log`：原9项几何组件回归。
- `ruff.log`、`verification.json`：静态检查、编译及整个验证流程状态。
- `protocol.json`：本次配置、代码哈希、冻结算法哈希、附件哈希和依赖版本。
- `results.json`、`results.csv`：逐场景工程集成结果。
- `接口适配_本地验证报告.md`：结果和适用边界。
- 每个案例目录的 `requests.jsonl`、`summary.json`、`checks.json`：完整请求/响应、状态和一致性检查。
- `synthetic_cases_evaluator_only.json`：本地合成真值，供评估器核对，策略不读取。

## 实现与保护措施

| 层 | 文件 | 职责 |
|---|---|---|
| HTTP客户端 | `D:\数模B题\B题\src\b2026_robot\client.py` | 四个端点、参数验证、串行保护、幂等重试、响应验证、状态与计时 |
| 观测适配 | `D:\数模B题\B题\src\b2026_robot\policy.py` | 将官方字段转为冻结策略的观测，不传入真值 |
| 会话管理 | `D:\数模B题\B题\src\b2026_robot\session.py` | 安全退出、现实期限看门狗、异常停止和运行摘要 |
| 持久日志 | `D:\数模B题\B题\src\b2026_robot\storage.py` | D盘路径校验、独占创建、逐事件flush/fsync |
| 本地假接口 | `D:\数模B题\B题\research\interface_validation\mock_server.py` | 公开协议测试替身、合成物理环境、错误和断线注入 |

### 请求、响应和状态

- 仅使用 `POST /enter`、`POST /measure`、`POST /clear`、`POST /exit`，没有额外动作字段或查询参数。
- 请求JSON为无BOM的UTF-8；只发送协议规定字段。坐标、频道、标识符在联网前验证；禁止NaN、无穷及布尔数值。
- 每个新动作有唯一 `request_id`。只对连接失败、超时和截断响应等传输故障重试，并复用原路径、原ID和完全相同的请求字节。
- 默认每次网络尝试超时3秒、最多4次尝试、指数退避起点0.15秒、进入阶段最多20秒；均受本局剩余现实时间约束。
- 不会把检测的5秒虚拟时间变成现实等待。正常合法请求串行发送；本地锁拒绝并发新动作。
- 同时检查HTTP状态和严格布尔 `accepted`。拒绝响应中的 `virtual_time_s=0` 是哨兵，不覆盖当前虚拟时间、位置或频道。
- HTTP 400/404/405/409/413/415/429/500和业务拒绝不盲目重试；停止并保留证据。
- 仅接受 `direction`、`near`、`no_signal` 及 `success`、`no_target_in_range` 的正确字段组合。
- 只有已确认接受的检测改变测向机频道；清除无论成功与否均不切频道。
- 以最近一次接受响应的虚拟时钟为准，同时独立核对移动、检测、切频和清除耗时；每动作容差3微秒。
- 时间不一致时保存已确认动作和服务器时间，停止继续规划，不伪装成未执行。

### 截止与安全停止

- 使用 `/enter` 的 `remaining_real_duration_s`，不固定假定有1200秒；从首次进入尝试的本地单调时刻计时，保守扣除传输/重试延迟。
- 默认保留10秒现实时间供退出；发送新动作前检查剩余现实时间，并按该动作最坏耗时检查虚拟预算。
- 独立看门狗在普通策略计算期间也会触发停止并尝试 `/exit`；它与动作共用串行锁，不与在途动作并发发送。
- 若操作系统冻结、进程崩溃或原生代码永久持有GIL，看门狗不构成硬实时保证。
- 动作可能已执行但最终响应未知时，置 `uncertain=true`，记录待决请求并停止，不更换ID重发或继续新动作。
- 测试已经结束或连接重试耗尽时，不用 `/exit` 查询结束原因；需要查看模拟器界面。
- 本版本不提供崩溃后的自动续跑。出现 `needs_attention` 时保留日志，不直接重启客户端尝试重新进入同一局。

## 后续官方演练（需要再次确认后才执行）

**下面是预备命令，本阶段没有执行。默认入口不连接官方接口，必须显式给出确认开关。**

1. 人工启动官方模拟器并登录，确认选择的是“演练测试”，不是“正式测试”。
2. 启动演练，等待界面明确显示机器狗接口就绪。
3. 使用当前登录参赛队号作为 `RobotId`；官方默认端口为2026，若人工修改过端口则指定 `BaseUrl`。
4. 只启动一个机器人进程，执行对应问题。确认开关仅表示操作者已检查，不会从协议中识别测试模式。

问题3：

```powershell
& 'D:\数模B题\B题\scripts\run-robot-practice.ps1' `
    -Problem q3 -RobotId '替换为当前登录参赛队号' -ConfirmPracticeReady
```

问题4默认网络：

```powershell
& 'D:\数模B题\B题\scripts\run-robot-practice.ps1' `
    -Problem q4 -RobotId '替换为当前登录参赛队号' -ConfirmPracticeReady
```

问题4的25点备份：

```powershell
& 'D:\数模B题\B题\scripts\run-robot-practice.ps1' `
    -Problem q4 -RobotId '替换为当前登录参赛队号' -Q4Network grid25 -ConfirmPracticeReady
```

实际演练日志将保存至 `D:\数模B题\B题\logs\practice\时间_问题_随机ID`。
`summary.json` 的 `completed` 仅表示策略正常完成并已确认主动退出；官方全清除情况仍须以演练结束反馈核对。
本地请求日志不替代题目要求的官方加密行为日志；后者应从官方界面正常导出，保留原文件名，不解密或读取内部真值。

## 证据边界

自建假接口复用了公开规则的离线物理环境，并独立实现协议和幂等记录。
它不实现官方登录、在线校时、25分钟窗口、真实服务器限流、加密日志上传或完整HTTP安全验证。
本地测试通过意味着通信层在列明场景中没有改变冻结策略的动作，不意味着官方接口已联调成功或正式测试必然全清除。
问题4的21点网络坐标敏感性及其他数学限制仍以第二轮报告为准；本次不重新调参或增加理论保证。

首轮单元测试全部通过；静态检查发现的类型/格式问题已修正。
随后回归中的一次 `ResourceWarning` 揭示了截断响应的资源关闭问题，已改为显式关闭HTTP响应；旧日志保留用于审计。
首次全流程在25点备份构造处发现返回值适配错误：布点函数返回“站点与三角形”两项，适配层应仅取站点。
该错误同时修复于策略适配与集成基线调用处，并增加三套网络的构造回归；冻结算法源文件没有修改。
另增加连接耗尽预留时间的检查，防止连接建立后未经重新核对就发送过期动作。
完整验证以带 `verification.json` 和 `integration_complete.json` 的最新成功目录为准。
