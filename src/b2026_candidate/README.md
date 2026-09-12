# 问题3候选适配（独立演练入口）

此包不修改原 `b2026_robot`、原CLI或冻结算法文件。仅支持已选定的 `q3 / E_joint+v3_combined / grid7`；问题4和原策略仍用原入口。

策略通过多继承复用原接口类的 `account`，保证权威位置、接收频道和虚拟时钟仍由原客户端管理。候选算法位于 `research/policy_optimization_v3/candidate.py`，加载前核对原离线实验的代码哈希和开发选型。独立会话模块保留原会话函数逻辑，并用AST一致性单元测试防止看门狗、退出或异常处理漂移。

`src/run_candidate_practice.py` 必须同时具备显式的 `--confirm-practice-ready` 和与当前代码哈希一致的本地验证通过记录；否则在创建机器人客户端前退出。这是本地防误操作措施，**不能识别官方当前究竟是演练还是正式**。每次仍需目视确认官方界面为“问题3 演练 测试 / 等待机器人进入”。监听端口存在不能替代该检查。

本地验证入口：`research/candidate_interface_v1/run_validation.py`，参数 `--output` 必须指向 `reports/candidate_interface_v1` 下全新的运行子目录。只使用自建随机回环端口。成功记录指针为 `reports/candidate_interface_v1/LATEST_SUCCESS.txt`。

用户明确同意演练、确认页面后使用 `scripts/run-q3-candidate-practice.ps1 -RobotId <队号> -ConfirmPracticeReady`。队号仅通过参数传入，不写死在源代码中。所有环境、缓存及产物放在D盘。

官方演练记录使用 `research/candidate_practice_v3/record.py`：先冻结五次批次，在每次进入前 `begin`，结束后 `register` 并核对公开总数；失败则保留记录并停止自动继续。**没有正式测试入口，不自动替换原策略。**
