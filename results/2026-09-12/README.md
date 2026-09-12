# 日期化结果数据

**官方与本地分表存放，不合并计算一个总分。**

| 文件 | 数据来源 | 说明 |
|---|---|---|
| `official_practice_cases.csv` | 真实官方PRACTICE的匿名数值导出 | Q3/Q4各5案例，10行；不是正式测试 |
| `official_practice_summary.json` | 同一官方批次的本地保存审计记录 | 含原档摘要、回放/动作核验计数；不含原始响应和标识 |
| `offline_validation_cases.csv` | 自建规则世界的历史独立验证 | Q3/Q4各32案例、两策略，共128行 |
| `offline_validation_summary.json` | 同一本地验证档案 | 24留出+8压力；这些留出已经使用，不能当新留出 |
| `current_status.json` | 本次归档状态 | 两个官方目标未达到；v48未测试 |
| `source_copy_manifest.json` | 源码及精选证据复制清单 | 文件长度与SHA-256；不含自身或新增汇总文档 |
| `publication_verification.json` | 本次交付的本地归档核验 | 1242份原档哈希、官方表数值、128条本地物理回放通过 |
| `publication_test_summary.json` | 本次交付的归档测试与入口复现 | 11项测试通过；Q3/Q4各1例历史轨迹完全一致，不是新成绩 |
| `publication_privacy_check.json` | 本次待上传文件的自动扫描 | 敏感标识、凭据模式、排除文件和体积检查；不保证识别所有秘密格式 |

CSV字段`evidence_type`分别固定为`official_practice`和`local_offline_simulation`。成本单位是虚拟秒，不是Python运行秒数。`seconds_per_source`是单个完整任务耗时除以实际源数，汇总时按案例等权。

官方案例标识已替换为无关联的顺序编号，未发布原始案例码、队伍标识或登录数据。本地合成场景保留真实的模拟源位置和种子用于评价器回放，它们**不是官方隐藏数据，也不得作为策略输入**。

数值一致性检查与本地回放命令见 `docs/复现与归档范围.md`。

本次交付回放覆盖25943个本地动作。官方匿名费用表保留原审计舍入残差，未改动成绩；本地压力案例使用历史验证器的空间相关噪声模型。全部交付检查均为本地操作，新增官方演练/正式调用为0，未测试v48草稿。
