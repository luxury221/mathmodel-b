# Q3定位门槛记录勘误

运行时实例化冻结ScanEconomyPolicy(mode='station_only')后，实际options包含target_radius=2000。
其来源是research/joint_search_v4/policy.py中MODES['adaptive']；RadialPolicy和ScanEconomyPolicy未覆盖此值。
run_tour()及v23的eligible_targets()使用options.get('target_radius', 130)，这里130只是键缺失时的默认值。

因此，旧文档和交接摘要中的“原策略实际130米门槛”“v23保留130米门槛”不准确。
v23保留的是实际2000米配置，identity逐动作复现证明其与原策略相同，不能反过来证明130米门槛已被测试。
旧运行分数、轨迹及完整性证据不变，原预登记文件和快照不回写；本文件明确补正其方法描述。

本次重新直接测试更低门槛是否通过真实巡查中的机会测向减少主动定位绕行，而不是重复一个已经实施的130米实验。
冻结59份依赖和正式客户端不改；对每个新实例复制options，绝不原地修改共享MODES字典。
