# 文献与适用边界

Y. Wang and G. Cao, Achieving Full-View Coverage in Camera Sensor Networks, ACM Transactions on Sensor Networks 10(1), 2013, DOI 10.1145/2529974。
作者版：`https://mcn.cse.psu.edu/paper/yiwang/tosn-wang13.pdf`。
已读取定义及第5节，PDF保存在D盘research/input/full_view_tosn_wang13.pdf。

可借鉴的是同时对目标位置及朝向表达覆盖要求，而非只检查位置是否处于某个接收圆内。
本文的相机朝向、有效夹角和大区域边界近似与本题并不完全相同；格长公式不直接代入本题，不据文献公式宣布新网络完整。
本轮依然以项目已有安全接收内盘、三/四点凸包证书、完整剩余集合及实际光学补洞为准。
尚未据该文献发现可直接替代21点网络的合格布局；混合补洞是否有性能收益由本轮独立计算决定。
