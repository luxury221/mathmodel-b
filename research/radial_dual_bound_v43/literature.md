# v43外部定理核对记录

## 实际使用的原始证明

- 原作：Henri Joris，Le chasseur perdu dans la forêt (Un problème de géométrie plane)，Elemente der Mathematik 35(1)，1980，1–14。
- 英译登记：Steven Finch，A translation of Henri Joris' "Le chasseur perdu dans la forêt" (1980)，arXiv:1910.00615，2019年10月1日。
- 已读取第1节式(1)：最短候选长度为7*pi/6+1+sqrt(3)。
- 第2节定理1的条件：从单位圆圆心出发，路径与全部圆周切线相交。
- 第8节明确说明：连通集与紧凸域的每条支撑线相交，当且仅当其凸包包含该紧凸域。这与v43所需条件相匹配。
- 本地读取副本：reports/radial_dual_bound_v43/joris_1910_00615_source.html。
- SHA-256：1200df9947cf2c124cfc0a3deb3ba2072f2f8e0765891b3036ed84353e78819a。

未使用随机首次到达意义下的平均Disk-Inspection常数；不存在频道必须排除整个合法区域，不能用其替换完整支撑线巡查下界。

## 误写题名的独立核对

已实际读取STACS 2026卷364第44篇网页的title与h1，两者均为Optimal Average Disk-Inspection via Fermat’s Principle。它不是主协议初稿误写的题名。见CORRECTION.md；本轮移动下界只依赖上述Joris原始证明。

## 数值实现依据

mpmath 1.3.0的iv上下文提供区间数的端点与向外包围算术。本轮只使用实际支持的pi、sqrt、atan2；最终预算使用Python Fraction复核。它不是Lean/Coq等形式化证明，也不能把高精度抽查本身当作全连续域证明。
