# 无 JVP 路线的 P2 方法审计：PaTR 不能目前被称为 certificate

**日期：** 2026-07-16  
**审计性质：** 无数据、无校准、无实现的数学与合约审查  
**结论：** `BLOCK-P2-AS-CERTIFICATE`；保留 P1 前向诊断路线，不授权 PaTR 实现或任何 `GO-CAL`。

## 结论先说清楚

原先的 PaTR（Paired Tail Racing）思路有一个合理直觉：先在共享参考历史上比较多个 natural latent action 的短前缀，只对仍不确定的 action 对延长一部分 reference suffix。

但在当前定义下，它**还不是一个合法的“horizon certificate”**，也不能无缝继承冻结 P0 的 P2 方法资格。原因不是代码没写，而是有三项尚未满足的数学条件：

1. “某个 action 自己的 tail 很小”不能推出“多个 action 的长程风险排序不会翻转”；
2. 全词表连续 KL 没有现成、已验证的统一有界增量或尾部条件，不能直接套用任何时刻有效的置信序列；
3. 冻结 P0 的 P2 是基于局部 Fisher/Loewner 矩阵的 certificate gate；前向 PaTR 是不同 estimand 的序贯 Monte Carlo 决策，不能偷偷换定义后继续叫同一个 P2。

所以当前正确状态不是“PaTR 有待调参”，而是：

> PaTR 只能作为一个未来的、需要独立问题定义与重新审查的候选研究想法；它不是本项目已拥有的方法，也不是 P1 后自动执行的下一阶段。

## 1. 第一处错误：逐 action 的 tail 界不能保证排序

令有限长风险为

\[
D_H(a)=D_h(a)+T_{h,H}(a).
\]

旧草案中的规则类似于：若 `UCB(T_{h,H}(a))` 小于 `D_h(a)` 的某个比例，就说短 horizon 对 `a` 足够。

这至多是一个**单 action 的相对 tail 大小描述**，不是排序保证。若短 horizon 选择 `a` 而竞争者为 `b`，真正要控制的是成对差：

\[
\Delta_H(b,a)=D_H(b)-D_H(a)
=\underbrace{D_h(b)-D_h(a)}_{\Delta_h(b,a)}
+\underbrace{T_{h,H}(b)-T_{h,H}(a)}_{R_{h,H}(b,a)}.
\]

要在一个预先冻结的有限 action bank 中认证 `a` 是较低风险者，至少需要对每个竞争者 `b` 建立同一同时有效事件上的区间：

\[
\operatorname{LCB}\!\left(\Delta_h(b,a)+R_{h,H}(b,a)\right)>m,
\]

其中 `m` 是预先冻结的实践裕量。或者，使用一个足以排除翻转的保守条件：

\[
\operatorname{LCB}(\Delta_h(b,a))
>
\operatorname{UCB}(|R_{h,H}(b,a)|)+m.
\]

结论：后续若仍研究此问题，采样单位与停止规则必须围绕**成对长程风险差**而不是每个 action 自己的 tail；否则“保住排序”的主张没有数学含义。

## 2. 第二处障碍：KL 的 optional-stopping 界并不免费

每步量是

\[
k_t(a;Y)=\operatorname{KL}(\pi_0(\cdot\mid Y_{<t})\,\|\,\pi_a(\cdot\mid Y_{<t})).
\]

对于有限、已知的词表，它在一次具体前向中是有限的；但这不等价于在所有可达历史上拥有一个**预先已知的统一上界**。没有这种上界、明确的条件次指数假设，或独立校准出的尾部包络，就不能声称任意时刻有效的 UCB/LCB 覆盖。

有两种可能的技术选择，但它们对应不同论文：

| 选择 | 能保证什么 | 代价 |
|---|---|---|
| 预先把每步 KL 截断为 `min(k_t, B)` | 对**截断风险**可用有界增量置信序列 | estimand 已变，不能把结论写成原始 `D_H` 的 certificate；还须报告截断偏差 |
| 保留原始 KL | 可能用经校准的次指数/重尾置信序列 | 需要独立校准、可辩护的转移假设与显式常数；当前没有 |

“在测试路径上看到 KL 看起来不大”不是上界，也不能在测试数据上拟合衰减率后再认证安全。这正是冻结 P0 禁止的 test-prefix extrapolation。

## 3. 第三处障碍：自适应延长并不天然省算力

从 `h` 延长一条自回归 reference history 到 `H` 需要先支付这条历史已走过的前缀；不能像访问独立表格一样零成本抽取 tail block。

唯一可能的省算力方式是：先生成多条独立 reference prefix，只延长其中仍会缩小**成对风险差**不确定性的那部分缓存。这个方案必须把以下成本全部计入：

- 所有前缀和被延长 suffix 的模型前向；
- 每个 candidate 的 teacher forcing；
- action pair 的共享历史与 cache 内存；
- 为 optional stopping 维持置信界的额外延长；
- 与 direct MC、固定 `H8`、以及通用 continuation-MLMC 的同 wall-clock、同显存比较。

如果绝大部分 prefix 最后都不得不延长到 `H_cert`，PaTR 没有获得计算优势；若 direct MC 或 MLMC 在风险/成本前沿上相同或更好，则该方法必须 `KILL` 或降级。

## 4. 与冻结 P0 的关系

冻结 P0 的 P2 明确要求局部 Fisher/Loewner 同时界、damping 审计及有限步桥接。无 JVP PaTR 的对象是直接前向 continuation-KL 的序贯比较，不再是该矩阵对象。

因此：

- PaTR **不能**在不修订 P0 的情况下声称通过 P2；
- 也不能把“前向 KL 的 UCB”写成冻结 P0 所定义的 Fisher certificate；
- 若将来 P1 现象足够强，PaTR 必须作为一个新的、独立的 action-ranking / abstention 问题，重新写 claim contract、estimand、统计假设、竞争基线与 red-team；
- 在那之前，当前无 JVP 路线只保留为 P1 诊断可能性，而不是强 method 主线。

## 5. 若未来重启 PaTR，最低合法版本是什么

以下不是当前授权的实现计划，而是一个最小“可重新审查”规格：

1. 冻结有限 action bank、有限候选 horizon `h`、有限 `H_cert`、实践裕量 `m`，以及所有 sample seed 角色；
2. 对每个独立 reference seed，记录完整 prefix，并在延长时仅继续该 seed 自己的 cache；
3. 对每个 action 对在同一条参考历史上计算成对 block 差 `R_{h,H}(b,a)`；
4. 明确选择“截断 risk”或经过独立校准的原始 risk 尾部模型，绝不混用；
5. 对 action 对和自适应样本数提供同时有效 coverage，未达到区分条件时只能 `continue` 或 `abstain`；
6. 以 seed-disjoint 评估验证：选择的 action、false-safe 决策和排序与 `H_cert` 直接 Monte Carlo 一致；
7. 在匹配前向数、时间、显存下同时胜过 direct MC 和 continuation-MLMC；
8. 不把有限 `H_cert` 的统计决策说成无限未来、全局动力学或答案 early-exit certificate。

即使全部成立，这仍是新方法的起点，而不是保证能达到 AAAI 主会强 method 标准的结论。

## 6. 对当前项目的影响

| 项目组件 | 审计后状态 |
|---|---|
| Family A 前向 action 兼容性 | `PASS-ENGINEERING`，但还没有自然退出/语义 tail 证据 |
| Family B | 尚未有真实 source-objective action；不能用合成 advantage 伪造 |
| P1 Forward Horizon Gap | 仍可作为一个待审查的诊断问题 |
| PaTR 作为当前 P2/certificate | `BLOCKED` |
| 无 JVP 路线的强 method 叙事 | 当前不成立 |
| `GO-CAL` / 数据实验 / 训练 | 仍然禁止 |

## 下一步的正确优先级

先完成 Family B 的**真实性可行性审计**和 Family A 的**自然退出/语义 tail eligibility**审计；如果这两项不能形成至少两个真正独立的自然 action family，或 P1 只能复现已有 latent-dynamics 现象，则停止这一主线。

在没有独立新合约、数学审查与严格 baseline 计划前，不再为 PaTR 写代码或投入实验预算。
