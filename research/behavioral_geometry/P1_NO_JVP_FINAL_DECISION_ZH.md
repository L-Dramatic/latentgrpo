# 无 JVP 改线的最终阶段决策

**日期：** 2026-07-16  
**决策：** `RECOMMEND-KILL-AS-AAAAI-MAIN`  
**保留状态：** 所有工程与审计材料保留；不删除、不伪装为失败实验，也不继续投入数据实验/训练。  
**这不是：** 对 Latent-GRPO 本身的否定，也不是对“短 horizon 可能失效”这一科学问题的逻辑否定。

## 决策一句话

以“低算力、冲击 AAAI main 的强 method 论文”为目标，这条 latent-GRPO 无 JVP 路线不值得再投入主力实验预算：它现在只有一个工程可运行的 action family，没有已验证的自然语义 tail；第二 family 需要真实 GRPO source group；而原本承担方法贡献的 PaTR 无法在冻结 P0 合约下合法成为 certificate。

## 已经确认的正面事实

1. 真实本地 1B checkpoint、官方 sampler、真实 request stop/524 语义可以运行；
2. 四个固定 Gumbel seed 产生了会实际消费的连续首步 action；
3. 参考分支采样、候选分支共享历史 teacher forcing、独立 cache 和前向 chain-rule KL 的工程闭环可运行；
4. 全过程不依赖 JVP 或全模型 FP32 反传，显存压力合理。

这些是扎实的工程资产，但它们不是论文主结论。

## 为什么不应再作为 AAAI main 主线

| 主会所需条件 | 当前状态 | 为什么不足 |
|---|---|---|
| 有辨识度的 P1 现象 | 未测 | 合成路径没有自然退出；不能用人工 524 后的 KL 当现象证据 |
| 两个自然 action family | 未满足 | Family A 只有 Gumbel exploration；Family B 必须来自真实 8-rollout GRPO reward/advantage group |
| 合法的强 method | 未满足 | PaTR 的旧规则不能防止成对风险排序翻转，也没有原始 KL 的有效 optional-stopping 浓缩包 |
| 与既有 latent-dynamics 工作拉开距离 | 未满足 | 单纯证明早期与后期不同，仍太接近已有动态/early-exit 叙事 |
| 低算力完成度 | 条件不足 | 真实 Family B 还需 source actor/runtime 与 construction split；即便 P1 有结果，方法包仍缺失 |

冻结 P0 的 P2 是 Fisher/Loewner certificate。无 JVP 的 PaTR 改成了前向 Monte Carlo action-ranking 问题，不能在不重开新合约的情况下替代 P2。把它继续包装成“证书”会造成论文最危险的理论/主张错配。

## 严格结论

- **作为 AAAI main 强 method：建议停止。** 当前估计潜力 `4.5/10`，不值得继续消耗大量校准、数据或训练预算。
- **作为小型诊断/技术笔记：可保留。** 前提是未来另有明确目的；它不能再占用主线资源。
- **不建议此刻开启 GO-CAL。** 开启后即使得到正现象，也仍缺第二 family 与合法方法贡献，投入产出比不符合用户的目标。
- **不删除任何文件。** Family A 的 runtime 验证、P2 方法审计、Family B provenance 审计都可复用为未来类似想法的反例与工程基线。

## 之后如果仍想重新打开它，唯一可接受的触发条件

必须同时出现以下外部变化：

1. 一个新的、独立审查过的 method claim contract（不借用冻结 P0 P2）；
2. 一个能以低算力、真实 source group 构造 Family B 或机制不同第二 family 的方案；
3. 对原始或明确截断的 KL，给出合法的成对排序统计决策与强于 direct MC/MLMC 的清晰机会；
4. 先完成新一轮 collision/novelty 审查，再授权任何数据实验。

否则不应因已经投入了工程工作而继续推进。

## 已保留的关键证据

- `P1_FORWARD_ACTION_PREFLIGHT_REPORT_ZH.md`：Family A 工程通过，但不是自然 tail 或 P1 证据；
- `P1_NO_JVP_P2_METHOD_AUDIT_ZH.md`：PaTR 作为当前 certificate 被阻断；
- `P1_FAMILY_B_SOURCE_OBJECTIVE_AUDIT_ZH.md`：Family B 当前不能被无数据 toy gradient 替代；
- `P1_NO_JVP_REDESIGN_DRAFT_ZH.md`：已同步降级后的完整设计状态。
