# P1 无 JVP 重设计草案：用前向 continuation-KL 审计 latent action 的 horizon 是否足够

**状态：** 设计草案，尚未授权校准、数据集实验、训练或论文方法实现。  
**目的：** 在不依赖 recursive latent JVP、FP32 全模型反传或局部 Fisher 的前提下，保留“短 horizon 是否会误判长期 continuation 风险”的核心问题。  
**当前结论：** 这是当前 8GB 显卡下唯一尚可完成**最后一轮资格审计**的 latent-GRPO 分支；P2 方法已被阻断，不能再把它当作可直接推进的 AAAI 主线，也不能自动继承旧 P1 草案的任何通过资格。

---

## 1. 一句话版本

我们不再问：

> 一个 latent 向量的局部梯度在后面几步会放大多少？

而改问：

> 对一个由模型原始 latent sampler 自己产生的第一步 action，若只看前 h 步可见 continuation，是否会把“哪个 action 更安全”判断错；并且能否只在必要时继续采样后续 tail，给出风险排序不再翻转的统计证据？

这直接使用模型实际生成分布和前向 logits，不需要对 recursive latent 过程求导。

---

## 2. 为什么必须改，而不是继续修 JVP

已经完成的真实 checkpoint 预检说明：

1. 官方 latent request、524、EOS、additional-stop、cache 与 source sampler 的基础执行语义已经核实；
2. 固定 top-k support 后，条件递归路径在参考点能精确重现 source sampler；
3. 但 BF16 下的有限差分在多个预先冻结步长上不稳定；而 8GB GPU 不能安全加载 FP32 的完整 1B 模型和递归反传。

因此，继续把 JVP 当作风险分数或论文主方法，会把研究结论建立在不可靠的数值测量上。这个风险不是调一个 epsilon 可以解决的。

本草案的原则是：

- 不把任何局部二次型、Fisher、JVP、Jacobian 谱半径或梯度几何叫作主结果；
- 不把“固定 support 条件导数”误说成真实有限 action 的行为变化；
- 主 estimand 直接是有限 horizon 的 continuation distribution 差异。

---

## 3. 新的研究对象：前向、共享历史、有限 horizon 的 action risk

对一个 context c，令 z0 是参考第一 latent action，za 是一个自然产生的候选第一 action。令 P0 是参考 action 后的可见 continuation 分布，Pa 是候选 action 后的可见 continuation 分布。

从参考分布采样一条可见历史 Y。对每个未来位置 t，在**同一条参考历史**上 teacher-force 两个分支，计算：

    k_t(a; Y) = KL( pi0_t(. | Y_<t) || pia_t(. | Y_<t) ).

定义前向、有限 horizon 的真实风险：

    D_H(a) = E_{Y ~ P0}[ sum_{t=1}^H k_t(a; Y) ].

它有四个关键优点：

1. 是真正的有限序列 continuation KL，不是单条独立 candidate rollout 的伪比较；
2. 不需要对 latent recurrence 求导；
3. 每个 k_t 非负，因此 tail 风险可以被直接分块审计；
4. 对同一条参考历史复用多个 action，可以做低方差的配对风险排序。

这里的 action risk 是“候选第一 latent action 让后续可见生成分布偏离参考 action 的程度”，不是答案正确率、奖励或人类偏好。

---

## 4. P1 要验证的现象：Forward Horizon Gap

选一个便宜 short horizon h，例如 h=8；选一个事先冻结的长 oracle horizon H，例如 H=32 或 H=64。定义：

    Tail_h,H(a) = D_H(a) - D_h(a).

主要现象不是“长一点会有更多 KL”这个显然事实，而是以下任一更强事件：

1. **风险排序翻转：** action a 在 D_h 下比 b 更安全，但在 D_H 下更危险；
2. **假安全：** D_h(a) 小于预先冻结的安全阈值，但 Tail_h,H(a) 足以使 D_H(a) 超过阈值；
3. **短观察没有识别力：** 仅用 h 内的前向特征无法可靠地区分大 tail 与小 tail，而 tail probing 可以；
4. **机制特异性：** 这种现象在早期 recursive latent interface 明显存在，但在末端 latent、普通 visible embedding、或不递归控制上显著减弱。

若只观察到 D_H 大于 D_h、但没有排序翻转、假安全或机制特异性，则不能称为强 Horizon Gap，只能降级为描述性诊断。

---

## 5. 自然 action 的保留与修改

### 5.1 保留 Family A：source-native noisy action

在共同的第一 latent interface，直接使用原始训练 sampler 的 Gumbel/噪声 action 产生候选 mixture。它是当前最干净、最不需要额外反传的自然 action 家族。

每个 action 必须保留：

- 初始 logits、top-p 可行集、Gumbel 值、top-k id 与 mixture 权重；
- request stop 结果、proxy、524 退出、后续 latent trace；
- RNG state、action norm、support churn、nearest-reference 距离；
- 是否实际被 source execution 消费，而不是只“提议”却已经 stop 的伪 action。

### 5.2 暂不删除 Family B，但移除其 suffix JVP 依赖

旧 Family B 的 source-objective gradient 只可作为“生成初始自然 action 的方式”，不能再作为递归 JVP 或 Fisher 风险估计的输入。

**2026-07-16 审计状态：** 当前 `BLOCKED-NO-DATA-AND-NO-SOURCE-ACTOR-RUNTIME`。Family B 需要真实同题八条 source rollout、reward/GRPO advantage、保存的 Gumbel record 与已验证 actor runtime；在无数据阶段以合成 advantage 得到的梯度只是 toy control，不能计作第二个 natural family。详见 `P1_FAMILY_B_SOURCE_OBJECTIVE_AUDIT_ZH.md`。

它是否保留，取决于后续单独的 source-objective 第一 action preflight：

- 若 source actor surrogate、PPO self-ratio、advantage 与初始 action 的反传能在当前算力下精确复现，B 可以作为第二自然 action 家族；
- 若不能，则不允许把 A 的不同随机 seed 冒充两个独立 action family；
- 此时必须设计真正机制不同、但仍由源算法自然产生的第二 action 家族，或将论文上限降为单家族诊断。

这是一条硬规则：不为了保住论文叙事而虚构“两个 family”。

---

## 6. PaTR 的状态：被阻断的研究想法，不是当前 P2 方法

独立数学审计后，原先的 PaTR 规则被**降级为未授权的候选研究想法**。它不能在当前项目中被称为 certificate，也不能继承冻结 P0 的 P2 方法资格。详见 `P1_NO_JVP_P2_METHOD_AUDIT_ZH.md`。

阻断原因有三条：

1. 单个 action 的 `Tail_h,H(a)` 很小，不能推出 action 对 `(a,b)` 的长程风险排序不会翻转；
2. 原始全词表 KL 缺少预先成立的统一有界/尾部条件，不能直接声称 optional-stopping UCB/LCB；若截断 KL，estimand 已变，不能再称原始 `D_H` 的 certificate；
3. 冻结 P0 的 P2 是 Fisher/Loewner certificate，而 PaTR 是前向 Monte Carlo 的成对排序决策。二者不是同一个方法对象。

若未来独立重启 PaTR，最小正确研究对象必须是预先冻结 action bank 上的**成对差**：

    Delta_H(b,a) = D_H(b) - D_H(a)
                   = [D_h(b)-D_h(a)] + [Tail_h,H(b)-Tail_h,H(a)].

任何“短 horizon 足够”的判断，至少要在同一同时有效事件上对每个竞争者证明：

    LCB( Delta_H(b,a) ) > practical_margin,

或用一个明确、可验证的 tail-difference 上界排除翻转。无法满足时只能继续采样或 abstain。

这仍需要一个独立选择并审查的浓缩性包：要么对事先截断的风险做有界置信序列并报告截断偏差，要么在独立校准下验证原始 KL 的尾部模型。采样成本还必须计入每个被延长 prefix、candidate teacher forcing、缓存与 optional-stopping 额外调用，并在 matched compute 下胜过 direct MC 与 continuation-MLMC。

**当前约束：** 不实现 PaTR；不把它写为论文方法；不把有限 `H_cert` 的统计决策说成无限未来或行为 certificate；除非用户明确启动一个独立的新 claim contract、数学审查和竞争基线计划。

---

## 7. 与近期工作如何区分，以及哪些路线禁止走

| 最近工作 | 它已经做了什么 | 本项目必须避开 | 我们只有在何种条件下仍有空间 |
|---|---|---|---|
| Dynamics Within Latent Chain-of-Thought | 对 latent step 做因果干预，发现早期输出偏向与后期表征承诺可能不同 | 不能只重复“早期读出不代表后期行为” | 必须是 source-native action、共享历史 distributional KL、可量化排序翻转或假安全 |
| EAT | 用 </think> 后的熵轨迹做 reasoning early exit | 不能用熵阈值、答案稳定或 Pass@1 停止做主方法 | 估计的是 action-induced continuation risk，不是何时输出答案 |
| Conformal Thinking | 用校准风险控制 reasoning token budget | 不能把 conformal wrapper 重新命名为新 certificate | PaTR 需直接估计 action tail KL，并与 direct MC/MLMC 做 matched-compute 比较 |
| Continuation MLMC | 按不同 level 的方差与成本分配样本 | 不能宣称首个 multilevel 或首个 adaptive horizon estimator | 必须证明共享历史的 action-ranking 决策和显著的实际计算优势 |
| Certified World Models / recurrent stability | 预测 horizon、Lyapunov 或稳定性分析 | 不能用收缩率、谱半径或“系统稳定”充当 tail guarantee | 只讨论实际自回归分布偏移的经验或统计证据，不宣称动力系统全局证书 |

关键文献：

- [Dynamics Within Latent Chain-of-Thought](https://arxiv.org/abs/2602.08783)
- [EAT: Entropy After </Think>](https://openreview.net/forum?id=hfEVqiJyF6)
- [Conformal Thinking](https://openreview.net/forum?id=noDJPmA3ha)
- [Conformal Risk Control](https://arxiv.org/abs/2208.02814)
- [Continuation Multilevel Monte Carlo](https://arxiv.org/abs/1402.2463)

---

## 8. 这条路线的真实评分

| 维度 | 当前评分 | 原因 |
|---|---:|---|
| 低算力可行性 | 7.5 / 10 | 只需前向 logits 与 teacher forcing；但长 suffix rollout 仍有成本 |
| 工程可靠性 | 7 / 10 | 524/request 语义已通过；source action B 仍需独立预检 |
| P1 现象新颖性 | 5.5 / 10 | 与 latent dynamics 工作距离很近，必须靠严格 estimand 和 natural action 拉开 |
| P2 方法新颖性 | 不评分 / 当前阻断 | PaTR 不能作为冻结 P2 的 certificate；若重启，须是独立新项目并重做合约与数学审查 |
| AAAI main 潜力 | 4.5 / 10（当前） | 目前只剩待证实的 P1 诊断，尚无合法方法包；不能再以“P1 强后自然有 PaTR”推高评分 |
| 继续价值 | 5.5 / 10 | Family A 工程已通，但自然退出、语义 tail、Family B 和跨机制复现都仍是硬门；适合再做有限审查，不适合投入大规模实验 |

这不是“稳投 AAAI”的方向。它只是一个尚未被 Family-A 工程问题直接击穿、但方法空间和第二自然 family 都仍未成立的高风险诊断分支；继续只能用于迅速做资格判断，而非大规模投入。

---

## 9. 接下来严格执行的顺序

1. Family A 的 source-action / cache / sample-force 工程预检已通过；但合成路径没有自然退出，不能越过自然语义 tail 门；
2. 做 Family B 是否能由**真实 source objective**生成首步 action 的真实性审计；不能用合成 advantage 或任意 loss 伪造第二 family；
3. 在不读取校准/测试题目的前提下，完成 Family A 的自然退出、可见语义 tail、动作 support 与低算力预算的 eligibility 审计；
4. PaTR 已完成数学审查并被阻断为当前 P2；除非另建 claim contract，不能把它作为下一阶段；
5. 只有两个自然 action family 与 P1 的有限行为 KL 设计都通过独立审查，才讨论是否值得请求新的校准授权；
6. 若自然 action 不能稳定构造、P1 只能复述“早期输出与后期承诺不同”、Family B 不真实、或自然 tail 在低算力下无法资格化，则 kill 这条主线。

**当前禁止：** 开数据集、开校准、跑 held-out、训练、把 PaTR 写成论文方法、或声称任何 certificate 已成立。
