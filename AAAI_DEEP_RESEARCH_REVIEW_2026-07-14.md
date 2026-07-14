# LatentGRPO AAAI-27 深度调研、独立评估与最终建议

调研日期：2026-07-14  
工作区：`E:\LantentGRPO`  
被评估报告：`C:\Users\30620\Downloads\LatentGRPO_AAAI_research_pivot_report_2026-07-14.md`  
结论性质：截至调研日期的独立文献检索、代码审计和研究决策，不是对未来录用结果的保证。

---

## 1. 最终结论

### 1.1 一句话判断

原报告对“当前 LatentGRPO 形态不能投 AAAI-27”和“必须先做可回放、可干预的 latent harness”的判断是正确的；但它把 **BCG-PO 评为强 Go 过于乐观**，并明显高估了 SVCCO、CSTR-PO 的独立新颖性。

补入截至 2026-07-14 的直接近邻后，我的最终排序是：

1. **首选新主线：Latent Reparameterization Audit + Functional Continuation Trust Region**  
   中文可概括为“latent thought 没有特权坐标系：面向连续隐式推理的重参数化压力测试与功能空间信赖域”。
2. **条件保留：BCG/CEBM 作为上述主线的行为度量或安全合并应用**，不能再把“未来行为等价”本身当作主要理论新意。
3. **SVCCO 只适合作为后续扩展**，除非能提出连续 latent 特有、坐标无关且支持内的 intervention semantics，并在训练中显著优于 IBPO、CVT-RL、BiPACE 等。
4. **CSTR-PO 不建议独立成文**。它当前更像 supported policy optimization、conformal RL 与 latent support modeling 的组合迁移。

### 1.2 AAAI-27 现实决策

**以当前证据和本机算力，AAAI-27 正式论文仍是 No-Go。** 官方时间表显示：摘要截止 2026-07-21，全文截止 2026-07-28，补充材料和代码截止 2026-07-31。[AAAI-27 官方时间表](https://aaai.org/conference/aaai/aaai-27/)

只有同时满足以下条件，才值得保留“抢 AAAI-27”的窗口：

- 24 小时内获得至少一张 24 GB GPU，正式训练最好有 4-8 张 A100/H100 级 GPU；
- 72 小时内在功能完全等价的坐标变换下，观察到现有 latent 操作显著、稳定、非极端的失效；
- 5 天内证明行为续写度量或功能信赖域能消除该失效，而不只是重新调阈值；
- 7 天内至少覆盖两个任务、两个随机种子，并有第二种 latent 表示作为最小跨架构证据；
- 9-10 天内出现真实任务收益或明确的稳定性提升，而非只有可视化。

任一硬门槛失败，应停止 AAAI-27 投稿，不要用弱实验包装一个概念稿。

---

## 2. 总评分与决策表

评分均为独立判断，10 分最高。“两周可行性”假设从当前代码与零正式结果出发。

| 方向 | 新颖性 | 科学重要性 | AAAI 适配度 | 两周可行性 | Idea 上限 | 最终决策 |
|---|---:|---:|---:|---:|---:|---|
| 当前仓库原始 LatentGRPO | 1.5 | 6.0 | 3.0 | 2.0 | 4.0 | **No-Go**，中心问题已直接碰撞 |
| 原报告 BCG-PO | 5.0 | 8.5 | 8.0 | 2.5 | 8.0 | **Conditional Go**，必须改写贡献 |
| 原报告 SVCCO | 4.5 | 8.5 | 7.5 | 1.5 | 8.0 | **No-Go as standalone** |
| 原报告 CSTR-PO | 3.5 | 7.0 | 7.0 | 2.5 | 6.5 | **No-Go as headline** |
| CEBM 安全分支合并 | 5.5 | 7.5 | 7.5 | 2.5 | 7.5 | 仅作强主线的应用闭环 |
| Interventional Latent Harness | 3.0 | 8.5 | 6.0 | 6.5 | 6.5 | 必须建设，但不是主会论文贡献 |
| **重参数化审计 + 功能续写信赖域** | **7.5** | **9.0** | **9.0** | **3.5** | **9.0** | **最推荐，先做 72 小时证伪** |

这里的 7.5 分新颖性是有条件的：如果最后只得到“欧氏距离不好，Fisher 距离更好”，新颖性会降到 5 分左右；只有形成“严格等价坐标变换下现有 latent RL 结论会改变”的系统性现象，并给出序列化、可训练、跨架构的解决方案，才能维持高评分。

---

## 3. 对原报告本身的评价

### 3.1 报告做对了什么

1. **正确否定了原始论文故事。** 当前项目的“连续 latent thought + 多轨迹扰动 + contrastive + GRPO-style reward”已与 Coconut、SoftCoT++、SofT-GRPO、Latent-GRPO、NF-CoT 等发生中心碰撞。
2. **正确抓住了 latent-specific 问题。** 支持、干预、合并和距离在连续 latent 中没有天然语义，这是比继续调 `contrastive_lambda` 更重要的问题。
3. **正确要求 matched-compute。** 任何 suffix rollout、probe、counterfactual 或 verifier 成本都必须计入，否则性能提升没有说服力。
4. **正确设置了停机条件。** 对高风险研究，先证伪现象再写方法是合理路线。
5. **正确建议建设 replay/intervention harness。** 当前代码还没有支持严谨研究所需的状态保存、任意步恢复和 common-random-number 对照。

### 3.2 报告最重要的缺陷

#### 缺陷 A：BCG 的概念新颖性被高估

把状态按未来分布或未来行为等价来组织，是 causal states、bisimulation、DeepMDP 和 successor measures 的经典主题。Causal State Representation 直接把历史按未来观测分布划分，并证明其与 bisimulation 的关系；Deep Bisimulation 进一步让表示距离匹配行为距离。[Causal State Representations](https://arxiv.org/abs/1906.10437)，[Deep Bisimulation](https://arxiv.org/abs/2006.10742)，[DeepMDP](https://proceedings.mlr.press/v97/gelada19a.html)

因此，“用未来 continuation distribution 定义 latent geometry”不能单独作为强新意。真正可发表的新意只能来自：

- latent reasoning 特有、跨模型可复现的失效规律；
- 比已有 bisimulation/Fisher/probe 更可靠且更便宜的估计器；
- 一个直接改善训练、合并或计算效率的算法闭环。

#### 缺陷 B：漏掉了两个非常直接的 2026 近邻

- **BiPACE** 已经把 bisimulation-guided grouping 与 action counterfactual estimation 用在 LLM policy optimization 中，还直接使用 actor hidden states 做行为分组；它报告了 1.5B/7B、三个 agent benchmark 和多种子结果。[BiPACE](https://arxiv.org/abs/2606.25556)
- **FishBack** 已经明确指出 Transformer activation space 的欧氏几何与输出行为几何严重偏离，并用输出 Fisher 的 pullback metric 推导最小失真 steering。[FishBack](https://arxiv.org/abs/2605.17231)

这两篇工作分别挤压了“BCG 用于策略优化”和“从输出行为反推 latent 几何”的空间。

#### 缺陷 C：固定 continuation policy 的 BCG 不够稳

原报告主要在一个 frozen reference continuation policy 下比较两个 latent state。如果该 policy 本身已经坍缩、能力有限或只偏好一种 mode，两个本来不同的状态可能被错误判为等价。更严重的是，策略更新后等价关系可能变化。

这不是全新的理论空白：已有工作已研究 \(\pi\)-bisimulation metric 随策略变化的偏移，并据此提出保守策略更新。[Approximate Policy Iteration with Bisimulation Metrics](https://arxiv.org/abs/2202.02881)

因此，BCG 必须处理 policy-induced aliasing 或策略漂移，不能只在一个 reference policy 下学一个静态距离。

#### 缺陷 D：SVCCO 的直接碰撞被低估

- IBPO 已通过多条反事实 reasoning path 构造隐式 process advantage。[IBPO](https://arxiv.org/abs/2605.16302)
- CVT-RL 已包含 validity gate、frozen continuation policy、selection-adjusted doubly robust estimator 和多类 controlled interventions。[CVT-RL](https://arxiv.org/abs/2606.05263)
- BiPACE 已把行为分组和 action-conditioned peer baseline 组合起来。[BiPACE](https://arxiv.org/abs/2606.25556)
- 神经网络因果干预研究已系统指出常用 intervention 会产生离自然分布的表示，并可能激活 dormant pathways。[Divergent Representations from Causal Interventions](https://arxiv.org/abs/2511.04638)
- latent-CoT 的 step-wise intervention 也已被用于分析阶段性、非局部因果路由。[Dynamics Within Latent CoT](https://arxiv.org/abs/2602.08783)

SVCCO 剩下的独立空间很窄：必须是 **连续 latent action 特有的、支持内的、重参数化不变的反事实定义**，而不能只是把 CVT-RL 的流程移到 hidden state 上。

#### 缺陷 E：CSTR-PO 的前置文献不完整

支持约束和 trust region 的组合不是新问题：

- SPOT 用密度估计显式约束行为策略支持。[SPOT](https://arxiv.org/abs/2202.06239)
- STR 在行为策略支持内做 trust-region optimization，并给出安全改进分析。[STR](https://proceedings.mlr.press/v202/mao23c.html)
- APO 在 LLM RLVR 中已把全局 KL 改写为 reference high-confidence support coverage。[APO](https://arxiv.org/abs/2602.05717)
- CCPO 已把 conformal prediction、constrained policy optimization 和 online adaptation 结合到 LLM agents。[CCPO](https://arxiv.org/abs/2511.11828)
- Conformal OPE 已明确讨论 policy shift 对 conformal validity 的挑战。[Conformal OPE](https://arxiv.org/abs/2304.02574)

所以 CSTR-PO 很容易被评价为 “SPOT/STR/APO + conformal calibration + latent actions”。而且在线 policy drift 破坏 exchangeability 后，普通 split conformal coverage 不能直接沿用。

#### 缺陷 F：三个 Top idea 实际上不是三个独立赌注

BCG、SVCCO、CSTR 都依赖同一个昂贵的核心对象：给定 prefix 和 latent state 的 continuation/support model。它们更像同一条研究树的三个终端用途，而不是三个可以并行下注的独立方向。

AAAI-27 主文只有 7 页，并明确鼓励把紧密相关思想组合成一个更强的科学贡献。[AAAI-27 Main Track CFP](https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/)

正确做法是选择一个中心命题，其他模块只作为必要机制或应用，不要同时讲 geometry、credit、conformal support 和 branch merge 四个故事。

#### 缺陷 G：七天计划和数值门槛偏乐观

AUROC 0.85、准确率 +3 个点、节省 30% 等门槛没有先验统计依据。更严谨的 gate 应使用：

- 预注册主指标；
- bootstrap confidence interval；
- 多任务、多种子方向一致性；
- 与最强 baseline 的 paired effect；
- 全部额外 rollout 和 probe 的实际 wall time；
- 对阈值和 horizon 的敏感性；
- 失败案例及 calibration drift。

---

## 4. 三个原方向的深层评估

### 4.1 BCG-PO：从“强 Go”降为“条件性 Go”

### 仍然有价值的科学问题

原始 hidden coordinate 的距离确实不等于行为距离。Latent-GRPO 自己承认其 one-sided Gumbel objective 是 surrogate latent likelihood，而不是对 embedding 的 representation-invariant density；它还明确指出 latent mixture non-closure。[Latent-GRPO](https://arxiv.org/html/2604.27998)

此外，Future Lens 已证明单个 hidden state 中可以包含多步未来 token 信息，这说明 continuation signature 在技术上是可学习的，但也意味着“从 hidden state 预测未来”本身不是新贡献。[Future Lens](https://arxiv.org/abs/2311.04897)

### 原 BCG 的四个根本问题

1. **理论祖先明确。** Causal states 和 bisimulation 已覆盖“按未来行为等价”。
2. **单策略别名。** 在一个 frozen policy 下相同，不代表对可用 continuation family 相同。
3. **状态与动作混淆。** 当前仓库中，prefix/KV cache 更接近 state，投影产生并回灌的向量更接近 latent action。两者的等价关系和 merge 条件不能混写。
4. **跨 prefix 合并不自然。** 两条分支具有不同 cache/history 时，仅平均最后一个向量不能合并完整状态；必须连同历史表示、cache transport 或充分状态一起定义。

### 能保住 BCG 的改写方式

不要声称“首次定义 behavioral geometry”。应改成：

> 我们发现 latent reasoning optimization 存在可测量的 continuation aliasing：坐标空间中接近的 latent actions 可能产生不同续写行为，而固定策略下看似等价的 actions 会在策略更新或 alternate continuation policies 下分裂。我们给出坐标无关、对策略漂移更稳的估计器，并证明它能预测或避免实际训练失败。

只有出现下面的闭环，BCG 才能达到 AAAI 强稿水平：

- 现象跨至少两种 latent 架构、两个任务稳定存在；
- 不被 cosine、whitened Mahalanobis、current-policy KL、next-token Fisher 或简单 verifier 解释；
- 行为度量在 matched compute 下改善训练稳定性或安全 merge；
- 有有限 horizon value-loss bound 或严格的 paired empirical bound。

### 上限判断

- **只有 metric probe：** 更像分析论文或 workshop，AAAI 主会边缘。
- **现象 + 理论 + 可训练算法：** AAAI strong accept 潜力。
- **跨架构揭示现有 latent RL 的系统性错误并显著改进：** 有 oral 潜力。
- **Best paper 级别：** 需要结论影响整个 continuous latent reasoning 范式，而不是只提升 GSM8K 2-3 点。

### 4.2 SVCCO：问题重要，但当前机制已高度拥挤

### 科学价值

terminal reward 对所有 latent step 同权确实是粗糙的。正确轨迹可能包含无用甚至有害步骤，错误轨迹也可能包含有价值前缀。连续 latent 又缺少自然的删除、改写和语义替换操作，因此 credit assignment 是真实问题。

### 新颖性瓶颈

SVCCO 报告中的关键组件几乎都已有直接近邻：frozen continuation、validity gating、doubly robust adjustment、反事实 path comparison、action-conditioned baseline。剩余贡献只能来自 continuous latent intervention semantics。

但这个剩余问题本身非常难：

- `zero`、Gaussian noise、线性插值通常 off-support；
- nearest neighbor 只保证坐标接近，不保证功能等价；
- conditional flow sample 可能改变多个潜在因子，无法解释为单一 decision 的替换；
- 一个 step 的 effect 依赖 intervention family 和 continuation policy，不是无条件 total causal effect；
- hidden recurrence 中早期 action 会改变后续整个 cache，positivity/overlap 很难满足。

### 算力风险

若每条轨迹有 \(T\) 个 latent steps，每个 step 需要 \(M\) 个 matched suffix rollouts，长度为 \(H\)，朴素成本约为 \(O(TMH)\) 次生成，远高于普通 outcome RL。即使只主动选择 10%-20% 的 steps，训练开销也可能翻倍或更多。

### 最终判断

不建议把 SVCCO 作为当前第一主线。只有在重参数化审计先证明“现有 intervention credit 随坐标任意改变”，并找到自然、坐标无关的 intervention family 后，它才值得重新进入候选。

### 4.3 CSTR-PO：工程价值高，论文上限最低

### 主要问题

1. 高维 conditional density 极难可靠估计，尤其每个 prefix 的有效样本很少。
2. policy density 不等于 reasoning validity；一个坏 policy 可以给坏状态高密度。
3. reference support 也可能漏掉新的有效模式，过强约束会抑制探索。
4. tangent/normal decomposition 依赖坐标和估计器；局部 PCA 的“normal”不天然具有行为含义。
5. online policy drift 下，普通 conformal coverage 不再自动成立。

### 可能保留的用途

CSTR 的 support score 可以成为 intervention gate 或安全过滤器，但不应做标题贡献。更适合在功能续写信赖域中作为便宜的第一阶段筛选，再由行为 divergence 做第二阶段判断。

### 最终判断

作为工程模块有用；作为 AAAI 主会中心 idea，新颖性和理论可信度都不够。

---

## 5. 我更推荐的新主线

### 5.1 工作标题

**Latent Thoughts Have No Privileged Coordinates: Reparameterization Stress Tests and Functional Trust Regions for Latent Policy Optimization**

可用中文表述：

> 连续 latent thought 只是内部计算的一个坐标表示。任何只依赖欧氏距离、cosine、各向同性噪声、线性平均、未经 Jacobian 修正的密度或局部 PCA 的 latent RL 规则，都可能在行为完全不变的重参数化下给出不同决策。优化约束应定义在可观察的续写行为分布上，而不是任意 hidden coordinate 上。

### 5.2 为什么它比原 BCG 更强

原 BCG 从“什么距离更合理”出发，容易被归类为 bisimulation 迁移。新主线从一个更基本、可证伪的公理出发：

> **功能完全等价的 latent coordinate charts 不应改变算法产生的行为更新。**

这能形成清晰的四段式论文故事：

1. 定义 latent policy 的 reparameterization equivariance/invariance；
2. 构造严格保持端到端行为的坐标变换压力测试；
3. 证明常见 latent 操作不满足该性质，并展示真实训练后果；
4. 提出基于 multi-horizon continuation distribution 的功能信赖域。

这比“学一个更好的距离”更容易回答 reviewer 的 `why now`、`why latent reasoning` 和 `why not ordinary bisimulation`。

### 5.3 最小数学框架

令 \(h_t\) 表示完整 prefix state，包括问题、显式 token、KV cache 和已有 latent history；令 \(z_t\in\mathcal Z\) 是 latent action；续写结果为

\[
O_{t:H}\sim P_\theta(\cdot\mid h_t,z_t).
\]

引入任意可逆光滑坐标变换 \(u=\phi(z)\)，并在进入原模型前应用 \(z=\phi^{-1}(u)\)。这样原始系统与变换后系统的端到端可观察分布完全相同。

一个 latent 操作 \(A\) 应满足：在 \(z\) 坐标中执行，和在 \(u\) 坐标中执行后映回 \(z\)，应诱导相同的可观察行为。

以下常用对象一般不满足该性质：

- \(\|z-z'\|_2\) 和 cosine；
- \(z+\epsilon,\ \epsilon\sim\mathcal N(0,\sigma^2I)\)；
- \(\alpha z_1+(1-\alpha)z_2\)；
- 不含 change-of-variables Jacobian 的 density threshold；
- coordinate PCA 得到的 tangent/normal；
- 普通 SGD 在重参数化 latent action head 上的更新轨迹。

而下面的行为散度天然坐标无关：

\[
d_F(z,z'\mid h)
=D\!\left(P(O_{t:H}\mid h,z),P(O_{t:H}\mid h,z')\right).
\]

局部近似可使用 multi-horizon continuation distribution 的 pullback Fisher：

\[
G(z)=J_z^\top F_{O}(z)J_z,
\qquad
d_F^2(z,z+\Delta z)\approx \Delta z^\top G(z)\Delta z.
\]

在坐标变换下，metric tensor 按 Jacobian 变换，因此二次型保持不变。注意，pullback Fisher 本身不是新数学；新意必须来自 latent sequential policy 的压力测试、multi-horizon 定义和训练算法。

### 5.4 建议的方法：Functional Continuation Trust Region

不要同时做 credit、support、merge 三套算法。第一版只解决一个问题：**如何约束 latent policy update 的真实行为漂移。**

建议目标：

\[
\max_{\theta'}\ \widehat J(\theta')
\quad\text{s.t.}\quad
\mathbb E_{h\sim d_{\theta}}
\left[
D_{KL}\big(P_{\theta'}(O_{t:H}\mid h),P_{\theta}(O_{t:H}\mid h)\big)
\right]\le \varepsilon.
\]

实际实现可分三层：

1. **精确小样本 oracle：** shared-RNG suffix rollouts，直接估计 answer、termination、reward 和短 suffix token 分布；
2. **amortized continuation head：** 预测多 horizon token sketch、answer class、termination 和 value distribution；
3. **局部 pullback/Fisher 近似：** 用 JVP/VJP 或低秩随机投影得到便宜的 trust-region penalty。

安全 branch merge 可以作为次要应用：不是平均 hidden vectors，而是寻找最接近多个 continuation distributions 的可达 latent action，即行为 barycenter 的投影。但它不应与主训练算法平分篇幅。

### 5.5 与最接近工作的明确边界

- **对 causal states/bisimulation：** 不声称行为等价概念新；贡献是 latent thought chart invariance、压力测试和可训练的序列功能信赖域。
- **对 FishBack：** FishBack 面向单次 activation steering 和局部输出失真；本方向面向递归 latent actions、multi-horizon continuation、policy update 和训练稳定性。
- **对 BiPACE：** BiPACE 用 actor hidden cosine 作为 policy-induced proxy；它恰好是应接受重参数化压力测试的强 baseline。
- **对 NF-CoT：** NF-CoT 的 exact density 和 change-of-variables 是重要基线，但 exact current-policy likelihood 不等于 reference behavior stability，也不自动解决 deterministic hidden recurrence。
- **对 Latent-GRPO：** 其论文明确承认 one-sided objective 是 coordinate-local surrogate，而不是 representation-invariant latent density；这是最直接的问题入口。
- **对 DPPO/TRPO：** 一般 LLM trust region 已很拥挤；必须强调连续内部 action 的 chart ambiguity 和 multi-horizon functional drift，而不是再次提出 token KL 变体。

### 5.6 必须完成的实验

### 实验 A：严格等价坐标压力测试

对同一 latent reasoner 插入 \(u=\phi(z)\)、\(z=\phi^{-1}(u)\)，先验证无操作时 logits、答案和 reward 在数值容差内一致。

变换族：

- orthogonal rotation，作为欧氏方法应通过的 sanity check；
- diagonal scaling，condition number 从 3、10、30 到 100；
- affine shear；
- 两层 invertible coupling flow，检验非线性 chart。

被测操作：

- isotropic noise；
- cosine/Euclidean nearest neighbor；
- raw interpolation/merge；
- local PCA tangent perturbation；
- Latent-GRPO-style coordinate surrogate；
- current-policy density；
- next-token Fisher；
- multi-horizon functional metric。

核心指标：

- chart-induced decision flip rate；
- mapped-back update discrepancy；
- continuation JS/KL；
- reward and termination variance across charts；
- nearest-neighbor rank correlation；
- merge failure rate；
- wall time 和额外 forward 数。

### 实验 B：非极端性检验

Reviewer 会质疑“任何算法在病态变换下都会坏”。因此必须证明：

- 中等 condition number 已产生稳定效应；
- 变换后数值范围、norm 和 precision 受控；
- whitening、adaptive threshold、Mahalanobis 等简单修复不能消除；
- 失效与 task difficulty、latent step、policy entropy 有系统关系。

### 实验 C：训练闭环

至少比较：

- 当前仓库修复后的 legacy objective；
- explicit GRPO；
- Latent-GRPO/SofT-GRPO 可复现版本；
- NF-CoT 或 SWITCH 之一；
- Euclidean/cosine trust region；
- exact policy KL；
- FishBack-style next-token pullback；
- proposed multi-horizon functional trust region。

报告 accuracy、Pass@k、reward slope、gradient SNR、invalid/overlong rate、跨 chart 方差、总生成 token、FLOPs 和 wall time。

### 实验 D：跨表示验证

最低要求是两类：

1. hidden-state recurrence，例如 Coconut/SWITCH；
2. stochastic continuous 或 vocabulary-superposition latent，例如 NF-CoT/Latent-GRPO。

只在当前自研 projection recurrence 上成立，不足以支撑一般性结论。

### 5.7 72 小时 Go/No-Go 门槛

**Go：**

- exact chart wrapper 在无操作时输出差异接近数值误差；
- 至少两种非极端 chart 使一个主流 coordinate method 的行为结果发生显著变化；
- 该变化在两个任务、两个 seeds 方向一致；
- multi-horizon functional distance 的跨 chart rank correlation 明显高于 cosine、whitening 和 next-token Fisher；
- effect 不能由 norm、precision、temperature 或阈值重调解释。

**No-Go：**

- 只有 condition number 极大时才观察到失效；
- 简单 whitening 就完全修复；
- functional metric 与 next-token Fisher 没有实质差异；
- 只改变 probe 数值，不改变 merge、update 或 reward；
- suffix rollout 成本高到抵消全部收益；
- 第二个架构上现象消失。

---

## 6. 算力与工程现实

### 6.1 当前仓库的实际成本

当前机器是 **RTX 4060 Laptop GPU，8 GB 显存**。仓库默认 teacher 为 7B 模型，`models/latentgrpo.py` 中的投影是两个 4096 到 4096 的线性层加 LayerNorm，约 **33.57M trainable parameters**。

真正的瓶颈不是投影参数，而是：

- 7B backbone 权重，FP32 约 28 GB，BF16 约 14 GB；
- 虽然 backbone 参数冻结，梯度仍需穿过 backbone 回到 latent inputs，因此必须保存激活；
- 默认 \(G=4\) trajectories、\(K=5\) recurrent thoughts；
- 每条 trajectory 还有 answer likelihood 和 generation；
- 当前加载路径没有完整的量化、FSDP、FlashAttention、activation offload 或专用 rollout engine。

所以本机 8 GB 不适合当前 7B 正式训练。量化 7B 推理可能勉强运行，但不能承担该研究所需的多轨迹、回放和反向传播。

官方 Latent-GRPO 代码对 1B 低难度和 7B 高难度配置都默认使用 8 张 GPU，并依赖定制 SGLang + verl；其 1B 配置每个 prompt 采样 8 条 rollout。[官方 Latent-GRPO 代码](https://github.com/DJC-GO-SOLO/Latent-GRPO)

BiPACE 的 7B 实验也报告在 4 张 H100 上测量训练成本，这说明严谨的 LLM policy-optimization 对比不是单卡 8 GB 级任务。[BiPACE](https://arxiv.org/html/2606.25556)

### 6.2 粗略资源预算

下面是规划区间，不是已测量报价；实际取决于 checkpoint、suffix 长度、batching 和 rollout engine。

| 阶段 | 建议模型与范围 | 最低硬件 | 粗略 GPU 预算 |
|---|---|---|---:|
| 72 小时现象 gate | 0.5B-1B，100-500 prompts，2 tasks | 1x24 GB | 20-80 A100-equivalent GPU-hours |
| Inference-only chart audit | 1B，3-4 chart families，完整 baselines | 1-2x24/48 GB | 80-250 GPU-hours |
| 1B 训练 pilot | 2 tasks，2-3 seeds | 4xA100/H100 更稳 | 300-900 GPU-hours |
| 最低主会实验 | 2 representations，3 tasks，3 seeds，多 baselines | 4-8xA100/H100 | 1,500-4,000 GPU-hours |
| 强版含 7B 与长数学任务 | 1B+7B，跨架构、完整消融 | 8xA100/H100 | 4,000-10,000 GPU-hours |

SVCCO 通常会比表中更贵，因为每个被干预 step 需要额外 suffix rollouts。CSTR 的密度模型训练较便宜，但高质量 calibration 需要大量 held-out conditional states。

### 6.3 当前代码应如何定位

- 保留为 `legacy_latentgrpo`，不要继续把它包装成正式方法；
- 继续保留已修复的 answer-label、gradient-flow 和 advantage stability tests；
- 新研究代码应从轻量 1B released latent checkpoint 开始；
- 第一阶段只实现 chart wrapper、trace/replay、continuation evaluation 和 compute accounting；
- 在现象 gate 通过前，不实现 SVCCO、conformal support 或完整 branch particle system。

---

## 7. AAAI 适配度与论文上限

AAAI-27 官方标准明确接受 theoretical、methodological、empirical、integrative 和 critical contributions，并偏好探索新问题、指出问题假设、对多个 AI 子领域有意义的工作，而不是狭窄增量。[AAAI-27 Review Criteria](https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/)

新的重参数化主线与 AAAI 的匹配点很强：

- 连接 representation geometry、reinforcement learning、causal/functional equivalence 和 LLM reasoning；
- 既可以是 critical finding，也可以有方法和理论；
- 研究问题超出某个 benchmark 的 SOTA；
- 有明确、可复现、可证伪的 stress test；
- 可影响 latent perturbation、credit、support、merge 和 policy optimization 多类方法。

但上限取决于证据层级：

| 证据层级 | 可能定位 |
|---|---|
| 单模型上 cosine 与行为距离相关性较低 | 不足以投 AAAI |
| 严格等价 chart 下多个方法排名翻转 | 有价值的 critical analysis |
| 跨两种 latent 架构稳定复现，并给出 invariant baseline | AAAI 主会可竞争 |
| 再加入理论、训练稳定性和真实效率收益 | strong accept/oral 潜力 |
| 改变领域对 latent optimization 正确对象的共识 | 才有 best-paper 级上限 |

---

## 8. 从现在开始的执行顺序

### Phase 0：今天，冻结故事

- 不再使用 “LatentGRPO” 作为新论文标题；
- 把原 BCG、SVCCO、CSTR 视为候选应用，不视为三个并行主线；
- 在实验记录中明确当前代码不是标准 stochastic latent policy；
- 确认外部 GPU 和可用 checkpoint。

### Phase 1：24 小时，最小 chart harness

- 实现固定可逆 linear/affine chart 和精确 inverse；
- 验证无操作 logits/reward identity；
- 保存 latent action、prefix/cache 标识、RNG 和 suffix；
- 实现 cosine、Euclidean、whitened、next-token KL 四个 baseline。

### Phase 2：72 小时，现象证伪

- 只跑 1B、两个任务、100-500 prompts；
- 测 nearest-neighbor、noise、interpolation 三类操作；
- 统计 chart flip、continuation divergence 和 reward variance；
- 做 condition number 和 precision 控制；
- 根据第 5.7 节硬门槛决定继续或停止。

### Phase 3：第 4-7 天，最小方法

- 构建 multi-horizon continuation sketch；
- 比较 exact suffix oracle、amortized probe、next-token Fisher；
- 先做 trust-region filtering，不做完整 RL；
- 若过滤能稳定预测 harmful update，再进入小规模训练。

### Phase 4：第 8-10 天，训练闭环

- 1B、两个任务、至少两个 seeds；
- matched compute 对比 exact policy KL、cosine/whitening 和 proposed method；
- 报 wall time、rollout tokens 和显存，不只报准确率。

### Phase 5：第 11-14 天，只在 Gate 全过时写稿

- 7 页主文只保留一个中心命题；
- 主图必须是“功能等价但算法不等价”的决定性结果；
- 第二主图展示方法恢复 chart stability 和训练收益；
- 全部关键证据放主文，不能依赖 reviewer 阅读补充材料。

---

## 9. 对 ChatGPT Pro 报告的最终比较

这份报告的优势是覆盖广、结构完整、敢于给 No-Go，并且提出了正确的基础设施优先路线。作为 brainstorming 和风险清单，它是高质量的。

但作为“最终研究选题决策”，它有三处关键不足：

1. 漏掉 BiPACE、FishBack、supported trust-region/APO 等会直接改变排序的近邻；
2. 把已有理论对象 BCG 当成了强主贡献，而没有把 latent-specific falsifiable failure law 放在第一位；
3. 对两周期限、跨架构集成和 suffix-rollout 成本估计过于乐观。

我的综合评价：

- 文献广度：8.5/10；
- 风险意识：8.5/10；
- 新颖性校准：6.0/10；
- 算力与期限现实性：5.0/10；
- 作为最终执行方案：7.0/10。

本次独立调研最重要的增量不是再增加一个 Top 4，而是把原报告中的“coordinate reparameterization robustness”从一个防守性 ablation，提升为整个研究的中心命题，并用最新近邻重新划清创新边界。

---

## 10. 最终推荐

### 研究层面

**最值得做的不是 BCG、SVCCO、CSTR 三选一，而是先回答：一个 latent optimization rule 是否会因内部坐标的任意选择而改变？**

如果答案是肯定的，而且这种变化在正常条件下会影响 reward、merge 或训练稳定性，那么就围绕重参数化审计和 Functional Continuation Trust Region 建立论文。BCG 成为其行为度量基础，CEBM 成为一个应用；SVCCO/CSTR 暂缓。

### 投稿层面

**当前不应承诺 AAAI-27。** 可以保留一个 72 小时高风险 gate，但必须把“不投稿”设为默认结果。没有外部 GPU、第二架构和训练闭环时，硬投只会得到一个概念新颖但证据不足的稿件。

### 工程层面

下一步只建设最小可证伪工具：

1. exact chart wrapper；
2. latent trace + deterministic replay；
3. matched-RNG continuation evaluator；
4. coordinate baselines 与 functional baseline；
5. compute accounting。

在现象被证实之前，不开始完整新 loss、不扩展所有 benchmark、不写论文正文。

---

## 11. 关键一手来源

### AAAI 与当前直接竞争

- [AAAI-27 Main Technical Track](https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/)
- [Latent-GRPO](https://arxiv.org/abs/2604.27998)
- [Latent-GRPO official code](https://github.com/DJC-GO-SOLO/Latent-GRPO)
- [NF-CoT](https://arxiv.org/abs/2606.06447)
- [SWITCH](https://arxiv.org/abs/2606.13106)
- [Latent Thought Flow](https://arxiv.org/abs/2606.16222)
- [BiPACE](https://arxiv.org/abs/2606.25556)

### 行为几何、表示与续写

- [Learning Causal State Representations](https://arxiv.org/abs/1906.10437)
- [DeepMDP](https://proceedings.mlr.press/v97/gelada19a.html)
- [Deep Bisimulation](https://arxiv.org/abs/2006.10742)
- [Approximate Policy Iteration with Bisimulation Metrics](https://arxiv.org/abs/2202.02881)
- [Future Lens](https://arxiv.org/abs/2311.04897)
- [FishBack](https://arxiv.org/abs/2605.17231)

### 反事实信用与因果干预

- [IBPO](https://arxiv.org/abs/2605.16302)
- [CVT-RL](https://arxiv.org/abs/2606.05263)
- [Dynamics Within Latent CoT](https://arxiv.org/abs/2602.08783)
- [Divergent Representations from Causal Interventions](https://arxiv.org/abs/2511.04638)

### 支持约束与 conformal RL

- [SPOT](https://arxiv.org/abs/2202.06239)
- [Supported Trust Region Optimization](https://proceedings.mlr.press/v202/mao23c.html)
- [Anchored Policy Optimization](https://arxiv.org/abs/2602.05717)
- [Conformal Constrained Policy Optimization](https://arxiv.org/abs/2511.11828)
- [Conformal Off-Policy Evaluation](https://arxiv.org/abs/2304.02574)
