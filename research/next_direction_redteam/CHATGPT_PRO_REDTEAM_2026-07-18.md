# LatentGRPO 下一研究方向：强红队选题审计

**审计日期：** 2026-07-18  
**固定仓库：** [`L-Dramatic/latentgrpo`](https://github.com/L-Dramatic/latentgrpo)  
**固定提交：** [`6f32fc37527f923e064927c8930c2c9a3d9f64a2`](https://github.com/L-Dramatic/latentgrpo/tree/6f32fc37527f923e064927c8930c2c9a3d9f64a2)  
**目标标准：** AAAI Main Track / Best Paper 级方法选题，而非“可以训练的工程变体”  
**最终结论：** **`NO-GO：当前没有足够强的新候选`**

---

## 1. Executive Summary

### 1.1 唯一结论

> **NO-GO：截至 2026 年 7 月 18 日，没有候选同时跨过“独立 estimand、非组合式机制、强简单基线、部署可辨识性、足够 oracle ceiling、跨策略复现”六道门。**

这不是“完全没有可研究问题”，而是没有一个问题已经强到值得立即投入训练算力并作为 AAAI Main Track 方法主线。充分检索后的三个残余候选是：

1. **PTPU：Policy-Transported Prefix Utility，目标策略可迁移前缀效用；**
2. **LRPE：Logged Reasoning Policy Evaluation，日志化推理控制策略评估；**
3. **APVC：Anytime Post-Selection Verifier Contract，自适应选择后的 verifier 可靠性合同。**

三者都可以构造清晰的低成本证伪实验，但都没有达到 GO：

- PTPU 的问题重要，但核心可被拆成 **PUM 的 prefix gain + GenAC/V0 的 policy conditioning + 标准 OPE/DR/MAGIC**；
- LRPE 会被拆成 **标准离线策略评估 + LLM 推理控制器日志化**，而在线重放通常是更简单、更可信的基线；
- APVC 会被拆成 **MSV + inference-time reward hacking + conformal/anytime calibration**，直接邻居过强。

因此，本报告只授权一个**两周、零训练、低预算的 sacrificial falsification probe**，用于彻底判定 PTPU 是否存在异常大的现象与 oracle 上限；它不是“隐含 GO”。失败即归档，成功也只意味着重新做一次文献与方法独立性审计。

### 1.2 为什么不是继续修 latent likelihood

固定提交中的证据已经把该方向的主要支点逐个拆除：

- **OMPI-R** 在相同 empirical marginal candidate logits 上与 TPO 的 loss 和 gradient 恒等，而且未包含 policy-dependent latent proposal 的 score-function 项；
- **PCMC** 在 Latent-GRPO-Llama-1B 上有强 non-closure，却在 SofT-GRPO-Qwen-1.5B 上因 action 近乎 one-hot 而彻底失败；
- **LPCA** 发现 exact score 与 released surrogate 的 reward-conditioned gradient 确实不同，但 exact candidate 的 held-out local utility 更差；
- **MMLPO、Top-K Concrete、score-squashed Gumbel、coupled group exploration、coordinate invariance/FCTR、Forward Horizon Gap** 均已按冻结规则停止；
- 这些结果共同否定了“只要把 latent action 定义得更精确、几何更漂亮、闭包更一致，性能就会自然提高”的默认先验。

### 1.3 证据标签

全文严格区分：

- **[仓库事实]**：固定提交中的代码、冻结协议、机器 JSON、正式决策；
- **[文献事实]**：论文原文、正式 proceedings、OpenReview 或作者官方代码；
- **[推断]**：从仓库与文献组合得到、尚未由本项目实验直接验证的判断；
- **[研究假设]**：本报告提出、必须先过 gate 才能继续的假设。

---

## 2. 仓库真实状态与已关闭方向

### 2.1 冻结机器产物，而非 README 叙事

| 方向 | 固定提交中的真实结果 | 不可再主张的内容 | 可复用资产 |
|---|---|---|---|
| OMPI-R | exact gate 证明 OMPI loss/gradient 与“经验边缘化后做 TPO”恒等；latent irrelevance null 还能让旧 Gate 1A 假阳性 | 新 optimizer、独立 latent-credit 方法、靠 responsibilities/ESS 证明多路径因果性 | all-pairs likelihood、responsibility、MI/ESS 诊断代码 |
| PCMC | 1000 个冻结事件；Latent Q75 JS `0.0638586935`，SofT Q75 JS `5.37182094e-8`；overall `KILL_PCMC_GATE_A0` | 通用 closure 方法、A1/B/训练、删掉 SofT 后的正结果故事 | source-native sampler replay、MATH-500 split、embedding-mixture evaluator |
| LPCA / exact latent density | 48 states、144 records；exact/surrogate gradient cosine `0.7948`、relative error `0.7071`，但 exact gain `0.006327` < surrogate `0.007956` | “exact likelihood 必然更好”、matched replacement training | policy-contract matrix、ratio/score controls、local utility evaluator |
| MMLPO | mixed-measure 理论可写，但 practical optimizer 与 novelty gate 失败 | 混合测度本身作为 AAAI 方法贡献 | measure-theoretic audit 与边界案例 |
| Top-K Concrete / FTK-PO | density 与分解成立；冻结 effect gates 失败；support component 承担中位数 `99.84%` joint KL | 两预算 trust region、selection correction 必然重要 | exact Top-K Concrete density 与 normalization tests |
| SSG-PO | 数学合同通过；squash 在 diffuse logits 下使 mixture 更尖，Gate 2 失败 | 用 smooth squash 替代 clipping 作为方法 | transformed-density tests |
| SGGE | antithetic/coupled rollout 增加 reward-varying groups，但 group-relative baseline 与 current score 相关，产生偏差 | “更多 mixed-reward group = 更好无偏 GRPO signal” | 必做的 cross-trajectory score–baseline audit |
| Coordinate invariance / FCTR | chart contracts 通过；held-out phenomenon threshold 失败；SWITCH C2 的 V32 margin 全为负 | 当前 FCTR/V32 estimator、再调 chart/半径 | chart/replay/JVP/Fisher instrumentation |
| Behavioral geometry / Horizon Gap | 40 actions、256 paths；无 robust H8/H64 flips，late-mass `0`，median `D8/D64=0.999839` | 当前 forward Horizon Gap、PaTR/ACRT、追加 GPU | endpoint-aware same-history continuation-KL evaluator |
| PCMC arithmetic closure | Latent 有非线性 gap，SofT action 有效支持度约 1 | 把 ordinary distillation 重命名为 closure | source-native action extraction 与直接 branch oracle |

**核心机器证据：**

- [PCMC A0 frozen decision JSON](https://github.com/L-Dramatic/latentgrpo/blob/6f32fc37527f923e064927c8930c2c9a3d9f64a2/artifacts/pcmc_gate/local_a0/a0_decision.json)
- [PCMC post-hoc diagnostics JSON](https://github.com/L-Dramatic/latentgrpo/blob/6f32fc37527f923e064927c8930c2c9a3d9f64a2/artifacts/pcmc_gate/local_a0/posthoc_a0_diagnostics.json)
- [LPCA Stage-B gradient JSON](https://github.com/L-Dramatic/latentgrpo/blob/6f32fc37527f923e064927c8930c2c9a3d9f64a2/artifacts/policy_contract_audit/stage_b_gradient_v1.json)
- [Forward Horizon Gap frozen summary](https://github.com/L-Dramatic/latentgrpo/blob/6f32fc37527f923e064927c8930c2c9a3d9f64a2/research/behavioral_geometry/results/p1_sacrificial_discovery_v1/summary.json)

### 2.2 PCMC 的跨实现否定比单模型负结果更强

[仓库事实]

- Latent-GRPO-Llama-1B：500/500 actions 有 10 个 content components；最大权重中位数 `0.364038`，有效支持度中位数 `4.42617`，top-token disagreement `42.4%`；
- SofT-GRPO-Qwen-1.5B：最大权重中位数 `0.999998927`，有效支持度中位数 `1.00000215`，top-token disagreement `0`；
- 两个 checkpoint 都通过 source-native sampler replay，不能把差异归因于通用 adapter bug；
- PCMC 的 A0 门要求两个自然 checkpoint 都通过；SofT 的 Q75 JS 比 `0.005` 门槛低约五个数量级。

[推断]

“soft/latent action”不是一个足够同质的方法类。不同实现的温度、support、proxy、noise 与执行图会改变问题是否存在。任何下一方向若只在 Latent-GRPO 的 diffuse mixture 上成立，原则上已触发“只能在一种 sampler 上成立”的硬 KILL 条件。

### 2.3 不能从 LPCA 的 semantic mismatch 推出方法价值

[仓库事实]

LPCA 的 exact filtered-Concrete score 与 released surrogate 的 reward-conditioned direction 有实质差异，但在 frozen RMS step 上：

\[
\Delta U_{\mathrm{exact}}=0.0063270,\qquad
\Delta U_{\mathrm{surrogate}}=0.0079559,
\]

\[
\Delta U_{\mathrm{exact}}-\Delta U_{\mathrm{surrogate}}
=-0.0016289.
\]

因此“定义不一致”是审计结果，不是“替换成 exact 就会更好”的方法证据。

### 2.4 可复用资产

下一方向真正能复用的不是旧 claim，而是：

1. 固定 MATH-500 数据版本、hash split 与 prompt contract；
2. Latent-GRPO / SofT-GRPO source-native sampler replay；
3. action fingerprint、weighted embedding、proxy-token 与 structural-end audit；
4. endpoint-aware continuation rollout 与 same-history teacher forcing；
5. exact/empirical likelihood ratio、ESS、support、normalization controls；
6. append-only journal、pre-registration、fail-closed decision 机制；
7. 单 RTX 4060 级 checkpoint smoke 与小模型推理基础设施。

---

## 3. 2024—2026 文献地图

本节按**会直接杀死候选的技术邻域**组织。正式会议与 arXiv 预印本分开看待；预印本仅作为碰撞证据，不等同于已同行评审结论。

### 3.1 Latent / continuous / soft reasoning

1. [Coconut: Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769) (2024)
2. [Soft Reasoning: Navigating Solution Spaces in Large Language Models](https://proceedings.mlr.press/v267/) (ICML 2025)
3. [Latent-GRPO](https://arxiv.org/abs/2604.27998) (2026)
4. [SofT-GRPO](https://arxiv.org/abs/2511.06411) (2025)
5. [LEPO](https://arxiv.org/abs/2604.17892) (2026)
6. [NF-CoT](https://arxiv.org/abs/2606.06447) (2026)
7. [SWITCH](https://arxiv.org/abs/2606.13106) (2026)
8. [Latent Thought Flow](https://arxiv.org/abs/2606.16222) (2026)
9. [Chain of Superposition / Latent-SFT](https://arxiv.org/abs/2510.15522) (2025)
10. [Soft Thinking](https://arxiv.org/abs/2505.15778) 与 [Mixture of Inputs](https://arxiv.org/abs/2505.14827) (2025)

**碰撞结论：** “在 latent space 做 RL、soft token、continuous thought、exact likelihood、flow likelihood、hidden recurrence”已经不是空白。剩余空间必须由新的、部署可观测的 estimand 支撑，不能只换 action parameterization。

### 3.2 RLVR 与 policy optimization

1. [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300) (2024)
2. [RLOO / Back to Basics](https://arxiv.org/abs/2402.14740) (2024)
3. [DAPO](https://arxiv.org/abs/2503.14476) (2025)
4. [Dr.GRPO](https://arxiv.org/abs/2503.20783) (2025)
5. [Target Policy Optimization](https://arxiv.org/abs/2604.06159) (2026)
6. [MinPRO](https://arxiv.org/abs/2601.22718) (2026)
7. [Step-GRPO](https://ojs.aaai.org/index.php/AAAI/article/view/40441) (AAAI-26)
8. [Self-Rewriting RL](https://ojs.aaai.org/index.php/AAAI/article/view/40738) (AAAI-26)
9. [PURE](https://proceedings.neurips.cc/paper_files/paper/2025/hash/be91eb86eb74efc055cff83e953f86ce-Abstract-Conference.html) (NeurIPS 2025)
10. [Fast and Effective On-Policy Distillation from Reasoning Prefixes](https://aclanthology.org/2026.findings-acl.1276/) (ACL Findings 2026)

**碰撞结论：** 改 advantage、ratio、length normalization、process reward aggregation、prefix weighting、加 verifier，均属于高度拥挤区。没有独立 estimand 的 objective 变体不应进入候选。

### 3.3 Prefix value、process supervision、credit assignment

1. [Math-Shepherd](https://arxiv.org/abs/2312.08935) (2024)
2. [ReasonFlux-PRM](https://proceedings.neurips.cc/paper_files/paper/2025/hash/26618fb384d3873b8ef6ab292a69095b-Abstract-Conference.html) (NeurIPS 2025)
3. [From Correctness to Utility / PUM](https://arxiv.org/abs/2606.07190) (2026)
4. [Bringing Value Models Back / GenAC](https://arxiv.org/abs/2604.10701) (2026)
5. [`V_0`: A Generalist Value Model for Any Policy at State Zero](https://arxiv.org/abs/2602.03584) (2026)
6. [InT](https://arxiv.org/abs/2601.14209) (2026)
7. [IBPO](https://arxiv.org/abs/2605.16302) (2026)
8. [CVT-RL](https://arxiv.org/abs/2606.05263) (2026)
9. [BiPACE](https://arxiv.org/abs/2606.25556) (2026)
10. [VPPO / Save the Good Prefix](https://aclanthology.org/2026.findings-acl.1767/) (ACL Findings 2026)

**碰撞结论：** PUM 已把 prefix value 从局部正确性改写为 outcome-grounded gain；GenAC 已把 critic 与 current actor 条件化；`V_0` 已显式输入 capability context；InT、IBPO、CVT-RL、BiPACE、VPPO 已覆盖 intervention、counterfactual、bisimulation、first-error 与 prefix reward。新候选必须精确说明自己估计的是哪个尚未覆盖的量。

### 3.4 Repair、branching、backtracking 与 test-time compute

1. [Scaling LLM Test-Time Compute Optimally](https://openreview.net/forum?id=4FWAwZtd2n) (ICLR 2025)
2. [Off-Trajectory Reasoning](https://openreview.net/forum?id=hVUIguIm14) (ICLR 2026 Poster)
3. [Multi-Sequence Verifiers / MSV](https://arxiv.org/abs/2603.03417) (2026)
4. [ST-BoN](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ed45d6a03de84cc650cae0655f699356-Abstract-Conference.html) (NeurIPS 2025)
5. [Majority of the Bests](https://proceedings.neurips.cc/paper_files/paper/2025/hash/36556567e8437f137da23047309155dd-Abstract-Conference.html) (NeurIPS 2025)
6. [ThinkBooster](https://arxiv.org/abs/2606.06915) (2026)
7. [Test-time Prompt Intervention](https://ojs.aaai.org/index.php/AAAI/article/view/40718) (AAAI-26)
8. [OpenDeepThink](https://arxiv.org/abs/2605.15177) (2026)
9. [Look Before You Leap](https://aclanthology.org/2026.eacl-long.367/) (EACL 2026)
10. [Certified Self-Consistency](https://arxiv.org/abs/2510.17472) (2025/2026)

**碰撞结论：** adaptive stop、branch prune、candidate rerank、repair、pairwise judge、confidence controller 都已有直接邻居。仅把 verifier 接到 controller 上，不构成方法新意。

### 3.5 Verifier calibration、post-selection、reward hacking

1. [Inference-Time Reward Hacking in LLMs](https://proceedings.neurips.cc/paper_files/paper/2025/hash/590a0cc0306c1c63e2d66a51a407718f-Abstract-Conference.html) (NeurIPS 2025)
2. [MSV](https://arxiv.org/abs/2603.03417) (2026)
3. [ORCA: Online Reasoning Calibration](https://arxiv.org/abs/2604.01170) (2026)
4. [Certified Self-Consistency](https://arxiv.org/abs/2510.17472)
5. [Majority of the Bests](https://proceedings.neurips.cc/paper_files/paper/2025/hash/36556567e8437f137da23047309155dd-Abstract-Conference.html)
6. [ReasonFlux-PRM](https://proceedings.neurips.cc/paper_files/paper/2025/hash/26618fb384d3873b8ef6ab292a69095b-Abstract-Conference.html)
7. [PURE](https://proceedings.neurips.cc/paper_files/paper/2025/hash/be91eb86eb74efc055cff83e953f86ce-Abstract-Conference.html)
8. [Calibrated Reasoning](https://arxiv.org/abs/2509.19681)
9. [ThinkBooster](https://arxiv.org/abs/2606.06915)
10. Verification-horizon / evaluator-shift work（2026 文献簇）

**碰撞结论：** pointwise calibration 在 adaptive max/stop 后失效是合理问题，但候选集联合建模、hedging、conformal risk、anytime stop 已分别出现，剩余空间很窄。

### 3.6 Off-policy evaluation、policy transport、可辨识性

1. [Doubly Robust OPE](https://proceedings.mlr.press/v48/jiang16.html) (ICML 2016)
2. [MAGIC / Data-Efficient OPE](https://proceedings.mlr.press/v48/thomasa16.html) (ICML 2016)
3. [MWL/MQL](https://proceedings.mlr.press/v119/uehara20a.html) (ICML 2020)
4. [Double Reinforcement Learning](https://proceedings.mlr.press/v119/kallus20b.html) (ICML 2020)
5. [Minimax-Optimal OPE](https://proceedings.mlr.press/v119/duan20b.html) (ICML 2020)
6. [DR with Shrinkage](https://proceedings.mlr.press/v119/su20a.html) (ICML 2020)
7. [Bootstrapping FQE](https://proceedings.mlr.press/v139/hao21b.html) (ICML 2021)
8. [MinPRO](https://arxiv.org/abs/2601.22718) (2026)
9. [ADWM for OPE of LLM Agents](https://arxiv.org/abs/2606.05558) (2026)
10. [`V_0`](https://arxiv.org/abs/2602.03584) (2026)

**碰撞结论：** 一旦候选使用 importance ratio、DR、FQE、density-ratio learning、world model 或 confidence interval，它必须把这些当成强直接基线，不能当作新贡献。LLM 序列的长 horizon、token support 与 moving target 只会让可辨识性更困难，不会自动创造算法新意。

### 3.7 文献地图总判断

2024—2026 的可组合积木已经非常齐全：outcome/process reward、generative critic、first-error、repair intervention、counterfactual continuation、prefix gain、actor/capability conditioning、Best-of-N/tree search、adaptive stop、pointwise/setwise verifier、conformal risk、reward-hacking hedge、IS/DR/FQE/world-model OPE、latent/soft action 与 exact density。

因此，下一篇 AAAI 方法论文不能再是“从每列各拿一个积木”。它必须提出一个**已有方法无法替代的目标量或识别问题**，并用 oracle 证明这个量值得估计。

---

## 4. 26 个候选及碰撞淘汰表

说明：

- “最近邻”列每个候选列出至少 5 个直接碰撞；完整链接见第 3 节与第 13 节。
- `KILL-COLLISION`：目标与机制已基本覆盖；
- `KILL-BASELINE`：最强简单/组合基线覆盖主要收益；
- `KILL-IDENT`：部署不可观测或不可辨识；
- `KILL-REPO`：固定仓库已有冻结负结果；
- `SHORTLIST`：仅保留用于深入设计，不代表 GO。

| ID | 候选 | 独立核心 | 最近邻碰撞（至少 5） | 最危险基线 / oracle | 判定 |
|---:|---|---|---|---|---|
| 1 | **PTPU：目标策略可迁移前缀效用** | 估计指定 continuation policy 下的 prefix utility，而非“通用步骤正确性” | PUM、GenAC、`V_0`、MinPRO、DR、MAGIC、MWL/MQL、ADWM | PUM + actor embedding + DR；fresh target-policy K=2/4 rollouts | **SHORTLIST-1，但触发组合碰撞风险** |
| 2 | **LRPE：日志化推理控制策略评估** | 不在线执行新 branch/stop/verify controller，直接从完整日志估计 accuracy–cost | ADWM、DR、MAGIC、FQE、ORCA、MSV、ThinkBooster | exact replay / online rerun；oracle 是真实在线 controller | **SHORTLIST-2** |
| 3 | **APVC：自适应选择后的 verifier 合同** | 对 adaptive max/stop 选中的输出控制 error，而非 pointwise ECE | MSV、Inference-Time Reward Hacking、ORCA、Certified SC、MoB、PURE | per-N calibration + MSV + HedgeTune | **SHORTLIST-3** |
| 4 | policy-adaptive PUM | critic 输入当前 actor/capability | PUM、GenAC、`V_0`、TAMPO、Step-GRPO | PUM + policy ID embedding | KILL-COLLISION |
| 5 | recoverability frontier | 测“错误前缀后能否恢复”并训练 | Off-Trajectory Reasoning、Math-Shepherd、PUM、VPPO、InT | corruption curriculum + SFT | KILL-COLLISION |
| 6 | counterfactual repair leverage | 用替换/删除一步后的 success delta 做 credit | InT、IBPO、CVT-RL、BiPACE、VPPO、PUM | frozen-policy paired continuation | KILL-COLLISION |
| 7 | first-error localized RL | 只惩罚首错后缀 | VPPO、InT、Math-Shepherd、PURE、Step-GRPO | first-error PRM + standard PPO/GRPO | KILL-COLLISION |
| 8 | censoring-corrected PRM | 修正只对可完成 prefix 取样的 selection bias | PUM、ReasonFlux、Math-Shepherd、DR、FQE | inverse-propensity weighting + PRM | KILL：标准 missing-data/OPE，缺独立机制 |
| 9 | generic adaptive compute controller | 按 uncertainty 决定继续、分支或停止 | ORCA、Certified SC、MSV、ST-BoN、ThinkBooster、Test-time PI | calibrated confidence threshold | KILL-COLLISION |
| 10 | set-conditioned verifier | verifier 联合看候选集 | MSV、OpenDeepThink、MoB、ReasonFlux、pairwise verifier | MSV | KILL-EXACT-COLLISION |
| 11 | diversity-aware branch coverage | 奖励候选集的策略/语义多样性 | MSV、MoB、OpenDeepThink、ST-BoN、ThinkBooster、set RL | DPP/MMR + BoN | KILL：“diversity + verifier” |
| 12 | reward-hacking-aware budget tuning | 随 N 调整 proxy reward 权重 | HedgeTune、MoB、PURE、ORCA、MSV | HedgeTune | KILL-COLLISION |
| 13 | generative PRM / reasoning critic | critic 先 CoT 再估值 | GenAC、ReasonFlux、GenPRM、Think-RM、`V_0` | GenAC | KILL-COLLISION |
| 14 | latent-state process reward | 从 hidden/latent state 预测 step value | LSRL、Coconut、Latent-GRPO、GenAC、Look Before You Leap | visible-prefix PRM + hidden probe | KILL：增量且泄漏风险高 |
| 15 | latent counterfactual credit | 干预 latent action 再看 outcome | repo SVCCO/FCTR、InT、IBPO、CVT-RL、BiPACE | visible-step intervention | KILL-REPO + COLLISION |
| 16 | policy abstraction / bisimulation | 依据 behavioral equivalence 聚合 reasoning states | BiPACE、Deep Bisimulation、Causal States、repo BCG、FCTR | actor hidden cosine clustering | KILL-COLLISION + IDENT |
| 17 | coordinate-invariant latent update | functional metric 替代 Euclidean update | natural gradient、Fisher geometry、BiPACE、repo FCTR、BCG | whitening / Fisher / next-token KL | KILL-REPO |
| 18 | exact mixed-measure latent likelihood | 给 atom+continuous action 定义 exact policy ratio | MMLPO、LPCA、Top-K Concrete、NF-CoT、TPO | released surrogate + local utility oracle | KILL-REPO |
| 19 | coupled/antithetic group exploration | 增加 GRPO 组内 reward variation | arithmetic sampling、RQMC PG、antithetic PG、repo SGGE、RLOO | raw-reward unbiased estimator | KILL-REPO：group baseline bias |
| 20 | policy-conditional mixture closure | 一次 soft mixture forward 模拟 branch mixture | Soft Reasoning、MoI、mixup、PCMC、distillation | randomized hard branch / ordinary KL distill | KILL-REPO，禁止复活 |
| 21 | conformal support trust region | 仅在 calibrated support 内更新 | SPOT、Supported TR、APO、CCPO、conformal OPE | ordinary KL + support mask | KILL：X+Y，仓库已归档 |
| 22 | agent critical-step verifier | 找出 agent trajectory 的关键/错误步骤 | AgentV-RL、CoVerRL、CVT-RL、BiPACE、VPPO、ReasonFlux | outcome verifier + first-error PRM | KILL-COLLISION |
| 23 | evaluator leakage / metamorphic benchmark | 系统测格式、长度、温度、judge leakage | PRM benchmarks、reward hacking、verification horizon、judge stress tests、ReasonFlux | stratified leakage audit | KILL as method；可作 benchmark/analysis |
| 24 | premature-confidence objective | 奖励早确定、短推理 | Certified SC、ORCA、ST-BoN、Step-GRPO、self-rewriting、PI | length penalty + calibrated stop | KILL-COLLISION |
| 25 | effective-support soft/hard router | action ESS 高时 soft，低时 hard | PCMC、SofT-GRPO、Latent-GRPO、mixture routing、entropy controller | max-weight/entropy threshold | KILL：PCMC 事后挽救 |
| 26 | posterior-aware latent-proposal gradient | 补齐 OMPI 缺失的 proposal score term | IWAE、VIMCO、DReG、RWS、SCG、TPO | standard latent-variable gradient estimator | KILL：OMPI 事后挽救 + 经典碰撞 |

### 4.1 淘汰统计

- `SHORTLIST`：3；
- 明确文献碰撞：14；
- 固定仓库负结果：7（与碰撞可重叠）；
- 组合式 “X + Y”：至少 9；
- 可辨识/部署观测问题：至少 8；
- 只有 benchmark/analysis 上限、缺方法主张：1；
- 违反用户明确禁止的旧方向复活：3。

短名单不是“前三名都值得做”，而是只有这三者还值得写出严格 gate。

---

## 5. Top 3 全面评分表

### 5.1 评分

| 候选 | 原始创新性 /25 | 重要性 /15 | 因果可验证与可辨识 /15 | 方法深度 /15 | 实验与 oracle /10 | AAAI 适配 /10 | 算力 /5 | 抵抗 Reviewer 2 /5 | 总分 /100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **PTPU** | 14 | 14 | 11 | 10 | 9 | 8 | 4 | 2 | **72** |
| **LRPE** | 12 | 13 | 11 | 9 | 9 | 7 | 4 | 2 | **67** |
| **APVC** | 9 | 13 | 10 | 8 | 8 | 7 | 4 | 1 | **60** |

### 5.2 判定规则

- `GO`：总分至少 82，且没有硬淘汰项；
- `HOLD`：72–81，且核心贡献不是直接组合；
- `NO-GO`：低于 72，或任一硬淘汰项成立。

PTPU 恰好 72，但不能判 HOLD，因为它尚未证明自己不是：

\[
\text{PUM prefix gain}
+\text{actor/capability conditioning}
+\text{standard OPE}.
\]

这命中“贡献只能描述为 X + Y”的硬淘汰条件。故整个 portfolio 的结论仍是 NO-GO。

---

## 6. Top 3 逐个深度设计

# 6.1 Top 1：PTPU — Policy-Transported Prefix Utility

### 1. 一句话核心主张

**一个 reasoning prefix 的“好坏”不是内禀属性，而是相对于指定 continuation policy 的成功概率；若跨政策排序翻转普遍存在，过程监督必须估计可运输的 `V^π(h)`，而不是训练一个脱离目标 policy 的通用标量。**

### 2. 具体科研 gap

PUM 已定义 prefix gain，GenAC 让 critic 通过 in-context conditioning 跟踪 current actor，`V_0` 在 state zero 用 capability context 估计 any-policy performance。剩余的窄 gap 是：

> 给定由多个 behavior continuation policies 产生的 prefix–continuation 日志，能否在**明确 overlap 条件**下，估计一个尚未大量 rollout 的目标 policy `π*` 对同一 prefix 的 utility，并在支持不足时给出 abstention/certificate？

这不是“prefix value 是否依赖 policy”——文献已经提示依赖；它必须是一个**target-policy transport estimand + identifiable estimator + support contract**。

### 3. 为什么现有工作没有完全解决

- **PUM**：主问题是 outcome-grounded prefix gain，不以 target-policy OPE 和 overlap 为核心；
- **GenAC**：actor-conditioned critic 可用当前 actor 数据同步训练，不是从多 logger 日志识别新 target policy；
- **`V_0`**：any-policy context 只在 state zero；
- **MinPRO**：prefix ratio 用于 off-policy optimization，不是 prefix-value evaluation；
- **ADWM**：针对交互 agent environment 的 model-based OPE；
- **DR/MAGIC/MWL/MQL/FQE**：提供成熟 estimator 家族，反而是必须击败的直接机制基线。

### 4. 方法机制和必要公式

定义题目 `x`、前缀 `h`、continuation policy `π`、终局 verifiable reward `R∈[0,1]`：

\[
V^\pi(h)
=
\mathbb E_{c\sim \pi(\cdot\mid h)}
[R(h\oplus c)].
\]

相对于裸题或父前缀的 policy-indexed gain：

\[
G^\pi(h)=V^\pi(h)-V^\pi(x),
\qquad
A^\pi(h_t)=V^\pi(h_t)-V^\pi(h_{t-1}).
\]

多 logger `μ_k` 下，基础 per-decision DR 为：

\[
\widehat V_{\mathrm{DR}}^{\pi^*}(h)
=
\widehat V(h)
+
\frac1n\sum_i\sum_t
\bar\rho_{i,1:t}
[r_{i,t}+\widehat V(s_{i,t+1})-\widehat Q(s_{i,t},a_{i,t})],
\]

\[
\rho_{i,1:t}
=
\prod_{j=1}^{t}
\frac{\pi^*(a_{i,j}\mid s_{i,j})}
{\mu_{k_i}(a_{i,j}\mid s_{i,j})}.
\]

reasoning 只有 terminal reward 时：

\[
r_{i,t}=0\;(t<T),\qquad r_{i,T}=R_i.
\]

必须比较 trajectory IS、per-decision IS、WIS/clipped IS、DR、MAGIC、FQE、marginalized density ratio、PUM+policy embedding、GenAC-style critic、fresh target-policy K=2/K=4 rollout。

本报告不把 DR 或 importance ratio 当新贡献。候选可主张的唯一方法增量是一个 prefix-specific multi-logger support contract：

\[
\mathcal S(h,\pi^*)=
(\mathrm{ESS},\max_i\bar\rho_i,D_\alpha(d_{\pi^*}\Vert d_\mu),\mathrm{coverage}),
\]

支持不满足时输出 `ABSTAIN`，而不是伪精确价值。

### 5. 与最近邻逐项区别

| 最近邻 | 已有内容 | PTPU 必须额外证明 |
|---|---|---|
| PUM | prefix gain、student-conditioned solve-rate、utility ranking | 对指定未充分 rollout 的 target policy 做可识别 transport，而非 pooled utility |
| GenAC | generative critic、current actor conditioning | multi-logger OPE、显式 overlap、无需同步 target rollout 的估计 |
| `V_0` | capability context、any-policy state-zero value | 非初始 prefix、sequence support、policy-specific ranking inversion |
| MinPRO | prefix ratio 用于 off-policy optimization | value evaluation 与 uncertainty，不是更新 surrogate |
| ADWM | world-model OPE for agents | 无环境数学 completion、token-level likelihood、prefix ranking |
| DR/MAGIC/MWL/MQL | 通用 OPE | LLM prefix-specific identification 与不可被通用 OPE 替代的结果 |

### 6. Reviewer 最可能拒稿理由

1. “这是 PUM 加 policy ID，再套 DR。”
2. “序列 IS 在语言模型上因 horizon 与 support mismatch 不可用。”
3. “目标 policy 直接 rollout 两次就更可靠。”
4. “跨 policy 排序翻转只是能力差异。”
5. “prefix 是外部注入，不是 target policy 自然可达状态。”
6. “目标 actor 在 RL 中持续变化，value 标签立即过时。”
7. “只在数学 binary verifier 上成立。”
8. “abstention 后覆盖率太低。”
9. “贡献是 diagnostic，不是方法。”
10. “theorem 只是 overlap 与 policy dependence 的直接推论。”

### 7. 最强简单基线

**PUM + capability/policy embedding + fresh K=2 target rollouts。**

它比单纯 PUM 更危险：policy embedding 可吸收大部分 policy identity；K=2 target rollouts 是无偏、易实现、无需 importance ratio 的直接测量。在相同 token 预算下，复杂 OPE 若不能显著击败它，就没有存在理由。

### 8. Oracle ceiling

对每个 `(h,π)` 生成 32–64 条独立 target-policy continuation：

\[
\widetilde V^\pi_{\mathrm{oracle}}(h)
=
\frac1M\sum_{m=1}^{M}R_m.
\]

用 oracle 做 prefix selection，计算：

\[
\mathrm{Regret}
=
\widetilde V^\pi_{\mathrm{oracle}}(h^*_{\mathrm{oracle}})
-
\widetilde V^\pi_{\mathrm{oracle}}(h^*_{\mathrm{method}}).
\]

**立即 KILL：** oracle target-specific selection 相对最强 pooled baseline 提升 `<2` 个绝对百分点；normalized regret reduction `<10%`；或 K=2 fresh rollouts 已达到 oracle 可获收益的 `>=80%`。

### 9. CPU Gate 0

构造 finite reasoning DAG：10,000 states、3 个已知 continuation policies、精确 transition/terminal reward，包含 rank-inversion、no-overlap、near-deterministic、long-horizon 四类情形，可精确枚举 `V^π(h)`。

比较 DM、IS、WIS、DR、MAGIC、FQE、MWL/MQL 与候选 estimator。

**PASS：**

- supported states 上 absolute bias `<=0.02`；
- 95% interval coverage `>=90%`；
- rank inversion 检测 F1 `>=0.85`；
- unsupported states 的错误高置信输出率 `<=5%`；
- 在至少两个非病态 regime 相对最强 generic baseline 的 RMSE 降低 `>=15%`。

**KILL：** 只能复现标准 DR/MAGIC；support contract 只是 ESS threshold；或任何 bias/coverage 门失败。

### 10. 单 GPU Gate 1

#### 数据

冻结 60 题：MATH-500 20、AMC/AIME 风格公开题 20、GSM8K 20；按题 hash 预先分 calibration 30 / confirmation 30。每题固定 4 个自然 prefix：来自不同 policy 的正确、可恢复错误、不可恢复错误、ambiguous prefix。

#### Policy

likelihood-ratio 主 gate 必须同 tokenizer：

- `Qwen/Qwen2.5-Math-1.5B`；
- `Qwen/Qwen2.5-Math-1.5B-Instruct` 或同基座 SFT snapshot；
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`；
- 仓库 SofT-GRPO-Qwen checkpoint 可作为第四 logger/target，前提是 tokenizer contract 完全一致。

仓库 Latent-GRPO-Llama-1B 只做**直接 rollout 的跨架构现象复现**，不能与 Qwen 做 token-level IS。

#### 采样

- 每个 prefix、每个 logger：16 continuations；
- 每个 target oracle：32 continuations；
- 最大 visible tokens 固定；
- 同一 policy 内 matched temperature/top-p；
- 记录每 token log probability、EOS/stop、格式、长度、entropy 与 verifier trace。

总量约 34,560 trajectories，预计 15–20M tokens。

#### PASS：必须全部满足

1. confirmation 上至少两个 policy pair 的 rank-inversion rate `>=15%`，95% CI lower bound `>10%`；
2. `Q75 |V^{πa}(h)-V^{πb}(h)| >=0.10`；
3. oracle target-specific selection 相对最强 pooled/capability baseline，在 N=8 candidate pool 提升 `>=3pp`，或 normalized regret 降低 `>=20%`；
4. 等生成 token 预算下，candidate 比 K=2 fresh target rollouts 的 RMSE/selection regret 至少低 `20%`；
5. 同时优于 generic DR、MAGIC、FQE 与 PUM+policy embedding；
6. median ESS `>=10`，supported/non-abstaining prefixes `>=80%`；
7. 跨两个 decoding regime 与两个 model snapshots 成立；
8. 控制长度、温度、熵、格式与 verifier 后主效应仍成立。

#### KILL：任一即停

- inversion `<10%`；
- oracle ceiling `<2pp`；
- K=2 fresh rollout 等价或更好；
- median ESS `<5` 或 abstention `>50%`；
- 仅一个 model/sampler 成立；
- 需要跨 tokenizer ratio；
- 主要效应可由长度、温度、格式、entropy 或 evaluator leakage 解释；
- estimator 与标准 DR/MAGIC 代数同构且没有新的识别结果。

### 11. 推荐数据集与检查点

主 gate：MATH-500、GSM8K、AMC23/AIME-style 公开集合；Qwen2.5-Math-1.5B base/instruct、DeepSeek-R1-Distill-Qwen-1.5B、仓库 SofT-GRPO-Qwen-1.5B。现象 replication 使用仓库 Latent-GRPO-Llama-1B，但不做 cross-tokenizer OPE。

### 12. 消融和负面对照

- policy ID 打乱；capability profile 打乱；
- same-prefix/same-policy split-half；
- same policy 不同 seed 的 null inversion；
- temperature-only、top-p-only、length-matched policies；
- format-normalized prefixes；
- correct final answer but invalid reasoning adversarial set；
- exact token logprob vs proxy；
- no-overlap synthetic positive control；
- K=1/2/4 fresh rollout curve；
- pooled PUM、per-policy PUM、GenAC-style critic、DR、MAGIC、FQE、MWL/MQL。

### 13. 算力、显存、时间、存储

- CPU Gate 0：16 CPU cores、`<2 h`、RAM `<16 GB`；
- 单 GPU Gate 1：建议 24 GB；1.5B BF16 + KV cache 峰值约 12–20 GB；
- 推理约 24–40 GPU-hours；本地 RTX 4060 Laptop 可能 60–100 wall-clock hours；
- 存储 60–100 GB；
- 训练：**0 GPU-hours**；
- 云成本计划上限：按 `$1.5–$3/GPU-hour` 假设，`<$150`；不是实时市场报价；
- H20/A100-80G：不授权。

### 14. 成功后的论文类型

若所有 gate 通过且 estimator 明显击败标准 OPE 与 fresh-rollout baseline，可形成：

**“Whose Value Is It? Policy-Transported Process Supervision for LLM Reasoning”**

贡献可能包括：policy-indexed prefix estimand、cross-policy ranking reversal、multi-logger transport + abstention、compute-matched fresh-rollout benchmark、moving-policy staleness 边界。若只发现 rank inversion 而 estimator 无独立优势，最多是 analysis/benchmark，不是 AAAI 方法主线。

### 15. AAAI 适配度和 idea 上限

- 优点：问题清晰，连接 RL、reasoning、process supervision、可靠评估；
- 缺点：容易被视为 OPE 应用；理论可能经典；只有 math 时上限有限；
- 当前 AAAI 适配：`8/10`；
- idea ceiling：现象与 estimator 都强时约 `8/10`，否则 `5/10`。

### 16. 当前决策

**不授权训练。只授权 sacrificial CPU Gate 0；Gate 0 通过后才可执行单 GPU Gate 1。即使 Gate 1 通过，也必须重新审计“PUM + GenAC + OPE”组合碰撞。**

---

# 6.2 Top 2：LRPE — Logged Reasoning Policy Evaluation

### 1. 一句话核心主张

**在一次性生成“最大、可重放的推理搜索 envelope”后，从日志离线评估 branch/verify/stop controller 的 accuracy–cost，而不是为每个 controller 重新在线采样。**

### 2. 具体科研 gap

大量 test-time compute 论文比较不同 controller，但每个 controller 各自采样，昂贵且噪声大。若先生成 branch-complete、nested、包含 verifier trace 的 envelope，某些 controller 可 exact replay，其他 controller 可 OPE。

### 3. 为什么现有工作没有完全解决

- ThinkBooster 是统一框架/benchmark，不等于有统计保证的离线 controller evaluation；
- ADWM 是 agent world-model OPE，环境型且模型化；
- ORCA/MSV 是具体 controller/calibration；
- 经典 OPE 没定义 reasoning search envelope、nested reuse 与 token compute contract。

但该 gap 很容易被审稿人视为“standard OPE applied to test-time search”。

### 4. 方法机制和必要公式

controller `κ` 在历史 `H_t` 上决定：

\[
a_t\in\{\text{continue},\text{branch},\text{verify},\text{stop}\}.
\]

目标：

\[
J(\kappa)=\mathbb E_{\tau\sim\kappa}[R(\tau)-\lambda C(\tau)].
\]

若 envelope 包含 controller 所需全部 nested branches，则 exact replay；否则用：

\[
\widehat J_{\mathrm{DR}}(\kappa)
=
\widehat J_{\mathrm{DM}}(\kappa)
+
\sum_t\rho_{1:t}
[r_t+\widehat V(H_{t+1})-\widehat Q(H_t,a_t)].
\]

贡献若成立，必须是**可重放日志合同 + exact/partial-identification 边界**，不是再发明 DR。

### 5. 与最近邻逐项区别

- ADWM：learned world model；LRPE 优先 exact/nested replay；
- ThinkBooster：系统与 benchmark；LRPE 要给统计识别和 CI；
- ORCA/MSV：特定策略；LRPE 评估策略族；
- Certified SC：特定 majority stopping certificate；LRPE 评估通用 controller。

### 6. Reviewer 最可能拒稿理由

1. 直接在线 rerun 更可信；
2. branch-complete envelope 的成本接近穷举；
3. controller 改变采样分布，日志 overlap 不足；
4. 贡献是 benchmark infrastructure，不是学习方法；
5. ADWM/standard OPE 已覆盖。

### 7. 最强简单基线

exact replay for nested controllers、online rerun、DM、IPS、WIS、DR、MAGIC、FQE、ADWM-style simulator。

### 8. Oracle ceiling

对冻结 controller 集做真实在线执行得到 `J_online`。若 controller 间最大 accuracy spread `<2pp`，或 cost frontier 差异 `<10%`，立即 KILL。

### 9. CPU Gate 0

随机 stochastic search tree，已知所有 controller value。

**PASS：** policy ranking Spearman `>=0.9`；95% CI coverage `>=90%`；exact replay 分支误差为 0；OPE estimator 比 generic DR/MAGIC 有结构性优势。

**KILL：** generic baseline 等价或 overlap 过低。

### 10. 单 GPU Gate 1

- 100 MATH problems；
- 每题生成最大 N=32、带 prefix snapshots、candidate logprobs、verifier traces；
- 固定 8 个 controller：N=4/8/16/32、confidence stop、MSV-like stop、verifier threshold、cost-aware policy；
- 对每个 controller 另做小规模 online oracle。

**PASS：** ranking Spearman `>=0.8`；RMSE `<= controller value spread 的 25%`；CI coverage `>=90%`；离线选出的 controller 在线提升 `>=2pp`，或同准确率省 `>=10%` compute；显著优于 exact replay + generic DR/MAGIC/ADWM。

### 11. 推荐数据与 checkpoint

MATH-500、DeepSeek-R1-Distill-Qwen-1.5B、Qwen2.5-Math-1.5B-Instruct、一个固定 outcome verifier；复用仓库 continuation logger。

### 12. 消融和负对照

nested vs non-nested、missing branch、logger temperature shift、verifier shift、cost definition、exact replay positive control、unsupported controller abstention、online seeds。

### 13. 算力、显存、时间、存储

CPU `<2 h`；单 GPU 20–40 h；24 GB；存储 50–100 GB；无训练。

### 14. 成功后的论文类型

更像“evaluation methodology + benchmark”。要成为 AAAI Main Track 强文，必须证明它改变 controller 排名并节省大量在线评估，而非只是复用日志。

### 15. AAAI 适配度和 idea 上限

AAAI 适配 `7/10`；idea ceiling `7/10`；Reviewer 2 抵抗力弱。

### 16. 当前决策

**不优先执行。只有 PTPU CPU Gate 0 被 KILL 且已有 branch-complete logs 时，才值得做 CPU toy；不授权 GPU。**

---

# 6.3 Top 3：APVC — Anytime Post-Selection Verifier Contract

### 1. 一句话核心主张

**verifier 的 pointwise calibration 不能保证“从不断增长候选池中自适应选出的最大分样本”仍可靠；需要直接控制停止时刻被选输出的 conditional error。**

### 2. 具体科研 gap

典型 verifier 学的是：

\[
\Pr(Y=1\mid S=s)\approx s.
\]

部署却做：

\[
\hat y_\tau=\arg\max_{i\le N_\tau}S_i,
\]

其中候选数 `N_τ` 和停止时间 `τ` 由此前分数自适应决定。目标应为：

\[
\Pr(Y_{\hat y_\tau}=0\mid\mathcal F_\tau)\le\alpha,
\]

或至少给出 lower confidence bound 与 abstention。

### 3. 为什么现有工作没有完全解决

- Inference-Time Reward Hacking 已分析 BoN/BoP 过优化与 HedgeTune；
- MSV 联合看候选集并用于 streaming early stop；
- ORCA 做 conformal/test-time calibration；
- Certified SC 给 majority mode 的 anytime certificate；
- MoB 用 bootstrap 改善 imperfect verifier 下 BoN。

剩余 gap 只能是“adaptive post-selection error contract”，但相邻积木已非常齐全。

### 4. 方法机制和必要公式

对候选历史 `F_t` 构造 selected-risk e-process 或 time-uniform lower bound：

\[
L_t(\hat y_t)
\le
\Pr(Y_{\hat y_t}=1\mid\mathcal F_t)
\quad\forall t
\]

以至少 `1-δ` 概率同时成立。停止规则：

\[
\tau=\inf\{t:L_t(\hat y_t)\ge1-\alpha\}.
\]

必须与 per-N temperature scaling、isotonic、MSV、ORCA、HedgeTune、Certified SC、MoB 比较。

### 5. 与最近邻逐项区别

APVC 若有新意，只能在 selected-item correctness、arbitrary adaptive generation、misspecification 下 abstention、selection-history time-uniform contract 四点，而非一般 calibration。

### 6. Reviewer 最可能拒稿理由

1. “MSV + conformal。”
2. “Reward hacking 已有 HedgeTune。”
3. “保证依赖 exchangeability，adaptive generator 下不成立。”
4. “只证明模式稳定，不证明答案正确。”
5. “真实 label 部署时不可得。”
6. “简单 per-N calibration 已足够。”

### 7. 最强简单基线

per-N isotonic/temperature calibration + MSV streaming + HedgeTune。

### 8. Oracle ceiling

用规则真值计算每个 N、每个 stop rule 的 selected error。若 naive adaptive selection 引入的 error inflation `<2pp`，或简单 per-N calibration 修复 `>=80%`，KILL。

### 9. CPU Gate 0

Gaussian/mixture score-label simulator，包含 adaptive max、correlated candidates、distribution shift。

**PASS：** anytime empirical miscoverage `<=α+0.02`；coverage 比 Bonferroni/per-N baseline 高 `>=10pp`；misspecification 下正确 abstain。

### 10. 单 GPU Gate 1

- 100 MATH problems；N=1,2,4,8,16,32,64；
- 2 generator models × 2 verifiers；
- 固定 adaptive stop rules；
- 所有 candidate 有 rule-based ground-truth correctness。

**PASS：** naive selected-error inflation `>=3pp`；APVC 相对 per-N calibration、MSV、ORCA、HedgeTune 将 inflation 降低 `>=30%`；fixed low-N accuracy loss `<=2pp`；跨 model/verifier；coverage `>=50%`。

### 11. 推荐数据与 checkpoint

MATH-500、AMC/AIME；1.5B 与 7B generator；一个 PRM 与一个 outcome verifier。若只能用同一 verifier 同时产生 score 与 label，直接 KILL。

### 12. 消融和负对照

fixed N、adaptive N、independent/correlated samples、temperature、candidate diversity、verifier swap、rule label vs judge label、format/length matching。

### 13. 算力、显存、时间、存储

10–24 GPU-hours，24 GB，存储 30–60 GB，无训练。

### 14. 成功后的论文类型

trustworthy inference / statistical reliability 论文；保证强且跨多个 adaptive strategies 时 AAAI 上限约 7/10，否则是增量 calibration。

### 15. AAAI 适配度和 idea 上限

AAAI 适配 `7/10`；idea ceiling `7/10`；原始创新性偏低。

### 16. 当前决策

**不执行。直接邻居太强，除非先找到现有方法共同失败的明确 counterexample。**

---

## 7. 对 Top 1 的最强红队攻击

| 攻击 | 严重度 | 为什么可能致命 | 必须怎样击败 |
|---|---:|---|---|
| PUM 已有同一核心量 | 极高 | PUM 已把 prefix 评价定义为条件 solve-rate gain | 证明 target-policy transport 不是 per-policy finetune/policy embedding 可替代 |
| GenAC 已做 actor conditioning | 极高 | critic 跟 current actor 对齐不是新意 | 在未 rollout target actor、multi-logger OPE 场景显著胜出 |
| `V_0` 已宣称 any-policy value | 高 | policy capability context 已有 | 展示 state-zero 方法不能推广到 prefix 且 transport 有独立识别 |
| 标准 OPE 覆盖 estimator | 极高 | DR/MAGIC/FQE/MWL/MQL 成熟 | 不能只换 LLM notation；需 prefix-specific theorem 或 estimator |
| K=2 fresh rollout 更简单 | 极高 | 直接测量无模型偏差 | 等 token 预算至少 20% regret/RMSE 优势 |
| token support 使 IS 爆炸 | 极高 | 长序列 ratio 乘积导致低 ESS | 预注册 ESS/coverage；不能事后 clipping 挽救 |
| target policy 是 moving target | 高 | RL 更新后旧 value 过时 | 明确 snapshot horizon 与 staleness curve |
| prefix 注入不是自然状态 | 高 | foreign prefix 可能离开 target state distribution | 区分 intervention value 与 reachable value |
| rank inversion 由温度/长度驱动 | 高 | 不代表 semantic policy dependence | matched temperature、length、format、entropy 对照 |
| binary math verifier 太窄 | 中高 | 外推 agent/非可验证任务弱 | 至少加入 code/unit-test 或降低论文定位 |
| cross-tokenizer 不可比 | 高 | token ratio 无共同 action space | 主 estimator 只用同 tokenizer；跨架构只 direct rollout |
| policy fingerprint 泄漏 | 中高 | critic 记住模型 ID | unseen-policy holdout、ID shuffle、capability context |
| oracle ceiling 很小 | 极高 | estimand 存在也不值得建方法 | `<2pp` 或 `<10%` regret reduction 立即 KILL |
| 贡献只是 diagnostic | 高 | AAAI 方法深度不足 | estimator 必须改变决策且胜强基线 |
| theorem 太显然 | 中 | policy dependence 是定义直接结果 | 给可测试识别界、支持证书或有限样本保证 |

### 红队总判定

PTPU 的最危险之处不是“可能做不出来”，而是**即便实验做出来，也可能只证明一个已有定义的自然后果**。因此不能先训练再找故事；必须先用 oracle 与 strongest composed baseline 判定独立价值。

---

## 8. 唯一最终推荐，或明确 NO-GO

# **NO-GO：当前没有足够强的新候选**

理由按优先级排序：

1. **独立贡献不够。** Top 1 仍可被精确拆成 PUM + policy conditioning + OPE；
2. **简单基线过强。** fresh target-policy K=2/4 rollout 很可能覆盖实用收益；
3. **识别边界苛刻。** token support、长 horizon、moving target、foreign prefix 都使 estimator 容易失效；
4. **仓库先验不支持继续围绕 latent semantics。** 多条方向都出现“数学区别存在、操作优势不存在”；
5. **2026 直接邻居密度过高。** prefix value、counterfactual credit、set verifier、adaptive stop、OPE 均已有强工作；
6. **当前没有测得 oracle ceiling。** 在投入训练前，连问题最大可获收益都未知。

### 允许的唯一后续动作

执行 **PTPU CPU Gate 0**。这是不超过两天的证伪动作，不是研究方向承诺。

只有 CPU Gate 0 显示非标准且稳定的 rank inversion、support-aware estimator 明显超越 generic DR/MAGIC、且不是简单 ESS threshold，才可执行单 GPU Gate 1。Gate 1 即使 PASS，也只把结论升级为“重新审计”，不能自动升级为 GO。

---

## 9. 前两周可执行计划

### Week 1：先杀现象与识别

| 日 | 工作 | 固定输出 | 停止条件 |
|---:|---|---|---|
| D1 | 冻结 PTPU claim contract、文献碰撞矩阵、数据与 policy snapshot hash | `PTPU_CLAIM_CONTRACT.md`、source manifest | 发现完全相同 target-policy prefix OPE 论文则立即 NO-GO |
| D2 | 实现 finite-DAG exact oracle、IS/DR/MAGIC/FQE baselines | CPU tests + JSON | estimator 只是 DR 重命名则 KILL |
| D3 | 实现 support certificate、abstention、coverage tests | Gate-0 prereg + hash | 仅 ESS threshold 有效则 KILL |
| D4 | 运行 CPU Gate 0 全部 seeds | append-only records | 任一 frozen gate fail 即停止 |
| D5 | source/tokenizer/logprob audit；固定 Qwen policy snapshots | policy matrix | tokenizer/action/support 不一致则停止 |
| D6 | 10 题 engineering preflight，不看主 effect | runtime/sampler/control JSON | logprob、stop、verifier、repeatability fail |
| D7 | 30 题 calibration phenomenon run | calibration summary | inversion/ceiling/ESS 任一低于门，永久停止 |

### Week 2：只在 Week 1 全 PASS 后执行

| 日 | 工作 | 固定输出 | 停止条件 |
|---:|---|---|---|
| D8 | 冻结 confirmation split、阈值、policy pairs、token budget | signed prereg | 不允许再改 |
| D9–D10 | 30 题 confirmation target/direct rollouts | raw JSONL | 工程错误可修；科学阈值不改 |
| D11 | 运行 PUM+policy ID、GenAC-style、DR/MAGIC/FQE、K=2/4 baselines | baseline matrix | K=2 覆盖收益则 KILL |
| D12 | oracle selection regret 与 rank inversion analysis | frozen summary | ceiling `<2pp` KILL |
| D13 | length/temp/entropy/format/evaluator leakage controls | robustness report | 主效应被解释则 KILL |
| D14 | 独立复算与最终 decision | `PTPU_GATE1_DECISION.md` | 只输出 PASS-REAUDIT 或 KILL；不训练 |

---

## 10. CPU Gate 0 和单 GPU Gate 1 预注册

### 10.1 CPU Gate 0 冻结协议

**Protocol ID：** `ptpu-cpu-identification-v1-20260718`

#### Primary estimands

1. `V^π(h)` absolute error；
2. cross-policy rank inversion F1；
3. 95% interval coverage；
4. support abstention precision/recall；
5. selection regret。

#### Baselines

DM、trajectory IS、per-decision IS、WIS、DR、MAGIC、FQE、MWL/MQL。

#### PASS

- MAE `<=0.02` on supported strata；
- coverage `>=0.90`；
- inversion F1 `>=0.85`；
- false-confident unsupported rate `<=0.05`；
- candidate 在至少两个 non-pathological regimes 相对最强 generic baseline 的 RMSE 降低 `>=15%`；
- 候选公式不能退化为已有 baseline 的完全同式。

#### KILL

任一 PASS 条件失败；不得换 synthetic family、删 hard regimes、改阈值后重跑同一 confirmation seeds。

### 10.2 单 GPU Gate 1 冻结协议

**Protocol ID：** `ptpu-checkpoint-phenomenon-v1-20260718`

#### Primary estimands

\[
I_{\pi_a,\pi_b}
=
\Pr[\operatorname{rank}_h V^{\pi_a}(h)
\ne
\operatorname{rank}_h V^{\pi_b}(h)],
\]

\[
C_{\mathrm{oracle}}
=
\mathrm{Acc}_{\mathrm{oracle,target}}
-
\mathrm{Acc}_{\mathrm{best\ pooled}},
\]

\[
E_{\mathrm{budget}}
=
\frac{\mathrm{Regret}_{K=2\ fresh}-\mathrm{Regret}_{\mathrm{PTPU}}}
{\mathrm{Regret}_{K=2\ fresh}}.
\]

#### Primary PASS

- `I >=0.15`，cluster-bootstrap 95% lower bound `>0.10`；
- `Q75 |ΔV| >=0.10`；
- `C_oracle >=0.03` 或 normalized regret reduction `>=0.20`；
- `E_budget >=0.20`；
- candidate 胜 PUM+policy embedding、GenAC-style、DR、MAGIC、FQE；
- median ESS `>=10`；
- supported coverage `>=0.80`；
- 两个 policy pair、两个 decoding regimes 复现。

#### Mandatory negative controls

- same-policy split；
- policy ID shuffle；
- capability profile shuffle；
- length match；
- temperature match；
- entropy strata；
- format canonicalization；
- alternative exact verifier；
- unreachable-prefix flag；
- cross-tokenizer prohibition。

#### Binding KILL

任一 primary PASS 失败；任何主要结果由 mandatory control 解释；或 K=2 fresh rollout 等优。

#### Authorization

- Gate 1 之前：不训练；
- Gate 1 PASS 后：只授权新一轮 novelty/identifiability review；
- 不自动授权 RL、H20、7B 扩展或 paper claim。

---

## 11. 算力与成本预算

| 阶段 | 设备 | 时间 | 峰值显存/RAM | 存储 | 预算上限 | 授权 |
|---|---|---:|---:|---:|---:|---|
| 文献/source audit | CPU | 1–2 d | 8 GB RAM | <2 GB | $0 | 已完成主体 |
| PTPU CPU Gate 0 | 16-core CPU | <2 h | <16 GB RAM | <5 GB | $0–20 | **授权** |
| Engineering preflight | 1×12–24 GB GPU | 2–4 h | 10–18 GB | 5–10 GB | <$20 | Gate 0 PASS 后 |
| PTPU Gate 1 | 1×24 GB GPU | 24–40 GPU-h | 12–22 GB | 60–100 GB | <$150（规划假设） | 条件授权 |
| LRPE Gate 1 | 1×24 GB | 20–40 h | 12–22 GB | 50–100 GB | 不授权 | NO-GO |
| APVC Gate 1 | 1×24 GB | 10–24 h | 12–22 GB | 30–60 GB | 不授权 | NO-GO |
| 任何训练 | GPU | — | — | — | **$0** | 不授权 |
| H20/A100-80G/7B | 云 GPU | — | — | — | **$0** | 不授权 |

成本是计划上限，不是实时云市场报价。所有运行必须记录 generated tokens、verifier calls、model forward calls、wall time、device、peak allocated/reserved memory、raw artifact SHA-256。

---

## 12. 论文贡献结构和最危险审稿意见

### 12.1 只有 PTPU 全 gate 通过时，允许设想的论文结构

1. **Problem:** prefix utility is policy-indexed；
2. **Identification:** target-policy prefix value under multi-logger overlap；
3. **Method:** support-aware transport estimator with abstention；
4. **Theory:** finite-sample error/support bound，或 impossibility under no overlap；
5. **Phenomenon:** cross-policy rank reversal；
6. **Decision consequence:** target-specific prefix selection / process supervision；
7. **Compute baseline:** fresh K=1/2/4 target rollouts；
8. **Robustness:** temperature、length、format、verifier、reachability；
9. **Boundary:** same-tokenizer requirement、moving-policy staleness。

### 12.2 最危险 Reviewer 2 意见

> “The paper repackages the prefix-gain estimand of PUM, conditions the critic on policy identity as in GenAC/`V_0`, and applies textbook doubly robust off-policy evaluation. The empirical rank reversals are expected when solvers have different capabilities, while the proposed estimator is less reliable than two fresh target-policy rollouts under realistic language-model support mismatch. Thus the contribution is an application and diagnostic, not a new AAAI-level method.”

当前证据下，这一拒稿意见是**成立概率最高**的意见，而不是可以靠写作化解的误解。

### 12.3 必须用实验击败，而不能辩论的点

- PTPU 必须胜 K=2 fresh rollouts；
- 必须胜 PUM+policy embedding；
- 必须胜 generic DR/MAGIC/FQE；
- 必须有 `>=3pp` oracle ceiling 或 `>=20%` regret reduction；
- 必须跨 policy pair、decoding regime；
- 必须有高支持覆盖；
- 必须不是长度/温度/格式/entropy artifact；
- 必须说明为何 PUM/GenAC/`V_0` 不能直接获得同样收益。

在这些证据出现前，不应训练，不应先写 introduction，不应命名一个夸张的新 optimizer。

---

## 13. 完整参考文献与链接

### 13.1 固定仓库与正式决策

1. [LatentGRPO fixed commit](https://github.com/L-Dramatic/latentgrpo/tree/6f32fc37527f923e064927c8930c2c9a3d9f64a2)
2. [Next Direction Decision](https://github.com/L-Dramatic/latentgrpo/blob/6f32fc37527f923e064927c8930c2c9a3d9f64a2/research/NEXT_DIRECTION_DECISION_2026-07-18.md)
3. [PCMC A0 Result](https://github.com/L-Dramatic/latentgrpo/blob/6f32fc37527f923e064927c8930c2c9a3d9f64a2/research/policy_conditional_mixture_closure/A0_RESULT_2026-07-18.md)
4. [PCMC Decision](https://github.com/L-Dramatic/latentgrpo/blob/6f32fc37527f923e064927c8930c2c9a3d9f64a2/research/policy_conditional_mixture_closure/DECISION.md)
5. [PCMC Preregistration](https://github.com/L-Dramatic/latentgrpo/blob/6f32fc37527f923e064927c8930c2c9a3d9f64a2/research/policy_conditional_mixture_closure/CHECKPOINT_PREREGISTRATION.md)
6. [OMPI Decision](https://github.com/L-Dramatic/latentgrpo/blob/6f32fc37527f923e064927c8930c2c9a3d9f64a2/research/observable_marginal_policy/DECISION.md)
7. [OMPI Exact Gate](https://github.com/L-Dramatic/latentgrpo/blob/6f32fc37527f923e064927c8930c2c9a3d9f64a2/research/observable_marginal_policy/EXACT_GATE.md)
8. [LPCA Method Matrix](https://github.com/L-Dramatic/latentgrpo/blob/6f32fc37527f923e064927c8930c2c9a3d9f64a2/research/policy_contract_audit/METHOD_MATRIX.md)
9. [LPCA Experiment Log](https://github.com/L-Dramatic/latentgrpo/blob/6f32fc37527f923e064927c8930c2c9a3d9f64a2/research/policy_contract_audit/EXPERIMENT_LOG.md)
10. [Mixed-Measure Policy Decision](https://github.com/L-Dramatic/latentgrpo/blob/6f32fc37527f923e064927c8930c2c9a3d9f64a2/research/mixed_measure_policy/DECISION.md)
11. [Coupled Group Exploration Decision](https://github.com/L-Dramatic/latentgrpo/blob/6f32fc37527f923e064927c8930c2c9a3d9f64a2/research/coupled_group_exploration/DECISION.md)
12. [Coordinate Invariance Experiment Log](https://github.com/L-Dramatic/latentgrpo/blob/6f32fc37527f923e064927c8930c2c9a3d9f64a2/research/coordinate_invariance/EXPERIMENT_LOG.md)
13. [SWITCH C2 Postmortem](https://github.com/L-Dramatic/latentgrpo/blob/6f32fc37527f923e064927c8930c2c9a3d9f64a2/research/coordinate_invariance/SWITCH_C2_ATTEMPT5_POSTMORTEM.md)
14. [Behavioral Geometry README](https://github.com/L-Dramatic/latentgrpo/blob/6f32fc37527f923e064927c8930c2c9a3d9f64a2/research/behavioral_geometry/README.md)
15. [Forward-KL Kill Report](https://github.com/L-Dramatic/latentgrpo/blob/6f32fc37527f923e064927c8930c2c9a3d9f64a2/research/behavioral_geometry/P1_SACRIFICIAL_DISCOVERY_KILL_REPORT_ZH.md)
16. [Top-K Concrete](https://github.com/L-Dramatic/latentgrpo/blob/6f32fc37527f923e064927c8930c2c9a3d9f64a2/research/topk_concrete/README.md)
17. [Score-Squashed Gumbel](https://github.com/L-Dramatic/latentgrpo/blob/6f32fc37527f923e064927c8930c2c9a3d9f64a2/research/score_squashed_gumbel/README.md)

### 13.2 Latent / soft reasoning

18. [Coconut](https://arxiv.org/abs/2412.06769) (2024)
19. [Soft Reasoning](https://proceedings.mlr.press/v267/) (ICML 2025)
20. [Latent-GRPO](https://arxiv.org/abs/2604.27998) (2026)
21. [SofT-GRPO](https://arxiv.org/abs/2511.06411) (2025)
22. [LEPO](https://arxiv.org/abs/2604.17892) (2026)
23. [NF-CoT](https://arxiv.org/abs/2606.06447) (2026)
24. [SWITCH](https://arxiv.org/abs/2606.13106) (2026)
25. [Latent Thought Flow](https://arxiv.org/abs/2606.16222) (2026)
26. [Chain of Superposition / Latent-SFT](https://arxiv.org/abs/2510.15522) (2025)
27. [Soft Thinking](https://arxiv.org/abs/2505.15778) (2025)
28. [Mixture of Inputs](https://arxiv.org/abs/2505.14827) (2025)
29. [Soft Concept Mixing](https://arxiv.org/abs/2511.16885) (2025)

### 13.3 RLVR 与优化

30. [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300) (2024)
31. [RLOO](https://arxiv.org/abs/2402.14740) (2024)
32. [DAPO](https://arxiv.org/abs/2503.14476) (2025)
33. [Dr.GRPO](https://arxiv.org/abs/2503.20783) (2025)
34. [Target Policy Optimization](https://arxiv.org/abs/2604.06159) (2026)
35. [MinPRO](https://arxiv.org/abs/2601.22718) (2026)
36. [Step-GRPO](https://ojs.aaai.org/index.php/AAAI/article/view/40441) (AAAI-26)
37. [Self-Rewriting RL](https://ojs.aaai.org/index.php/AAAI/article/view/40738) (AAAI-26)
38. [PURE](https://proceedings.neurips.cc/paper_files/paper/2025/hash/be91eb86eb74efc055cff83e953f86ce-Abstract-Conference.html) (NeurIPS 2025)
39. [Fast On-Policy Prefix Distillation](https://aclanthology.org/2026.findings-acl.1276/) (ACL Findings 2026)
40. [VPPO / Save the Good Prefix](https://aclanthology.org/2026.findings-acl.1767/) (ACL Findings 2026)

### 13.4 Prefix utility、critic、credit assignment

41. [Math-Shepherd](https://arxiv.org/abs/2312.08935) (2024)
42. [ReasonFlux-PRM](https://proceedings.neurips.cc/paper_files/paper/2025/hash/26618fb384d3873b8ef6ab292a69095b-Abstract-Conference.html) (NeurIPS 2025)
43. [PUM](https://arxiv.org/abs/2606.07190) (2026)
44. [GenAC](https://arxiv.org/abs/2604.10701) (2026)
45. [`V_0`](https://arxiv.org/abs/2602.03584) (2026)
46. [InT](https://arxiv.org/abs/2601.14209) (2026)
47. [IBPO](https://arxiv.org/abs/2605.16302) (2026)
48. [CVT-RL](https://arxiv.org/abs/2606.05263) (2026)
49. [BiPACE](https://arxiv.org/abs/2606.25556) (2026)

### 13.5 Test-time compute、verifier、calibration

50. [Scaling LLM Test-Time Compute](https://openreview.net/forum?id=4FWAwZtd2n) (ICLR 2025)
51. [Off-Trajectory Reasoning](https://openreview.net/forum?id=hVUIguIm14) (ICLR 2026)
52. [MSV](https://arxiv.org/abs/2603.03417) (2026)
53. [Inference-Time Reward Hacking](https://proceedings.neurips.cc/paper_files/paper/2025/hash/590a0cc0306c1c63e2d66a51a407718f-Abstract-Conference.html) (NeurIPS 2025)
54. [ST-BoN](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ed45d6a03de84cc650cae0655f699356-Abstract-Conference.html) (NeurIPS 2025)
55. [Majority of the Bests](https://proceedings.neurips.cc/paper_files/paper/2025/hash/36556567e8437f137da23047309155dd-Abstract-Conference.html) (NeurIPS 2025)
56. [ORCA](https://arxiv.org/abs/2604.01170) (2026)
57. [Certified Self-Consistency](https://arxiv.org/abs/2510.17472) (2025/2026)
58. [ThinkBooster](https://arxiv.org/abs/2606.06915) (2026)
59. [Test-time Prompt Intervention](https://ojs.aaai.org/index.php/AAAI/article/view/40718) (AAAI-26)
60. [OpenDeepThink](https://arxiv.org/abs/2605.15177) (2026)
61. [Look Before You Leap](https://aclanthology.org/2026.eacl-long.367/) (EACL 2026)

### 13.6 OPE 与 identification

62. [Jiang & Li, Doubly Robust OPE](https://proceedings.mlr.press/v48/jiang16.html) (ICML 2016)
63. [Thomas & Brunskill, MAGIC](https://proceedings.mlr.press/v48/thomasa16.html) (ICML 2016)
64. [Uehara et al., MWL/MQL](https://proceedings.mlr.press/v119/uehara20a.html) (ICML 2020)
65. [Kallus & Uehara, Double Reinforcement Learning](https://proceedings.mlr.press/v119/kallus20b.html) (ICML 2020)
66. [Duan et al., Minimax-Optimal OPE](https://proceedings.mlr.press/v119/duan20b.html) (ICML 2020)
67. [Su et al., DR with Shrinkage](https://proceedings.mlr.press/v119/su20a.html) (ICML 2020)
68. [Hao et al., Bootstrapping FQE](https://proceedings.mlr.press/v139/hao21b.html) (ICML 2021)
69. [ADWM for LLM-agent OPE](https://arxiv.org/abs/2606.05558) (2026)

### 13.7 推荐 checkpoint / 官方模型页

70. [Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B)
71. [Qwen2.5-Math-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct)
72. [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)

---

# 最终一行

## **NO-GO：当前没有足够强的新候选**

PTPU 仅保留为两周、零训练、低预算的证伪探针；它不是主方向推荐，也不得在未击败 PUM + policy conditioning + 标准 OPE + fresh-rollout baseline 前进入训练。
