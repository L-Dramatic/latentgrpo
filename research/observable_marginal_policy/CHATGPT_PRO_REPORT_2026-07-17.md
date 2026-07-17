# Latent-GRPO：AAAI 主会级研究选题独立审计、文献碰撞与 Reviewer 2 红队

**审计日期：** 2026-07-17  
**目标仓库：** [`L-Dramatic/latentgrpo`](https://github.com/L-Dramatic/latentgrpo)  
**冻结分支：** `main`  
**冻结提交：** [`9ccd18295941b59d4862ba5a790f7a44c4b9fae2`](https://github.com/L-Dramatic/latentgrpo/tree/9ccd18295941b59d4862ba5a790f7a44c4b9fae2)  
**官方 Latent-GRPO 源码审计提交：** [`c0994fb781a2d180662bb522d8ff3e8638dcf56d`](https://github.com/DJC-GO-SOLO/Latent-GRPO/tree/c0994fb781a2d180662bb522d8ff3e8638dcf56d)  
**判断立场：** 独立研究选择 + 严格 Reviewer 2；不继承仓库旧报告排名，不把负结果重新包装成方法。

---

## 0. 执行结论

> **总判断：当前有一个值得做“checkpoint inference 证伪 gate”的首选假设，但当前没有值得直接启动 full RL training 的 idea。**

### 唯一首选

**Observable-Marginal Policy Improvement（OMPI）**  
中文：**可观测边际策略改进**。

一句话：**把私有 latent path 积分掉，直接拟合 verifier 所见的 reward-tilted response distribution；用跨 latent-path 的 response likelihood responsibility 分配信用，而不是给生成该答案的单一路径广播同一个 terminal advantage。**

它是目前唯一同时满足以下条件的候选：

- 不需要 executed latent action 的错误 likelihood；
- 明确改变优化单位：从“私有 latent trajectory”改为“可观测 response marginal”；
- 有精确的有限候选目标、责任分解和可证伪机制；
- 可在不训练的 checkpoint gate 中被快速否决；
- 若机制成立，有可能形成“latent-variable policy improvement”而不是 Latent-GRPO 局部补丁。

但它有严重的 **[COLLISION RISK]**：容易被评价为 “TPO + IWAE/Rao–Blackwellization + latent CoT”。因此目前仅授权做最容易杀死它的 cross-path responsibility gate；**未授权训练**。

### 唯一备选

**Policy-Conditional Mixture Closure（PCMC）**  
中文：**策略条件混合闭包**。

一句话：**一个 soft token 应代表其 component continuations 的概率混合，而不应只等于 token embeddings 的算术平均；训练一个单次前向可执行的 behavioral barycenter，使其 continuation law 接近 hard-branch mixture。**

它直接攻击 vocabulary-superposition latent reasoning 的基础语义问题，且不碰 latent likelihood。但它与 Soft Thinking、Soft Concept Mixing、probabilistic mixup 和 distillation 高度邻近；只有在“non-closure 对 reward 有因果损害”以及“低成本 barycenter 确实存在”两个 gate 同时通过时，才值得训练。

### 明确不建议继续

以下方向永久不进入本轮候选池：

- ordinary latent-likelihood repair、exact-density substitution；
- clipping-aware PPO/CAPG 改名；
- mixed-measure likelihood ratio 修复；
- Top-K Concrete / factorized Top-K / score-squashed Gumbel 继续调参；
- coordinate-invariant / Fisher trust region / FCTR；
- horizon-gap、HorizonGuard、PaTR/ACRT；
- correlated、antithetic、Arithmetic Sampling、RQMC group exploration；
- generic entropy、diversity reward、KL schedule、curriculum；
- generic counterfactual step credit、process reward、test-time tree search；
- 仅做 policy-contract audit、benchmark 或工程修复并包装成 method paper。

### 本轮最重要的独立判断

**[INFERENCE]** 仓库已证明“latent sampler 与训练 surrogate 不同”是真实事实，但也已经反复证明“把 surrogate 换成更精确的 latent density”不是性能方向。更有普适意义的剩余问题不是“latent action 的 likelihood 到底是什么”，而是：

> **当 reward 只取决于可见答案时，训练是否应该首先优化由所有私有 latent paths 共同诱导的 response marginal，而不是把每条私有路径当作独立、可识别、可正确打分的 RL action？**

这个 gap 仍未被仓库负结果直接否定；但它是否具有足够大的实际效应，完全是开放问题。

---

## 1. 审计范围、访问情况与证据等级

### 1.1 用户指定文件读取清单

以下文件均在冻结提交 `9ccd1829...` 上实际读取：

| 指定内容 | 状态 | 固定链接 |
|---|---|---|
| `RESEARCH_IDEA_ARCHIVE.md` | **已读** | [文件](https://github.com/L-Dramatic/latentgrpo/blob/9ccd18295941b59d4862ba5a790f7a44c4b9fae2/RESEARCH_IDEA_ARCHIVE.md) |
| `AAAI_DEEP_RESEARCH_REVIEW_2026-07-14.md` | **已读** | [文件](https://github.com/L-Dramatic/latentgrpo/blob/9ccd18295941b59d4862ba5a790f7a44c4b9fae2/AAAI_DEEP_RESEARCH_REVIEW_2026-07-14.md) |
| `AAAI_NOVELTY_COLLISION_AUDIT.md` | **已读** | [文件](https://github.com/L-Dramatic/latentgrpo/blob/9ccd18295941b59d4862ba5a790f7a44c4b9fae2/AAAI_NOVELTY_COLLISION_AUDIT.md) |
| `research/mixed_measure_policy/DECISION.md` | **已读** | [文件](https://github.com/L-Dramatic/latentgrpo/blob/9ccd18295941b59d4862ba5a790f7a44c4b9fae2/research/mixed_measure_policy/DECISION.md) |
| `research/mixed_measure_policy/MATHEMATICAL_RED_TEAM.md` | **已读** | [文件](https://github.com/L-Dramatic/latentgrpo/blob/9ccd18295941b59d4862ba5a790f7a44c4b9fae2/research/mixed_measure_policy/MATHEMATICAL_RED_TEAM.md) |
| `research/coupled_group_exploration/DECISION.md` | **已读** | [文件](https://github.com/L-Dramatic/latentgrpo/blob/9ccd18295941b59d4862ba5a790f7a44c4b9fae2/research/coupled_group_exploration/DECISION.md) |
| `research/coupled_group_exploration/EXACT_GATE.md` | **已读** | [文件](https://github.com/L-Dramatic/latentgrpo/blob/9ccd18295941b59d4862ba5a790f7a44c4b9fae2/research/coupled_group_exploration/EXACT_GATE.md) |
| `research/policy_contract_audit/README.md` | **已读** | [文件](https://github.com/L-Dramatic/latentgrpo/blob/9ccd18295941b59d4862ba5a790f7a44c4b9fae2/research/policy_contract_audit/README.md) |
| `research/policy_contract_audit/METHOD_MATRIX.md` | **已读** | [文件](https://github.com/L-Dramatic/latentgrpo/blob/9ccd18295941b59d4862ba5a790f7a44c4b9fae2/research/policy_contract_audit/METHOD_MATRIX.md) |
| `research/policy_contract_audit/EXPERIMENT_LOG.md` | **已读** | [文件](https://github.com/L-Dramatic/latentgrpo/blob/9ccd18295941b59d4862ba5a790f7a44c4b9fae2/research/policy_contract_audit/EXPERIMENT_LOG.md) |
| `research/topk_concrete/README.md` | **已读** | [文件](https://github.com/L-Dramatic/latentgrpo/blob/9ccd18295941b59d4862ba5a790f7a44c4b9fae2/research/topk_concrete/README.md) |
| `research/topk_concrete/EXPERIMENT_LOG.md` | **已读** | [文件](https://github.com/L-Dramatic/latentgrpo/blob/9ccd18295941b59d4862ba5a790f7a44c4b9fae2/research/topk_concrete/EXPERIMENT_LOG.md) |
| `research/score_squashed_gumbel/README.md` | **已读** | [文件](https://github.com/L-Dramatic/latentgrpo/blob/9ccd18295941b59d4862ba5a790f7a44c4b9fae2/research/score_squashed_gumbel/README.md) |
| `research/score_squashed_gumbel/EXPERIMENT_LOG.md` | **已读** | [文件](https://github.com/L-Dramatic/latentgrpo/blob/9ccd18295941b59d4862ba5a790f7a44c4b9fae2/research/score_squashed_gumbel/EXPERIMENT_LOG.md) |
| 官方 sampler | **已读** | [`sampler.py`](https://github.com/DJC-GO-SOLO/Latent-GRPO/blob/c0994fb781a2d180662bb522d8ff3e8638dcf56d/sglang_latent_reasoning_pkg/python/sglang/srt/layers/sampler.py) |
| 官方 actor replay | **已读** | [`dp_actor.py`](https://github.com/DJC-GO-SOLO/Latent-GRPO/blob/c0994fb781a2d180662bb522d8ff3e8638dcf56d/verl-0.4.x/verl/workers/actor/dp_actor.py) |
| 官方 latent objective | **已读** | [`torch_functional.py`](https://github.com/DJC-GO-SOLO/Latent-GRPO/blob/c0994fb781a2d180662bb522d8ff3e8638dcf56d/verl-0.4.x/verl/utils/torch_functional.py) |
| 官方 GRPO/PPO core | **已读** | [`core_algos.py`](https://github.com/DJC-GO-SOLO/Latent-GRPO/blob/c0994fb781a2d180662bb522d8ff3e8638dcf56d/verl-0.4.x/verl/trainer/ppo/core_algos.py) |
| 仓库 source-faithful replay | **已读** | [`official_replay.py`](https://github.com/L-Dramatic/latentgrpo/blob/9ccd18295941b59d4862ba5a790f7a44c4b9fae2/research/topk_concrete/official_replay.py) |

此外，为防止复活已经在较晚提交中否决的方向，还读取了：

- [`research/coordinate_invariance/EXPERIMENT_LOG.md`](https://github.com/L-Dramatic/latentgrpo/blob/9ccd18295941b59d4862ba5a790f7a44c4b9fae2/research/coordinate_invariance/EXPERIMENT_LOG.md)；
- [`SWITCH_C2_ATTEMPT5_POSTMORTEM.md`](https://github.com/L-Dramatic/latentgrpo/blob/9ccd18295941b59d4862ba5a790f7a44c4b9fae2/research/coordinate_invariance/SWITCH_C2_ATTEMPT5_POSTMORTEM.md)；
- [`research/behavioral_geometry/README.md`](https://github.com/L-Dramatic/latentgrpo/blob/9ccd18295941b59d4862ba5a790f7a44c4b9fae2/research/behavioral_geometry/README.md)；
- [`P1_SACRIFICIAL_DISCOVERY_KILL_REPORT_ZH.md`](https://github.com/L-Dramatic/latentgrpo/blob/9ccd18295941b59d4862ba5a790f7a44c4b9fae2/research/behavioral_geometry/P1_SACRIFICIAL_DISCOVERY_KILL_REPORT_ZH.md)。

### 1.2 无法访问项

**用户点名的文件：无无法访问项。**

本地匿名 `git clone` 因执行环境 DNS 解析失败而不可用；因此读取通过 GitHub 连接器按固定 SHA 完成。这不影响上述文件内容核验。

**[OPEN QUESTION] source pin 冲突：**

- `policy_contract_audit/SOURCE_MANIFEST.json` 和实际官方仓库使用 `c0994fb781a2d180662bb522d8ff3e8638dcf56d`；
- `mixed_measure_policy/SOURCE_CONTRACT.md` 写的是 `c0994fbefb9de023912878534c7ae213b44b1966`；
- 后者在官方仓库中无法解析为 commit。

本报告以可解析、与 manifest 和实际文件一致的 `c0994fb781...` 为准，并把 `c0994fbefb...` 视为需要仓库作者修正的 provenance typo；不会静默合并二者。

### 1.3 证据标签

- **[VERIFIED][SOURCE]**：由固定提交源码或配置直接决定。
- **[VERIFIED][EXPERIMENT]**：由仓库冻结实验、结果文件或复算支持。
- **[INFERENCE]**：从源码和实验推出的合理解释，但不是已证明事实。
- **[OPEN QUESTION]**：目前没有足够证据。
- **[COLLISION RISK]**：与现有方法或理论祖先重叠，可能被审稿人判为组合或迁移。
- **[KILL CONDITION]**：一旦出现就停止，不事后改阈值救活。

---

## 2. 第一步：重建项目事实

## 2.1 官方 Latent-GRPO sampler 实际做了什么

设当前 latent step 的全词表 logits 为 \(l\)，词表嵌入为 \(E_v\)，要求保留 \(K\) 个 component。

**[VERIFIED][SOURCE]** 固定源码的执行顺序是：

1. 对全词表做 `log_softmax` 得 \(\log p_v\)；
2. 按 `top-p` 构造 policy-dependent candidate mask，同时至少保留 `max_topk=K` 个 token；
3. 对 candidate 采样标准 Gumbel 噪声；
4. 把噪声硬裁剪到 \([-1.5,3.0]\)；
5. 默认 one-sided 模式下再做从下界平移的非负噪声处理，并乘 `noise_scale`；
6. 将噪声加到 candidate log probability；
7. 在加噪后选 ordered Top-K token IDs \(s_1,\ldots,s_K\)；
8. 对 selected scores 做 softmax，得 mixture weights \(q_1,\ldots,q_K\)；
9. 执行
   \[
   e_{\mathrm{exec}}=\sum_{k=1}^{K}q_k E_{s_k};
   \]
10. 把第一个 selected ID \(s_1\) 暴露为 proxy token，写入普通 token/control stream。

因此 sampler 不是“抽一个连续 Gumbel action”，也不是“先选固定 Top-K 再在 simplex 上采样”。它包含：

\[
a_t=\big(C_\theta,\;S_\theta,\;g^{\mathrm{clip}},\;q_\theta,\;e_{\mathrm{exec}},\;s_1\big),
\]

其中 candidate set \(C_\theta\)、ordered support \(S_\theta\)、clipping atoms、weights、executed embedding 和 proxy 都有不同语义。

### 重要配置事实

**[VERIFIED][SOURCE]**

- 官方低难度脚本：1B LLaMA3.2 latent-SFT，GSM8K-style，`n=8` rollouts，8 GPUs，10 epochs，response length 128；
- 官方高难度脚本：7B Qwen2.5-Math，DAPO-Math-17k，`n=8`，8 GPUs，5 epochs，response length 4096；
- 两者默认 `max_topk=10`、Gumbel clipping、one-sided noise；
- token embeddings 在 RL 中冻结。

链接：[`README`](https://github.com/DJC-GO-SOLO/Latent-GRPO/blob/c0994fb781a2d180662bb522d8ff3e8638dcf56d/README.md)、[1B script](https://github.com/DJC-GO-SOLO/Latent-GRPO/blob/c0994fb781a2d180662bb522d8ff3e8638dcf56d/Latent-GRPO-gsm8k-llama3.sh)、[7B script](https://github.com/DJC-GO-SOLO/Latent-GRPO/blob/c0994fb781a2d180662bb522d8ff3e8638dcf56d/Latent-GRPO-math500-qwen.sh)。

## 2.2 optimizer 实际做了什么

### rollout replay

**[VERIFIED][SOURCE]** 训练 actor 从 rollout 中读取：

- stored selected IDs；
- stored selected perturbed scores；
- ordinary hard tokens；
- latent/visible mask。

它用 stored score 的 softmax 重建 mixture embedding，并把该 embedding `detach` 后作为 `inputs_embeds` 喂入当前模型。也就是说，当前训练 forward 重新计算 transformer conditional likelihood，但不会沿“rollout 时如何选出这组 support/score”反向传播。

### latent surrogate

对每个 selected ID \(s_k\)，当前模型给出 \(\log p_\theta(s_k)\)。令 stored selected score 为 \(r_k\)，重建

\[
\Delta_k=r_k-\log p_\theta(s_k).
\]

源码计算标准 Gumbel log-density 的 component：

\[
\ell_k^{\mathrm{surr}}
=-\Delta_k-\exp(-\Delta_k),
\qquad
\ell_t^{\mathrm{latent}}
=\frac1K\sum_{k=1}^{K}\ell_k^{\mathrm{surr}}.
\]

**[VERIFIED][SOURCE]** 它没有包含：

- dynamic candidate mask 的概率；
- post-noise ordered Top-K selection event 的概率；
- clipping 产生的 point masses；
- executed mixture 的 pushforward density；
- proxy/control event 的独立概率。

当 advantage 为负且特定 reconstructed margin 落入源码条件时，objective 还使用 **forward value 不变、backward gradient 改向** 的 straight-through 分支。随后它被放入普通 PPO ratio / clip / dual-clip machinery；visible suffix 使用普通 categorical token likelihood。

### GRPO advantage

**[VERIFIED][SOURCE]** 终局 reward 在同 prompt 的 rollout group 内做 group-relative centering/scaling，再广播到 response positions。仓库还记录了某些 first-step path-selection 逻辑，但这不改变上述 latent surrogate 不是 executed-action likelihood 的事实。

## 2.3 四个语义对象不能混用

| 对象 | 精确定义 | 实际作用 | 不是 |
|---|---|---|---|
| executed latent mixture | \(\sum_k q_kE_{s_k}\) | 真正送入模型的连续输入 | 不是单个 token，也不是离散分支的概率混合 |
| selected support | 加噪、裁剪、动态 candidate 后的 ordered Top-K IDs | 决定 mixture components；被归档 | 不是事前固定 support |
| proxy token | 第一 selected ID \(s_1\) | 写入 token/control history，影响 latent/visible 切换和请求逻辑 | 不是完整 mixture 的充分统计量 |
| training surrogate | selected-score Gumbel component 的均值 + ST backward | 构造 PPO ratio 的 latent 部分 | 不是 executed mixture 的 log density |

**[INFERENCE]** 这不是一个小的“日志字段错位”，而是至少三个 policy contracts 叠在同一 trajectory 上：continuous execution、discrete control、surrogate optimization。

## 2.4 仓库已经验证了哪些现象

| 现象 | 证据 | 结论等级 |
|---|---|---|
| selected clipping atoms 很常见 | selected upper atom 约 27.89%–35.1% | **[VERIFIED][EXPERIMENT]** |
| clipping 会大幅改变 ordered support | 至少 98.65% samples 的完整 ordered Top-K support 改变 | **[VERIFIED][EXPERIMENT]** |
| dynamic current support 造成 replay support violation | old clean actions 平均约 7.69% 被 current top-p support 拒绝 | **[VERIFIED][EXPERIMENT]** |
| proxy 与 soft execution 的主导语义经常不一致 | LEPO Stage A proxy disagreement 约 51.38%；source-faithful v2 最低约 54.99% | **[VERIFIED][EXPERIMENT]**，但这不是性能因果证明 |
| surrogate 与 alternative gradient 显著不同 | B2 exact/surrogate gradient cosine 中位数 0.79481，relative error 中位数 0.70711 | **[VERIFIED][EXPERIMENT]** |
| 更精确 density 不自动更好 | exact local gain 0.0063270，released surrogate 0.0079559，差值 -0.0016289；各温度均为负 | **[VERIFIED][EXPERIMENT]** |
| public-base stress test 可能退化 | 60.16% latent steps 为 singleton support | **[VERIFIED][EXPERIMENT]**；不能代替 trained LEPO checkpoint |
| Top-K exact law / factorization 可写出，但 effect gate 不够 | SCLPO、FTK-PO 多项 frozen gate 失败 | **[VERIFIED][EXPERIMENT]** |
| smooth score squash 没有保留 hard-clipped mixture concentration | entropy/max-weight gap 超 frozen threshold | **[VERIFIED][EXPERIMENT]** |
| antithetic coupling 提高 mixed-reward groups，但会偏置 group-relative baseline | \(p=0.3\) 时 IID/true 0.21，antithetic LOO expectation 0.30，+42.86% | **[VERIFIED][EXACT]** |
| V32/FCTR 没有通过 SWITCH calibration | 16/16 calibration 完成；无 global gain 通过，四个 probe scales 均输给简单 V3 | **[VERIFIED][EXPERIMENT]** |
| forward horizon gap 在冻结 Family A 上不存在 | 0/8 robust flips，late mass 0，median \(D_8/D_{64}=0.999839\) | **[VERIFIED][EXPERIMENT]** |

### 不能从这些结果推出的结论

- **不能**说 released surrogate “数学上完全错误，所以性能一定差”；
- **不能**说 exact latent density 不存在就没有任何可用梯度；
- **不能**说 proxy mismatch 本身造成 reward 下降；
- **不能**把某个 source contract mismatch 直接当作训练瓶颈；
- **不能**从单个 public checkpoint 推断所有 latent architecture；
- **不能**忽略已冻结的失败阈值后重跑更宽松版本。

## 2.5 已永久 KILL 的候选及原因

| 方向 | 最终状态 | 绑定原因 |
|---|---|---|
| ordinary latent likelihood 修复 | **KILL** | executed law 含 moving atoms、dynamic support 与 discrete/continuous混合；近 GRPO 的普通 ratio 不成立 |
| exact-density substitution | **KILL** | 数学差异真实，但 frozen B2 中 local utility 一致更差 |
| clipping-aware PPO / CAPG rename | **KILL** | CAPG 只解决 fixed clipping atoms；无法处理 policy-dependent moving atoms 与黑箱长后缀 |
| Mixed-Measure Latent Policy Optimization | **KILL** | theorem 有价值，实用 estimator 需要边界项、反事实或 finite difference；碰撞 HPO/CAPG/weak derivatives |
| Top-K Concrete / SCLPO | **KILL as mainline** | exact clean law 成立，但 official replay 中关键独立 method effect 不够 |
| FTK-PO | **KILL** | ordered support term 几乎支配 joint KL；factorization 没产生可训练新优势 |
| score-squashed Gumbel | **KILL** | smooth common-support law不能复现 hard-clipped mixture concentration |
| Simplex-GRPO | **KILL standalone** | exact quotient density可算，但 gradient/clip effect 太小 |
| FCTR / coordinate/Fisher trust region | **KILL exact protocol** | SWITCH calibration 失败、数值局部性弱、V32 全尺度输简单 baseline |
| horizon gap / HorizonGuard / PaTR/ACRT | **KILL** | 最贴近 source 的牺牲实验无 ranking flip、无 late mass |
| correlated/antithetic group exploration | **KILL** | 保持 marginal 不等于保持 GRPO estimator；cross-rollout baseline term 造成偏差 |
| Arithmetic Sampling / RQMC direct migration | **KILL as novelty** | 去掉 group baseline 后只剩已有 unbiased diversity/variance-reduction方法 |
| CSTR support/conformal standalone | **KILL headline** | density 不等于 reasoning validity，漂移下 coverage 不自动成立；只能是模块 |
| generic entropy/diversity/KL/curriculum | **KILL** | 高度拥挤、缺少 latent-specific机制，且用户明确排除 |
| diagnostics/benchmark-only | **不作为 method paper** | 可诚实发表 analysis，但不能伪装优化方法 |

## 2.6 当前真正未解决的科学 gap

### Gap A：observable objective 与 private latent path 的错层

终局 verifier 观察 \(y\) 并给 \(R(x,y)\)，却不观察 latent path \(z_{1:T}\)。多个 latent paths 可能诱导同一或相近 response law。

**[OPEN QUESTION]** 现有 latent RL 是否把大量梯度预算浪费在区分 reward-equivalent private paths，而没有充分聚合它们对可见答案的证据？

这比 likelihood repair 更普适：任何 hidden recurrence、soft token、flow latent 或 internal planner 都有 private computation / observable outcome 的层级差。

### Gap B：arithmetic embedding mixture 不具有 behavioral mixture semantics

现有方法常把

\[
\bar e=\sum_iq_iE_i
\]

解释为“同时保留多个 concepts/paths”，但一般

\[
P_\theta(\cdot\mid h,\bar e)
\neq
\sum_iq_iP_\theta(\cdot\mid h,E_i).
\]

**[OPEN QUESTION]** 这种 non-closure 是否只是无害的模型非线性，还是会系统性破坏 latent exploration、credit 和可见答案？

### Gap C：latent credit 的自然干预单位尚未找到

zero/noise/interpolation 往往 off-support；自然 alternative rollout 很贵；coordinate metric 又已失败。

**[INFERENCE]** 可行方向应尽量使用“已有自然 path 对同一可见 response 的 likelihood responsibility”，而不是人为 latent intervention。

### 本报告的优先级

1. **先检验 Gap A 是否有足够大的 cross-path explanatory mass；**
2. 若 Gap A 不成立，再检验 Gap B 的 causal harm 与可压缩 barycenter；
3. 两者都失败，则结论是：**当前没有值得训练的 idea。**



---

## 3. 第二步：2024–2026 文献检索与碰撞地图

本节优先给出论文原文、官方代码或正式 proceedings。分类不是互斥的；同一工作可能跨越 latent reasoning、RLVR、credit 与 inference。

## 3.1 Latent reasoning / continuous thoughts / soft tokens

| 工作 | 实际方法对象 | 对本项目的含义 |
|---|---|---|
| [Pause Tokens](https://arxiv.org/abs/2310.02226) | 在输入中加入可学习/固定 pause tokens，给 transformer 更多 hidden computation slots | 说明“无可见语义的内部计算 token”早已存在；仅增加 latent steps 不新 |
| [Coconut](https://arxiv.org/abs/2412.06769) | 把前一层 hidden state 递归作为下一步输入，形成连续思维 recurrence | hidden-state latent reasoning 的关键基线 |
| [SoftCoT](https://arxiv.org/abs/2502.12134) | 用 soft token distribution/embedding 传递中间思维 | 直接祖先；soft token 不是新贡献 |
| [Soft Thinking](https://arxiv.org/abs/2505.15778) | 对下一 token distribution 做 embedding 加权平均，训练外直接递归 | 明确采用 arithmetic mixture；PCMC 的最近对象之一 |
| [Text Generation Beyond Discrete Token Sampling / MoI](https://arxiv.org/abs/2505.14827) | 把 sampled token 与未使用的 token distribution 通过 Bayesian-style posterior expectation 混合为下一输入 | 说明 distribution-carrying input 已有 training-free强基线 |
| [LLMs are Single-threaded Reasoners](https://arxiv.org/abs/2508.03440) | probe soft thinking，报告后续 decoding 常由最强 component 主导，并用 Dirichlet/Gumbel randomness 缓解 | 对“soft mixture 自动并行探索”构成直接反证风险 |
| [LLM Latent Reasoning as Chain of Superposition / Latent-SFT](https://arxiv.org/abs/2510.15522) | vocabulary-space superposition latent chain 的 SFT 初始化 | 官方 Latent-GRPO 的直接前置训练范式 |
| [SofT-GRPO](https://arxiv.org/abs/2511.06411) | soft latent sampler + GRPO-style surrogate；官方代码含 fixed top-k、clipped Gumbel 与 mixture replay | 必须比较的同任务近邻 |
| [Soft Concept Mixing](https://arxiv.org/abs/2511.16885) | 训练时把 probability-weighted soft concept vector 混入 hidden states，再用 RL 优化 | PCMC 最危险的直接碰撞；仅“训练模型适应 mixture”不够新 |
| [Latent Thinking Optimization](https://arxiv.org/abs/2509.26314) | 用 latent classifier 作为 latent reward model，在 test time 优化 latent thoughts | generic latent reward/steering 已被覆盖 |
| [LEPO](https://arxiv.org/abs/2604.17892) | filtered logits 上 Gumbel-softmax latent expectation，优势加权 soft cross-entropy | 提供较适合 pathwise OMPI 的 host，但其 released objective 不是 response marginal |
| [Latent-GRPO](https://arxiv.org/abs/2604.27998) | vocabulary-space latent sampler + selected-Gumbel surrogate + group-relative reward | 本项目对象；sampler 与 optimizer contract 已在第 2 节重建 |
| [NF-CoT](https://arxiv.org/abs/2606.06447) | 用 normalizing-flow/stochastic latent transition 表达多模态 continuous thoughts | 若需要 clean reparameterizable host，属于强基线/实现候选 |
| [SWITCH](https://arxiv.org/abs/2606.13106) | hidden recurrence、latent/visible switch、可变 latent 计算；released final path 默认关闭部分随机机制 | 终止/控制架构近邻；不能笼统称为 Gaussian latent policy |
| [Latent Thought Flow](https://arxiv.org/abs/2606.16222) | variable-length continuous trajectories；continuous GFlowNet 匹配由答案质量与计算成本定义的 reward-induced posterior | OMPI 直接碰撞：它优化 latent trajectory posterior，而 OMPI 必须证明“先 quotient 掉 private path”有额外价值 |
| [RLTT: Rewarding Latent Thought Trajectories](https://arxiv.org/abs/2602.10520) | 在 looped LMs 中把 reward 分配到完整 latent thought trajectory，提供 process-level credit | generic “给 latent trajectory 稠密信用”已被直接覆盖 |
| [Dynamics Within Latent CoT](https://arxiv.org/abs/2602.08783) | 分析 latent CoT 内部动力学与信息演化 | causal/dynamics analysis 的关键背景，不直接提供优化方法 |
| [SCOLAR](https://arxiv.org/abs/2605.12163) | 用 information-gain 视角研究 latent reasoning collapse/利用不足 | information reward 或 representation-collapse 叙事已有直接邻居 |

### 结论

**[VERIFIED—LITERATURE]** “continuous thought”“soft token”“vocabulary superposition”“latent stochastic trajectory”“latent reward posterior”均已是拥挤方向。新方法不能把“在 latent space 做 RL”当贡献。

## 3.2 RLVR、GRPO、RLOO、DAPO 与推理策略优化

| 工作 | 核心优化单位/机制 | 对候选的约束 |
|---|---|---|
| [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300) | 同 prompt group 内相对 reward，无独立 critic | 基础 baseline |
| [Back to Basics / RLOO](https://arxiv.org/abs/2402.14740) | sequence-level REINFORCE leave-one-out baseline | group-relative estimator 的最简单强基线 |
| [DAPO](https://arxiv.org/abs/2503.14476) | clip-higher、dynamic sampling、token-level loss、overlong shaping | 仅调 clip/filter/length 不是新科学 |
| [Understanding R1-Zero / Dr.GRPO](https://arxiv.org/abs/2503.20783) | 指出 GRPO normalization/length 等偏差并给简化修正 | generic estimator cleanup 已拥挤 |
| [GSPO](https://arxiv.org/abs/2507.18071) | sequence-level importance ratio 与 clipping | sequence-level/off-policy 必须纳入 baseline |
| [S-GRPO](https://arxiv.org/abs/2508.05928) | 针对 think–answer mismatch 与 reward noise 推导 noise-aware advantage reweighting | generic advantage weighting / reward-noise robustness 已有直接近邻 |
| [Target Policy Optimization](https://arxiv.org/abs/2604.06159) | 对 sampled completions 构造 \(q_i\propto p_i^{old}\exp(u_i/\eta)\)，在 candidate simplex 上做 cross-entropy fitting | OMPI 的最大碰撞；reward-tilted target 本身不能算新贡献 |
| [A Step Back / MinPRO](https://arxiv.org/abs/2601.22718) | 指出 token ratio 的 off-policy问题，提出 prefix-ratio 稳定 surrogate | 不能把 prefix/off-policy correction 当 latent-specific 主线 |
| [VESPO](https://arxiv.org/abs/2602.10693) | sequence-level policy optimization/variance-efficiency路线 | 任何“改 sequence surrogate”都需实质相减 |
| [iGRPO](https://arxiv.org/abs/2602.09000) | 两阶段 best-draft self-conditioning：先探索并选最高奖励 draft，再对 draft-conditioned refinement 做 GRPO | generic self-refinement / extra-context wrapper 不是 latent-specific 新空间 |

### TPO 的方法级碰撞

TPO 不是“优势乘 log-prob”的简单改写。它在 sampled candidate set 上定义：

\[
p_i^\theta=
\frac{\exp(\log\pi_\theta(y_i\mid x))}
{\sum_j\exp(\log\pi_\theta(y_j\mid x))},
\quad
q_i\propto p_i^{old}\exp(u_i/\eta),
\]

并最小化：

\[
-\sum_iq_i\log p_i^\theta.
\]

梯度对 candidate logit 为 \(p^\theta-q\)，在命中 target 时消失。  
因此 OMPI 若只把 \(\log\pi_\theta(y_i)\) 换成 `logsumexp`，Reviewer 2 会合理地说“这是 TPO 的 latent-variable implementation”。

## 3.3 Credit assignment、causal intervention 与 group-relative estimator

| 工作 | 方法对象 | 碰撞/约束 |
|---|---|---|
| [RUDDER](https://arxiv.org/abs/1806.07857) | return decomposition / reward redistribution | telescoping latent reward 不是新 |
| [VinePPO](https://arxiv.org/abs/2410.01679) | 从 prefix 重新 rollout，估计更细粒度 value/advantage | step counterfactual rollout 的强基线 |
| [Rewarding Progress / PAV](https://arxiv.org/abs/2410.08146) | 用 progress verifier 给过程差分奖励 | information/progress reward 已被覆盖 |
| [PRIME](https://arxiv.org/abs/2502.01456) | process-reward model / implicit process signal | 仅增加 PRM 不足以成新方法 |
| [CVT-RL](https://arxiv.org/abs/2606.05263) | counterfactual value/trajectory credit | latent intervention credit 的直接碰撞 |
| [IBPO](https://arxiv.org/abs/2605.16302) | intervention-based policy optimization | 任何 latent intervention 主张都必须比较 |
| [BiPACE](https://arxiv.org/abs/2606.25556) | branch/prefix-aware credit assignment 与高成本 trajectory evaluation | 多分支 credit 的强基线及成本警告 |
| [CRAFT](https://arxiv.org/abs/2606.29476) | counterfactual reasoning/trajectory-level attribution | generic counterfactual attribution 空间拥挤 |
| [InT](https://arxiv.org/abs/2601.14209) | reasoning trajectory intervention/training | “找关键 latent step 再训练”高度碰撞 |
| [GPO: Learning from Critical Steps](https://openreview.net/forum?id=c6RDAutyNE) | 找 pivotal step，reset 并重采样 | critical-step 方法不是空白 |

### group-relative 关键事实

**[VERIFIED—MATH]** 对独立 rollout，leave-one-out baseline 可保持 score term 的期望结构；对相关 rollout，别的 trajectory reward 与本 trajectory score 不再独立。保持每条 rollout marginal 正确，不足以保持 GRPO/RLOO estimator 正确。这正是仓库 antithetic gate 的永久否决点。

## 3.4 Stochastic computation graphs、hybrid policies 与 gradient bias

| 工作 | 关键对象 | 对本项目的含义 |
|---|---|---|
| [Stochastic Computation Graphs](https://arxiv.org/abs/1506.05254) | score-function 与 pathwise gradient 的统一图表示 | 所有 latent-gradient claim 的基础 |
| [REBAR](https://arxiv.org/abs/1703.07370) | discrete relaxation + control variate | 仅用 Gumbel relaxation 降方差不是新 |
| [DiCE](https://arxiv.org/abs/1802.05098) | 高阶正确的 SCG objective | ST estimator 必须与其区分 |
| [CAPG](https://proceedings.mlr.press/v80/fujita18a.html) | fixed clipped continuous action 的 marginal score | 不覆盖 policy-dependent moving atoms |
| [IWAE](https://arxiv.org/abs/1509.00519) | 多样本 evidence lower bound、importance responsibilities | OMPI `log-mean-exp` 的理论祖先 |
| [VIMCO](https://arxiv.org/abs/1602.06725) | 多样本离散 latent 的 leave-one-out control variate | 多路径 credit 的直接祖先 |
| [DReG](https://arxiv.org/abs/1810.04152) | importance-weighted objective 的 doubly reparameterized gradient | pathwise responsibility 不能忽略它 |
| [Reweighted Wake-Sleep](https://arxiv.org/abs/1406.2751) | latent posterior/importance-weighted learning | EM/wake-sleep 版候选高度碰撞 |
| [Hybrid Policy Optimization](https://arxiv.org/abs/2605.14297) | hybrid discrete-continuous action；在 smooth region 用 pathwise、离散分支用 score gradient，并显式处理 cross term | “score + pathwise 混合梯度”本身已被直接覆盖 |
| [Reparameterized Policy Learning for Multimodal Trajectory Optimization](https://arxiv.org/abs/2307.10710) | latent-variable multimodal trajectory policy + variational bound + differentiable world model | latent-variable trajectory optimization 的强祖先 |
| [Beyond Verifiable Rewards / JEPO（NeurIPS 2025；arXiv v1 标题为 Learning to CoT with Jensen’s ELBO）](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6bd67a424dc59481e1e5a5061ffc8dfe-Abstract-Conference.html) | 把 CoT 当 latent variable，以 Jensen evidence lower bound 做 verifier-free / unverifiable-data policy optimization | OMPI 的 latent-marginal claim 必须实质超出此工作 |

### 结论

**[COLLISION RISK]** OMPI 的数学组件没有任何一个单独是新的：reward tilt 来自 REPS/MPO/TPO；latent marginal 来自 VI/IWAE；responsibility gradient 来自 mixture likelihood。可能的新贡献只能是：

1. 明确指出 latent RLVR 的正确 policy-improvement 对象是 verifier-observable marginal；
2. 给出对 private latent path 的 quotient/refinement contract；
3. 证明并观测到 cross-path responsibility 是现有 path-local update 缺失的实质机制；
4. 在同 sampler、同 rollouts、同计算量下产生训练收益。

缺一不可。

## 3.5 Hybrid discrete–continuous policy 与 surrogate objective

官方 Latent-GRPO 的 action 不是标准 hybrid tuple \((d,c)\)，而是：

- dynamic support selection；
- clipped noise atoms；
- ordered Top-K；
- simplex interior weights；
- continuous embedding pushforward；
- discrete proxy/control。

经典 hybrid RL 通常假定可写出 \(p_\theta(d)p_\theta(c\mid d)\) 或有 differentiable simulator。这里 selection boundaries、moving atoms 和长 autoregressive continuation 共同破坏该简单分解。

因此：

- HPO 是 mixed-gradient 近邻，但不能直接解决 official sampler；
- CAPG 只处理固定 clipping；
- Exact Top-K Concrete 只覆盖 clean law；
- score-squashed Gumbel 改变 rollout distribution；
- **OMPI 的吸引力恰在于它只要求 sampler 能产生/回放 latent path，而不要求其 density。**

## 3.6 Latent-state intervention、causal attribution 与 representation collapse

必要背景：

- [Do LLMs Latently Perform Multi-hop Reasoning?](https://arxiv.org/abs/2402.16837)；
- [Divergent Representations from Causal Interventions](https://arxiv.org/abs/2511.04638)；
- [When Chain-of-Thought Fails, the Solution Hides in the Hidden States](https://arxiv.org/abs/2604.23351)；
- [Reasoning Beyond Chain-of-Thought](https://arxiv.org/abs/2601.08058)；
- [Dynamics Within Latent CoT](https://arxiv.org/abs/2602.08783)。

这些工作共同提示：hidden-state patching/steering 可以有因果效应，但 interventions 常落到模型训练分布之外，且“能改变输出”不等于“给出了正确信用分配”。仓库现有 SVCCO/BCG 失败进一步要求：新方法不能把 arbitrary zero/noise/interpolation 当自然 action。

## 3.7 Representation mixture、non-closure 与 probabilistic fusion

| 工作 | 实际机制 | 与 PCMC 的边界 |
|---|---|---|
| [Manifold Mixup](https://arxiv.org/abs/1806.05236) | 中间表示凸组合 + label mixup | generic hidden mixup 祖先 |
| [Soft Thinking](https://arxiv.org/abs/2505.15778) | arithmetic embedding expectation 直接递归 | 不定义 branch-mixture continuation target |
| [MoI](https://arxiv.org/abs/2505.14827) | sampled token 与 prior distribution 的 Bayesian-style input blend | 不保证 behavioral closure |
| [Soft Concept Mixing](https://arxiv.org/abs/2511.16885) | 训练暴露 soft concept vector，再用 RL | 最危险碰撞；PCMC 必须证明 target law 和单-forward barycenter 是新增 |
| [Mixup Regularization: A Probabilistic Perspective](https://proceedings.mlr.press/v286/el-laham25a.html) | 对 conditional density 用 log-linear pooling，并允许任意中间层 probabilistic fusion | 概率 mixture target 不是空白 |
| [Jensen gap bounds](https://arxiv.org/abs/1712.05267) | 非线性下 expectation 与 function-of-expectation 差异的通用分析 | PCMC 不能只“发现 Jensen gap” |
| [Mixup calibration](https://proceedings.mlr.press/v162/zhang22f.html) | mixup 对 calibration/uncertainty 的影响 | 需要报告 calibration，不只 accuracy |

PCMC 的剩余可能新意是：**针对 policy-induced token-support distribution，学习一个可在一次 autoregressive forward 中执行的 latent behavioral barycenter，并给出多步 continuation/reward error bound。**

## 3.8 Verifier、process reward 与 test-time search

必要近邻：

- [LLaMA-Berry](https://arxiv.org/abs/2410.02884)：process verifier + MCTS；
- [rStar-Math](https://arxiv.org/abs/2501.04519)：深度搜索与过程偏好；
- [Rewarding Progress](https://arxiv.org/abs/2410.08146)：progress advantage；
- [Variational Best-of-N Alignment](https://arxiv.org/abs/2407.06057)：把 BoN 分布蒸馏/变分化；
- [Soft Best-of-N](https://arxiv.org/abs/2505.03156)：soft selection/BoN 训练；
- [Latent Chain-of-Thought for Visual Reasoning](https://arxiv.org/abs/2510.23925)：posterior inference、diversity-seeking RL/GFlowNet、marginal-likelihood inference scaling；
- [Latent Thought Flow](https://arxiv.org/abs/2606.16222)：reward-induced continuous trajectory posterior。

因此，tree search、Best-of-N、process verifier、reward posterior sampling 本身都不是本项目可主张的新空间。

## 3.9 Off-policy、counterfactual 与 sequence/trajectory optimization

- RLOO、GSPO、MinPRO 已覆盖 sequence/prefix importance 的大量设计空间；
- VinePPO、GPO、CVT-RL、IBPO、BiPACE、CRAFT 已覆盖 prefix restart、branch credit、counterfactual rollout；
- TPO 已覆盖 finite candidate target fitting；
- LaCoT、JEPO、IWAE 覆盖 latent marginal/inference；
- LTF/GFlowNet 覆盖 reward-proportional trajectory distribution。

**[INFERENCE]** 仍有空隙的不是“再设计一个 trajectory surrogate”，而是 private latent computation 与 verifier-observable policy 之间的 **quotient interface**，以及这个 interface 是否能在有限组 rollout 中产生非退化责任分配。

---

## 4. 直接碰撞相减：两个最终候选的最近工作

## 4.1 OMPI 的 5 个最近邻

### 1. Target Policy Optimization

**相同：**

- finite candidate simplex；
- \(p^{old}\exp(R/\beta)\) reward tilt；
- cross-entropy target fitting；
- fixed-point gradient \(p-q\)。

**实质差异：**

- TPO 的 candidate logit 是一条 completion 的直接 \(\log\pi_\theta(y_i\mid x)\)；
- OMPI 的 candidate logit 是对多条 private latent paths 的
  \[
  m_j^\theta=\log \frac1K\sum_i p_\theta(y_j\mid x,z_i),
  \]
  并产生 \(\rho_{ij}\) 跨路径 responsibility；
- OMPI 还要求对相同可见 response 去重/聚合，避免把 private path identity 当 verifier action。

**[COLLISION RISK]** 若 \(\rho_{ij}\) 基本为 one-hot，OMPI 就退化为 TPO/diagonal TPO，应立即 KILL。

### 2. IWAE / VIMCO / DReG

**相同：**

- 多 latent samples；
- `log-mean-exp` evidence；
- posterior-like responsibilities；
- pathwise/score gradient variance 问题。

**实质差异：**

- 它们解决 latent-variable likelihood/inference；
- OMPI 把该 marginal score 嵌入 verifier-reward 的 policy-improvement operator；
- OMPI 的核心实证对象不是 bound tighter，而是“跨 naturally sampled reasoning paths 的 outcome credit 是否非退化”。

**[COLLISION RISK]** 若论文只给 IWAE bound 和标准 responsibility gradient，没有 latent-RLVR 特有定理/现象，则不够 AAAI method novelty。

### 3. Beyond Verifiable Rewards / JEPO（Jensen ELBO）

**相同：**

- reasoning trace 是 latent variable；
- 优化 answer evidence 而非直接监督每条 trace；
- 不必有外部 process reward。

**实质差异：**

- JEPO 以 CoT latent-variable evidence、verifier-free / unverifiable-data policy optimization 为主；
- OMPI 使用 verifier reward 构造 policy-improvement target，并研究 private continuous path 的 cross-credit；
- OMPI 必须在 online RLVR、group rollouts、matched compute 下证明优势。

### 4. Latent Chain-of-Thought for Visual Reasoning

**相同：**

- reasoning 作为 posterior inference；
- latent rationales 被边际化/用于 marginal likelihood；
- diversity-seeking/GFlowNet。

**实质差异：**

- LaCoT 的 latent rationales 是显式可生成、可排名的 visual reasoning objects，并训练 amortized posterior；
- OMPI 的 latent paths 是模型内部 private computation，最终从 policy action space quotient 掉；
- OMPI 不要求学习一个可解释 rationale posterior，而要求对 response target 的跨路径 likelihood responsibility。

### 5. Latent Thought Flow

**相同：**

- stochastic continuous latent trajectories；
- reward 与 computation cost；
- 多模态 latent reasoning。

**实质差异：**

- LTF 直接让 sampler 匹配 reward-induced **trajectory posterior**；
- OMPI 主张 reward-equivalent private trajectories 不应首先成为优化 target，而应被积分掉；
- LTF 追求多样的高 reward trajectories，OMPI 追求正确的 observable response marginal。

**[COLLISION RISK]** 如果 empirical result 表明 path diversity 本身是关键，而 cross-path marginalization无增益，LTF 会是更自然、更完整的方法。

## 4.2 PCMC 的 5 个最近邻

### 1. Soft Thinking

Soft Thinking 直接把 probability-weighted token embeddings 作为下一输入。PCMC 不把这个 arithmetic vector 当天然有“多个 concept”的语义，而显式要求其 continuation distribution 接近 component continuation mixture。

### 2. Soft Concept Mixing

SCM 在训练中暴露 soft vector 并用 RL 优化。PCMC 只有在以下两点同时成立时才有独立性：

- target 是可测的 branch-mixture continuation law，而不是 generic performance reward；
- 学到的 barycenter 在单次 forward 中近似 K-branch teacher，且有多步误差界。

否则 PCMC 就是 “SCM + consistency loss”。

### 3. Latent-GRPO / LEPO / SofT-GRPO

这些方法改变 soft sampler 和 training surrogate，但都没有强制：

\[
P(\cdot\mid \text{soft action})
\approx
\sum_iq_iP(\cdot\mid \text{hard component }i).
\]

PCMC 改的是 latent transition contract，不是 likelihood。

### 4. Probabilistic Mixup

该工作已经把 conditional density fusion 和 intermediate-layer probabilistic mixup 形式化。PCMC 的独立点必须是 autoregressive latent reasoning、policy-induced support、single-forward amortization 和 sequence/reward guarantee。

### 5. Manifold Mixup

Manifold Mixup 是 representation regularization；PCMC 是 behavioral kernel matching。若 PCMC 只使用 embedding MSE 或 hidden interpolation，就退化为 generic mixup，应 KILL。



---

## 5. 第三步：候选生成、硬淘汰与最终保留

共生成 17 个候选。评分前先做三层硬筛：

1. **contract filter**：是否依赖不存在的 latent-action likelihood；
2. **collision filter**：是否只是已有通用方法迁移；
3. **mechanism filter**：是否有低成本、失败即停的 phenomenon gate。

## 5.1 候选池

| # | 候选 | 一句话核心 | 最强吸引力 | Reviewer 2 / 碰撞 | 第一个否决 gate | 决策 |
|---:|---|---|---|---|---|---|
| 1 | **OMPI：可观测边际策略改进** | marginalize private latent paths，再对 distinct responses 做 reward-tilted target fitting | 不用 latent density；改变优化单位；可产生 cross-path credit | **TPO + IWAE/LaCoT** 组合风险极高 | high-reward response 的 cross-path responsibility ESS 是否显著 \(>1\) | **保留：首选 gate** |
| 2 | **PCMC：策略条件混合闭包** | 学一个单-forward behavioral barycenter，使 soft action continuation 匹配 hard-branch mixture | 攻击 soft token 的基础语义；不碰 likelihood | Soft Concept Mixing + probabilistic mixup + distillation | non-closure 是否因果伤害 reward；oracle barycenter 是否存在 | **保留：备选 gate** |
| 3 | mixed SCG latent gradient | smooth weights 用 pathwise，support/termination 用 score | 理论对象清楚 | HPO、SCG、REBAR、DiCE；official sampler 边界复杂 | 无需实验：若必须估 selection boundary/长后缀 score 就不可承受 | **KILL** |
| 4 | outcome-conditioned wake-sleep / EM | 用 reward posterior 选 latent paths，再 wake/sleep 更新 | 可聚合 latent posterior | LaCoT、LTF、RWS、IWAE 已覆盖 | 若去掉 trajectory posterior 后无独立贡献 | **KILL / 吸收到 OMPI** |
| 5 | latent information-gain process reward | 用 answer/verifier uncertainty 的下降给 latent step credit | 稠密信号 | Rewarding Progress、PRIME、RLTT、RUDDER、SCOLAR | simple potential-based shaping 是否同效 | **KILL** |
| 6 | randomized-telescoping counterfactual credit | 随机抽一个 latent step 做自然 branch continuation，构造无偏差分 | 理论可证伪 | VinePPO、CVT-RL、IBPO、BiPACE；rollout 昂贵 | 每 prompt 额外 rollout 超 1× 或 variance 不降 2× | **KILL** |
| 7 | hazard-factored latent termination | 把 exit 从 proxy token 拆成 Bernoulli hazard，content mixture 条件化于 continue | 修复 control/execution 语义 | ACT/PonderNet、SWITCH、TARPO；可能只是 length control | proxy/end-mass disagreement 是否导致 ≥2pp accuracy/compute Pareto 损失 | **KILL 当前 headline；可作架构 ablation** |
| 8 | set-conditioned latent action encoder | 用 DeepSets/Set Transformer 编码 \(\{(s_i,q_i)\}\)，替代算术 embedding average | 保存 support/weight 高阶信息 | generic set encoder；额外参数；可能只是更大模型 | 同参数 MLP/adapter 能否匹配 | **KILL / 可并入 PCMC adapter ablation** |
| 9 | cross-fitted group-relative estimator | 用 cross-fitting/U-statistic 去除 group baseline bias | estimator 清楚 | RLOO、Dr.GRPO、BiPACE、generic statistics；不 latent-specific | 离开 correlated sampling 后是否仍有独立问题 | **KILL** |
| 10 | behavioral bisimulation quotient clustering | 按 continuation law 聚类 latent states，再在 quotient 上优化 | 与 observable equivalence 一致 | 仓库 BCG/FCTR 负结果；bisimulation literature；估计昂贵 | 简单 next-token/hidden baseline 是否已等效 | **KILL** |
| 11 | discrete macro-action codebook | 把多步 latent trajectory 压成离散 option/code，再做 semi-Markov RL | 可获得合法 likelihood 与可解释 action | NF-CoT、SWITCH、options、VQ；改变任务并需预训练 | codebook 是否优于普通 discrete CoT/extra tokens | **KILL / 长期另项目** |
| 12 | verifier-guided latent tree search + distillation | latent branch MCTS/BoN 找优路径再蒸馏 | 有性能上限 | LLaMA-Berry、rStar-Math、LaCoT、LTF、vBoN；更多 compute/supervision | compute-matched BoN 是否已相同 | **KILL** |
| 13 | causal bottleneck / intervention-invariant latent state | 学对 nuisance intervention 不变、对答案因果充分的 latent | 理论漂亮 | intervention OOD；causal labels 不可得；hidden patching 拥挤 | synthetic intervention 之外能否自然识别 | **KILL** |
| 14 | worst-seed / CVaR latent policy | 优化 latent noise 下的低分位 return | 抑制坏 latent paths | generic robust RL；可能牺牲探索与 pass@k | simple temperature/top-p 是否同效 | **KILL** |
| 15 | noise-aware group reweighting | 按 latent entropy/uncertainty 重权重 group advantages | 易实现 | S-GRPO、DAPO、Dr.GRPO；用户明确排除 generic weighting | strongest generic weighting baseline | **KILL** |
| 16 | consensus-triggered soft-to-hard commit | mixture 达到一致性后切到 hard branch | 可减少 non-closure 与长度 | top-1/hard sampling、TARPO、SWITCH；阈值技巧 | always-hard / top-1 是否相同 | **KILL；PCMC 的 baseline** |
| 17 | corrected antithetic/RQMC group sampler | 设计 joint correction 恢复 unbiased group gradient | 可能兼顾覆盖与偏差 | 仓库 exact gate 已明确；Arithmetic/RQMC 直接迁移 | correction 后 variance/compute 是否仍优 | **永久 KILL** |

## 5.2 为什么不保留第三个候选

可以勉强保留 “hazard-factored termination” 或 “set-conditioned action encoder”，但这会降低研究标准：

- termination 方向很可能被评价为 ACT/SWITCH 的 latent 版本，性能提升也容易归因于可变 compute；
- set encoder 很可能被评价为 architecture capacity/DeepSets；
- 两者都没有 OMPI 的优化对象变化，也没有 PCMC 的明确 behavioral target。

因此最终只保留两个；**不为了满足“最多 3 个”而凑第三个。**

## 5.3 淘汰原则回查

所有被保留方向均满足：

- 不依赖 official executed-action likelihood；
- 不需要复活 exact-density substitution；
- 不通过 correlated group sampling 获益；
- 不以 horizon gap、coordinate geometry 或 clipping correction 为核心；
- 第一阶段都能在不训练或极少训练下触发明确 KILL；
- 成功后才可能构成 method paper。

---

## 6. 最终候选一：Observable-Marginal Policy Improvement（OMPI）

## 6.1 精确问题定义

给定 prompt \(x\)，模型先产生 private latent path \(z\)，再生成 visible response \(y\)：

\[
z\sim \mu_{\theta}(z\mid x),\qquad
y\sim p_{\theta}(y\mid x,z),\qquad
R=R(x,y).
\]

关键是 verifier 不观察 \(z\)。可观测 response policy 为

\[
\bar\pi_\theta(y\mid x)
=
\int p_\theta(y\mid x,z)\,\mu_\theta(dz\mid x).
\]

official sampler 的 \(\mu_\theta\) 可以是 mixed/discontinuous measure；OMPI 不要求写出 \(d\mu_\theta/d\mu_{\theta_0}\)。

在一次 rollout/update iteration 中，先冻结 behavior latent proposal \(\mu_0=\mu_{\theta_0}\)，定义 replay-marginal：

\[
\bar\pi_{\theta}^{\mu_0}(y\mid x)
=
\mathbb E_{z\sim\mu_0(\cdot\mid x)}
[p_\theta(y\mid x,z)].
\]

这本身是合法的 visible response distribution，且只需要：

1. 从 \(\mu_0\) 采样 path；
2. 回放 path；
3. 计算普通 visible token likelihood。

## 6.2 有限组方法

对每个 prompt：

- 采样 \(K\) 条 natural latent paths \(z_1,\dots,z_K\)；
- 采样并去重 \(M\) 个 visible response candidates \(y_1,\dots,y_M\)；
- 计算 all-pairs teacher-forced matrix：
  \[
  \ell_{ij}^{\theta}
  =
  \log p_\theta(y_j\mid x,z_i)
  =
  \sum_{t=1}^{|y_j|}
  \log p_\theta(y_{j,t}\mid x,z_i,y_{j,<t});
  \]
- response marginal candidate logit：
  \[
  m_j^\theta
  =
  \log\left(\frac1K\sum_{i=1}^{K}\exp \ell_{ij}^{\theta}\right);
  \]
- 在 distinct candidate set 上归一化：
  \[
  p_j^\theta
  =
  \frac{\exp m_j^\theta}{\sum_{r=1}^{M}\exp m_r^\theta};
  \]
- 构造冻结 target：
  \[
  q_j
  =
  \frac{p_j^{0}\exp(\tilde R_j/\beta)}
  {\sum_r p_r^{0}\exp(\tilde R_r/\beta)};
  \]
- 最小化：
  \[
  \mathcal L_{\mathrm{OMPI}}
  =
  -\sum_{j=1}^{M}q_j\log p_j^\theta.
  \]

其中 \(\tilde R_j\) 是 within-prompt standardized reward，可加冻结的 length/format penalty，但不能后验调到有利于 OMPI。

### 跨路径 responsibility

\[
\rho_{ij}^\theta
=
\frac{\exp \ell_{ij}^{\theta}}
{\sum_{r=1}^{K}\exp \ell_{rj}^{\theta}}.
\]

于是：

\[
\frac{\partial\mathcal L}{\partial \ell_{ij}}
=
(p_j^\theta-q_j)\rho_{ij}^\theta.
\]

这意味着高 reward response 的信用不只给生成它的 diagonal path \(z_j\)，而是给所有能解释该 response 的 natural paths，权重由 ordinary visible likelihood 决定。

## 6.3 两个实现层级

### OMPI-R：source-compatible replay 版

- latent paths 来自 official sampler；
- stored mixture inputs 固定；
- 只对 \(p_\theta(y\mid z_i)\) 求梯度；
- 不写、不估 latent density；
- 每批 rollout 后刷新 proposal。

这是最适合先做 checkpoint gate 的版本，但它对 latent generator 的更新是共享参数下的间接更新，理论上限较低。

### OMPI-P：pathwise 版

使用 reparameterizable latent transition：

\[
z_i=F_\theta(x,\epsilon_i),\qquad \epsilon_i\sim p_0,
\]

把 \(\nabla_\theta\ell_{ij}\) 通过 \(F_\theta\) 反传。可在 full-support Gumbel-softmax、NF-CoT 或 policy-independent fixed support 上实现。

**[COLLISION RISK]** 这会改变 sampler。公平实验必须做 objective × sampler 的 2×2 分解，不能把 sampler 改进算到 OMPI objective 上。

## 6.4 最小理论包

### 定理 A：reward-tilted candidate target

给定 old candidate distribution \(p^0\)，\(q\) 是

\[
\max_{r\in\Delta^{M-1}}
\left[
\sum_j r_j\tilde R_j
-
\beta\,\mathrm{KL}(r\Vert p^0)
\right]
\]

的唯一解。该部分与 REPS/MPO/TPO 同源，**不是新定理**。

### 定理 B：private-path refinement invariance

若 latent path \(z\) 被拆成若干 behaviorally identical 子路径，子路径 proposal mass 之和等于原 mass，且

\[
p_\theta(y\mid x,z^{(a)})=p_\theta(y\mid x,z)
\]

对所有 \(y\) 成立，则 \(\bar\pi_\theta^{\mu_0}\)、candidate target 和 OMPI observable gradient 不变。

新意不在证明难度，而在把它设为 latent RL optimizer 的合同。

### 定理 C：responsibility gradient identity

对多样本 empirical marginal：

\[
\nabla_\theta m_j^\theta
=
\sum_i\rho_{ij}^\theta
\nabla_\theta \ell_{ij}^\theta.
\]

若 \(z_i=F_\theta(\epsilon_i)\)，右端包括 latent transition pathwise derivative；不需要 \(\log\mu_\theta(z)\)。

### 定理 D：拟合误差到 reward 的有限候选界

若 \(\mathrm{KL}(q\Vert p^\theta)\le\varepsilon\)，reward span 为 \(B_R\)，则由 Pinsker：

\[
\mathbb E_{p^\theta}[\tilde R]
\ge
\mathbb E_q[\tilde R]
-
B_R\sqrt{\varepsilon/2}.
\]

该界只对 sampled candidate simplex 成立，不能伪装成全 response space monotonic improvement。

### 可选定理 E：多样本 evidence 下界

对 iid latent samples：

\[
\mathbb E\left[
\log \frac1K\sum_i p_\theta(y\mid z_i)
\right]
\le
\log \bar\pi_\theta(y\mid x),
\]

并在常见条件下随 \(K\) 收紧。该部分来自 IWAE，不应作为主要 novelty。

## 6.5 为什么它可能击败强 baseline

仅在以下机制成立时：

1. 一个高 reward response 在多个 natural latent paths 下有非忽略 likelihood；
2. path-local GRPO/LEPO/TPO 把信用限制在生成该 response 的路径；
3. OMPI responsibility 能把稀疏 terminal reward 传播给多个可解释路径；
4. 该聚合降低 gradient variance 或避免 latent path fragmentation；
5. all-pairs 计算能用 \(M<K\)、top-responsibility truncation 或缓存降低到可接受成本。

**[OPEN QUESTION]** 上述第 1 条最可能失败；因此它必须是第一个 checkpoint phenomenon gate。

---

## 7. 最终候选二：Policy-Conditional Mixture Closure（PCMC）

## 7.1 精确问题定义

soft latent action：

\[
a=(S,q),\qquad
\bar e(a)=\sum_{i\in S}q_iE_i.
\]

算术执行得到：

\[
P_{\mathrm{arith}}^H(\cdot\mid h,a)
=
P_\theta^H(\cdot\mid h,\bar e(a)).
\]

理想的 branch-mixture law 为：

\[
P_{\mathrm{branch}}^H(\cdot\mid h,a)
=
\sum_{i\in S}q_iP_\theta^H(\cdot\mid h,E_i).
\]

closure gap：

\[
C_H(h,a)
=
D_{\mathrm{JS}}
\left(
P_{\mathrm{branch}}^H
\;\Vert\;
P_{\mathrm{arith}}^H
\right).
\]

PCMC 学习一个 contextual barycenter：

\[
b_\phi(h,S,q)
\]

并单次执行

\[
P_{\theta,\phi}^H(\cdot\mid h,b_\phi(h,S,q))
\approx
P_{\mathrm{branch}}^H(\cdot\mid h,a).
\]

## 7.2 方法结构

低成本版本先匹配 one-step categorical law：

\[
\mathcal L_{\mathrm{closure}}
=
D_{\mathrm{KL}}
\left(
\operatorname{stopgrad}\left[
\sum_iq_iP_\theta(\cdot\mid h,E_i)
\right]
\;\Vert\;
P_{\theta,\phi}(\cdot\mid h,b_\phi)
\right).
\]

为避免 generic distillation 解释，必须加入：

- hard-token identity：\(q=e_i\Rightarrow b_\phi=E_i\)；
- permutation invariance；
- discrete behavior preservation；
- multi-step held-out closure；
- inference 时只做一个 barycenter forward；
- 与 always-hard randomized branch 同 compute 比较。

## 7.3 最小理论包

若每一步 conditional transition 的 TV closure error 为 \(\delta_t\)，且后续 kernel contraction coefficient 为 \(\kappa_t\)，则 H-step continuation error 可按 telescoping/Dobrushin 形式界为：

\[
\mathrm{TV}
\left(
P_{\mathrm{branch}}^H,
P_{\mathrm{bary}}^H
\right)
\le
\sum_{t=1}^{H}
\delta_t
\prod_{s=t+1}^{H}\kappa_s.
\]

对 bounded reward \(R\in[0,1]\)：

\[
\left|
\mathbb E_{\mathrm{branch}}R
-
\mathbb E_{\mathrm{bary}}R
\right|
\le
\mathrm{TV}
\left(
P_{\mathrm{branch}}^H,
P_{\mathrm{bary}}^H
\right).
\]

**[COLLISION RISK]** 这是标准 kernel perturbation 思路；新意取决于 policy-induced mixture 与单-forward amortized barycenter，不在不等式本身。

## 7.4 它可能击败 baseline 的唯一理由

- arithmetic mixture 确实把多个有用 branches 非线性地压坏；
- randomized hard branch 在期望上更好但方差/compute 更高；
- barycenter 能用一次 forward 保留 branch-mixture behavior；
- closure regularization不是单纯额外监督，而是修复 sampler 的语义合同。

若 hard top-1 或随机 hard component 已经同样好，则 PCMC 没有存在必要。



---

## 8. 第四步：Reviewer 2 红队

## 8.1 OMPI 红队

| Reviewer 2 问题 | 严格回答 |
|---|---|
| 它是否只是已有工作的组合？ | **很可能。** 最危险的描述是 “TPO logits 换成 IWAE marginal”。只有 observable/private-path quotient contract、非退化 cross-path responsibility、以及 matched-compute training gain 同时成立，才超过组合。 |
| 最简单 baseline 是否足以击败它？ | 很可能。`TPO + exact visible sequence log-prob`、reward-weighted SFT、OMPI \(K=1\)、或多 rollout 的普通 TPO 都可能匹配。它们必须是首批 baseline。 |
| 是否依赖无法观测或无法估计变量？ | exact \(\bar\pi_\theta(y)\) 不可精确计算，但 \(z_i\) 是实际 natural rollout，\(p_\theta(y\mid z_i)\) 是 ordinary teacher-forced likelihood，可 Monte Carlo 估计。proposal-shift gap 仍不可直接消失。 |
| 理论是否只在 toy setting 成立？ | candidate-target 与 responsibility identity 是一般的；monotonic improvement 只在固定 proposal、有限 candidate set 上成立。对 online refreshed latent sampler 的全局保证目前没有，不能夸大。 |
| 是否改变 sampler，造成不公平比较？ | OMPI-R 不改变；OMPI-P 改为可重参数化 sampler。OMPI-P 必须做 objective × sampler 2×2 并与 LEPO/NF-CoT 同 sampler 比。 |
| 提升是否可能只来自更多计算、rollout 或监督？ | 是。all-pairs teacher forcing 是额外计算。必须报告 prompt-matched、rollout-matched、GPU-hour-matched三种比较；不得增加 verifier label。 |
| 是否需要不可承受的反事实 rollout？ | 不需要额外 free-generation counterfactual；但需要 \(K\times M\) teacher-forced cross-scoring。若 top-2/truncated responsibility 不能保持梯度，成本可能不可承受。 |
| 是否存在同样有效但更简单的方法？ | `K=1 TPO`、去重后的 response-level TPO、best-path weighted CE、增加独立 prompts、普通 BoN distillation 都更简单。 |
| 哪个结果立即否决？ | correct responses 的 median responsibility ESS \(<1.20\)，或 80% 以上 max responsibility \(>0.90\)；OMPI 与 diagonal TPO gradient cosine \(>0.98\)；compute-matched local gain不优于 TPO。 |
| AAAI Reviewer 最可能抓住的致命缺陷？ | **“这是 TPO + IWAE，且 latent paths 对完整 response likelihood 几乎互斥，所以 all-pairs responsibility 退化；额外算力解释了全部收益。”** |

### Reviewer 2 暂定裁决

**Weak Reject / Encourage falsification experiment。**  
在 checkpoint gate 前不应给出 Accept 倾向，更不应进入 full training。

## 8.2 PCMC 红队

| Reviewer 2 问题 | 严格回答 |
|---|---|
| 它是否只是已有工作的组合？ | 高风险：Soft Concept Mixing + probabilistic mixup + knowledge distillation。只有 single-forward behavioral barycenter 与多步 closure guarantee 可能形成独立点。 |
| 最简单 baseline 是否足以击败它？ | 很可能。随机采一个 hard component、top-1、普通 MLP adapter、或 SCM-style soft exposure 都可能相同。 |
| 是否依赖无法观测或无法估计变量？ | one-step branch mixture 可用 K 个 teacher forwards 精确估计；完整 H-step mixture 代价指数/线性增长，只能抽样。 |
| 理论是否只在 toy setting 成立？ | kernel perturbation bound 一般成立，但松；“小 one-step KL 导致更好答案”必须实证，不能靠界。 |
| 是否改变 sampler，造成不公平比较？ | 改变 executed latent representation/transition contract。必须保留同 support、weights、rollout count，并与等参数 adapter 比。 |
| 提升是否可能只来自更多计算、rollout 或监督？ | 训练 teacher 需要 K branches，属于额外蒸馏算力；推理必须保持单 forward，且 baseline 要获相同训练计算。 |
| 是否需要不可承受的反事实 rollout？ | one-step K-branch 尚可；多步 exact target 不可承受。若 one-step target不能预测长期 reward，方向死亡。 |
| 是否存在同样有效但更简单的方法？ | hard sampling、top-1、temperature sharpening、SCM、generic consistency loss。 |
| 哪个结果立即否决？ | hard-branch mixture 不优于 arithmetic mix；closure gap 与 reward harm 的 Spearman 95% CI 包含 0.10 以下；oracle latent optimization不能降低 gap 60%；简单 adapter 匹配。 |
| AAAI Reviewer 最可能抓住的致命缺陷？ | **“非线性 mixture 不是 bug；你强迫模型线性化，可能抹掉 soft concept 的真正优势。结果只是额外 distillation/parameters。”** |

### Reviewer 2 暂定裁决

**Reject as current method；保留为机制 gate。**  
只有 causal non-closure 与 oracle barycenter 两项先通过，才重开方法设计。

---

## 9. 第五步：悲观 / 基准 / 乐观评分

评分均为 10 分制。对“实现难度”和“GPU 算力要求”，**10 表示更容易/更低成本**；这两项权重很低，不因易做而抬高排名。

### 9.1 OMPI

| 维度 | 悲观 | 基准 | 乐观 | 解释 |
|---|---:|---:|---:|---|
| 问题重要性 | 8.0 | 9.0 | 9.5 | private computation vs observable reward 很普适 |
| 原始创新性 | 4.0 | 6.5 | 8.0 | 最大风险是 TPO + IWAE |
| 与最新工作的碰撞安全性 | 3.0 | 5.0 | 6.5 | TPO、LaCoT、LTF、JEPO 都很近 |
| 方法论文适配度 | 5.0 | 7.5 | 9.0 | 取决于 cross-path mechanism 是否真实 |
| 理论上限 | 6.0 | 7.5 | 9.0 | quotient + responsibility + finite-candidate bound |
| 实验清晰度 | 6.0 | 8.0 | 9.0 | all-pairs matrix 与强 kill gate 很清楚 |
| 可证伪性 | 9.0 | 9.5 | 10.0 | ESS/gradient/local utility 可直接杀死 |
| 实现难度（高分=易） | 4.0 | 4.5 | 5.0 | replay 与 \(K\times M\) scoring 较重 |
| GPU 要求（高分=低） | 4.0 | 4.5 | 5.5 | 可先 inference gate，训练不便宜 |
| AAAI 主会适配度 | 5.0 | 7.0 | 8.5 | 若有通用 latent-variable RL 叙事则合适 |
| 最佳论文潜力 | 2.5 | 5.0 | 8.0 | 需要漂亮机制图和跨架构结果 |
| 综合期望收益 | **3.5** | **6.3** | **8.2** | 当前只值一次 checkpoint gate |

### 9.2 PCMC

| 维度 | 悲观 | 基准 | 乐观 | 解释 |
|---|---:|---:|---:|---|
| 问题重要性 | 7.0 | 8.0 | 9.0 | soft-token semantics 是整个方向的基础 |
| 原始创新性 | 3.5 | 5.5 | 7.5 | mixup/SCM/distillation 碰撞高 |
| 与最新工作的碰撞安全性 | 3.0 | 4.5 | 6.5 | SCM 和 probabilistic mixup 最危险 |
| 方法论文适配度 | 4.5 | 6.5 | 8.5 | 需从 regularizer 上升为 transition operator |
| 理论上限 | 5.0 | 6.5 | 8.0 | multi-step kernel bound 可做但可能松 |
| 实验清晰度 | 7.0 | 8.5 | 9.0 | closure/hard-mixture/oracle gate 很直接 |
| 可证伪性 | 9.0 | 9.5 | 10.0 | 很容易证明不需要该方法 |
| 实现难度（高分=易） | 4.0 | 4.5 | 5.0 | K-branch teacher 与 adapter 复杂 |
| GPU 要求（高分=低） | 3.5 | 4.0 | 5.0 | teacher 计算可能比 OMPI 更高 |
| AAAI 主会适配度 | 4.0 | 6.0 | 8.0 | 若展示普适 soft-token closure，才够 |
| 最佳论文潜力 | 2.0 | 4.0 | 7.0 | 容易被视为工程/蒸馏 |
| 综合期望收益 | **3.0** | **5.5** | **7.5** | 备选，不先训练 |

### 9.3 情景定义

- **悲观：** cross-path likelihood 接近互斥；non-closure 无 causal harm；最近工作被审稿人视为直接覆盖。
- **基准：** 至少一个 mechanism gate 通过，但效果中等，需要严格 compute matching 才能立住。
- **乐观：** OMPI 在两类 latent architecture 上有高 posterior entropy、明显 gradient advantage，并在 1B/1.5B 三 seed 训练中稳定领先；或 PCMC 证明单-forward barycenter 可替代 K-branch mixture。

## 9.4 最终选择

### 唯一首选：OMPI

理由不是“最容易”，而是它改变了真正的 optimization object，并能避开仓库已经证明无效的 latent-likelihood repair。

### 唯一备选：PCMC

理由是它攻击 sampler 语义本身，且与 OMPI 机制正交；但碰撞与额外计算风险更高。

### 明确不建议继续

除这两个 mechanism gate 外，本报告第 5 节其余 15 个候选全部不建议。  
尤其不建议“先训起来看看”；仓库已有足够多负结果表明，这种策略会把项目拖回 post-hoc narrative。



---

## 10. 第六步：首选 OMPI 的可执行计划

## 10.1 一句话核心 claim

> **在 verifier-only latent reasoning 中，直接优化由多条 private latent paths 共同诱导的 visible-response marginal，并按跨路径 response likelihood responsibility 分配信用，比 path-local GRPO/LEPO/TPO 更有效且不需要 latent-action likelihood。**

该 claim 只有在 cross-path responsibilities 非退化、compute-matched training 胜出时才成立。

## 10.2 精确问题定义

输入 \(x\)，private latent computation \(z\)，visible response \(y\)，terminal verifier \(R(x,y)\)。

目标不是为 \(z\) 构造一个可能不存在的 PPO ratio，而是改进：

\[
\bar\pi_\theta(y\mid x)
=
\mathbb E_{z\sim\mu_\theta(\cdot\mid x)}
[p_\theta(y\mid x,z)].
\]

一次 online iteration 使用旧 proposal \(\mu_0\)：

\[
\bar\pi_\theta^{\mu_0}(y\mid x)
=
\mathbb E_{z\sim\mu_0}
[p_\theta(y\mid x,z)].
\]

**研究假设 H1：** 高 reward response 在多条 natural latent paths 下具有显著 likelihood，而不是只由其生成 path 唯一解释。  
**研究假设 H2：** 这种共享解释质量能通过 responsibility gradient 产生比 diagonal update 更好的局部及训练收益。  
**研究假设 H3：** 该收益在 GPU-hour matching 后仍存在。

H1 失败即停止；不允许跳过 H1 直接训练。

## 10.3 方法结构与必要公式

### Algorithm：OMPI-R

对每个 rollout batch：

1. 冻结 behavior checkpoint \(\theta_0\)；
2. 每个 prompt 采样 \(K\) 条 source-native latent paths；
3. 每条 path 生成一个 response，按 exact response string 去重；
4. 对每个 path–response pair 做 teacher-forced replay，得 \(\ell_{ij}^\theta\)；
5. 计算
   \[
   m_j^\theta=\operatorname{LSE}_i(\ell_{ij}^\theta)-\log K;
   \]
6. 在 candidate set 上计算
   \[
   p^\theta=\operatorname{softmax}(m^\theta);
   \]
7. 从 frozen old marginal 与 verifier reward 构造
   \[
   q=\operatorname{softmax}(m^{0}+\tilde R/\beta);
   \]
8. 最小化
   \[
   -q^\top\log p^\theta;
   \]
9. 可用 top-\(r\) responsibilities 做稀疏近似，但 \(r\) 在 gate 前冻结；
10. 每个 rollout batch 后刷新 paths，不跨过大的 policy drift 长期复用。

### 实现细节冻结

- primary likelihood 是完整 visible sequence probability，不做 per-token length normalization；
- length/format 只进入预先冻结的 reward；
- exact duplicate response strings 合并，其 behavior mass 用 `logsumexp` 聚合；
- canonical final-answer grouping 只作为 secondary analysis，不能替代 sequence marginal；
- all-pairs cross-scoring 不调用 verifier 额外标注；
- numerical computation 全部在 log space；
- old target `stop_gradient`；
- official sampler 的 stored latent mixture replay 与源码完全一致；
- embeddings 保持与 official 配置一致的 frozen 状态。

### OMPI-P

在 clean reparameterizable host 上：

\[
z_i=F_\theta(x,\epsilon_i),\quad \epsilon_i\sim p_0,
\]

并反传：

\[
\nabla_\theta \ell_{ij}
=
\nabla_\theta
\log p_\theta(y_j\mid x,F_\theta(x,\epsilon_i)).
\]

OMPI-P 不是第一版；只有 OMPI-R checkpoint gate 通过后才实现。

## 10.4 与 Latent-GRPO、LEPO、SofT-GRPO、GRPO 的区别

| 方法 | 优化单位 | latent update signal | 是否跨路径解释同一 response | 是否需要 latent density |
|---|---|---|---|---|
| GRPO | sampled visible token trajectory | token/sequence likelihood × group advantage | 否 | 不适用 |
| Latent-GRPO | stored selected IDs/scores 对应的 path-local surrogate | selected-Gumbel mean + ST backward | 否 | 使用 surrogate，非 executed density |
| LEPO | archived soft support 的 advantage-weighted soft CE | support-local soft-label CE | 否 | 否，但仍 path-local |
| SofT-GRPO | stored support/score 的 soft latent surrogate | current-dependent support/mask surrogate | 否 | 不等于 executed density |
| TPO | sampled visible completion candidate simplex | \(p-q\) target fitting | 不建模 private latent alternatives | 需要普通 completion log-prob |
| **OMPI** | distinct visible-response marginal candidate simplex | \((p_j-q_j)\rho_{ij}\) | **是** | **否**；只需 path sample + visible likelihood |

### 与 TPO 的必须说明

OMPI 继承 TPO 的 target-fitting operator；**不把 reward tilt 声称为新贡献**。新增主张仅是：

- private path marginalization；
- cross-path responsibility；
- latent refinement/observable contract；
- 实证上 path-local baseline 未利用的共享解释质量。

### 与 LaCoT / LTF 的必须说明

- LaCoT/LTF 把 latent trajectory/rationale 作为 posterior target 或推理对象；
- OMPI 把 private trajectory 视作 nuisance computation，最终优化 response marginal；
- 若实验显示 trajectory diversity 本身比 marginalization 更重要，应承认 LTF 方向更合理。

## 10.5 最小可行理论结果

进入训练前必须完成以下四项：

1. **有限候选 reward-tilted target proposition**：明确标注与 TPO/REPS 同源；
2. **latent refinement invariance lemma**：mass-preserving、behaviorally identical path splitting 不改变 observable objective；
3. **cross-path responsibility gradient theorem**；
4. **fixed-proposal candidate-level reward approximation bound**。

可选但不能当 headline：

- IWAE-style bound tightening；
- Rao–Blackwell variance inequality；
- top-\(r\) responsibility truncation error：
  \[
  \left\|
  \nabla m_j-
  \sum_{i\in \mathrm{Top}r}\tilde\rho_{ij}\nabla\ell_{ij}
  \right\|
  \le
  \text{tail-mass}\times \max_i\|\nabla\ell_{ij}\|
  \]
  的简单界。

**[KILL CONDITION]** 若理论贡献在删除 “latent reasoning” 名词后完全等价于标准 TPO+IWAE recipe，且无法提出独立 observable-contract theorem/estimator claim，则在代码前停止。

## 10.6 第一个零 GPU / CPU gate

### 目的

这是按时间顺序的第一个实验，设计目标是**最容易否决方法的数学价值**，不是展示正例。

### 冻结设置

构造 12 个可精确枚举的 latent-variable contextual bandit families：

- 4 个 aliasing families：多个 \(z\) 诱导同一/重叠 outcome；
- 3 个 separated families：每个 \(z\) 几乎唯一决定 outcome；
- 3 个 refinement families：对一个 path 做 2×/4× mass-preserving split；
- 2 个 proposal-shift families：M-step 后 latent proposal 发生可控漂移。

每个 family 50 seeds。比较：

- exact observable gradient；
- path-local REINFORCE/GRPO；
- candidate TPO；
- OMPI \(K=1,2,4,8\)；
- exact Rao–Blackwell oracle；
- diagonal-only OMPI。

### 冻结 PASS 阈值

全部满足才通过：

1. analytic gradient 对 central finite difference：最大相对误差 \(<10^{-5}\)，最大绝对误差 \(<10^{-7}\)；
2. refinement invariance：observable objective/gradient 最大绝对差 \(<10^{-8}\)；
3. aliasing families 中，\(K=4\) OMPI 相对 diagonal/TPO 的 exact-gradient MSE 中位数至少降低 **30%**，且至少 9/12 relevant settings 改善；
4. separated null families 中，OMPI 不虚构收益：相对 K=1 的 gradient direction median cosine \(>0.99\)，utility 差绝对值 \(<1\%\)；
5. proposal-shift bound 的数值复算无违例；
6. 所有结果、seed、阈值和代码哈希在运行前冻结。

### [KILL CONDITION]

- 任一数学 control 失败；
- aliasing MSE 改善 \(<20\%\)；
- separated null 中出现系统性优势/劣势，说明方法解释不清；
- 只有精心构造的单一 toy 才有收益；
- novelty subtraction 后仅剩 “TPO with log-mean-exp logits”。

### 预计资源

- 8–16 CPU cores；
- < 2 小时；
- < 5 GB RAM；
- < 1 GB artifacts。

## 10.7 第一个 checkpoint inference gate

### 目的

直接测试最脆弱的 H1：**真实 checkpoint 上，正确 response 是否能由多条 natural latent paths 共同解释？**

### 冻结模型

1. `DJCheng/LLaMA3.2-1B-Instruct-Latent-GRPO-Top10`；
2. SofT-GRPO 官方 1.5B trained checkpoint；
3. latent-SFT 1B 可作 pre-RL control，但不用于主要 PASS。

若第二个 checkpoint 的运行环境无法 source-faithful 重放，必须在结果中明确，不可用 base model 代替 trained checkpoint。

### 冻结数据

每个主要 checkpoint 128 prompts：

- 32 GSM8K；
- 32 GSM8K-Hard；
- 32 SVAMP；
- 32 MATH-500。

prompt IDs 在运行前提交 manifest。每 prompt：

- \(K=8\) natural latent paths；
- 每 path 1 个 visible response；
- exact-string dedup，\(M\le 8\)；
- 全 \(K\times M\) teacher-forced cross-score；
- primary reward 为官方 exact verifier；
- primary likelihood 从 visible start 到 EOS 的完整 sequence likelihood。

### Gate 1A：cross-path existence

指标：

\[
\mathrm{ESS}_j
=
\frac{1}{\sum_i\rho_{ij}^2},
\qquad
H_j
=
-\frac{\sum_i\rho_{ij}\log\rho_{ij}}{\log K}.
\]

若同一 exact response 由多条 path 生成，source set 合并；`off-source mass` 是 source set 外的 responsibility mass。

### Gate 1A PASS 阈值

两个主要 checkpoint **分别**满足：

1. 至少 32 个 informative prompts（至少一个 correct、一个 incorrect unique response）；
2. correct candidates 的 median ESS \(\ge 1.50\)；
3. median ESS 的 prompt-bootstrap 95% CI 下界 \(>1.25\)；
4. 至少 25% correct candidates 的 off-source responsibility mass \(\ge0.25\)；
5. normalized posterior entropy median \(H_j\ge0.25\)；
6. 结果在 low-task 与 MATH-500 两个 strata 中方向一致，不允许完全由单一数据集驱动。

### Gate 1A [KILL CONDITION]

任一主要 checkpoint 出现：

- correct median ESS \(<1.20\)；
- 超过 80% correct candidates 的 \(\max_i\rho_{ij}>0.90\)；
- informative prompts \(<24\)；
- 信号只来自 exact duplicate outputs；
- 完整 sequence likelihood 下 collapse，而只有事后 length-normalized score 才通过。

单 checkpoint 通过、另一 checkpoint 失败：**不进入 broad AAAI method training**；最多保留为架构特定观察。

### Gate 1B：local update utility

仅 Gate 1A 通过后运行。每 checkpoint 固定 64 informative prompts，生成两个独立 \(K=8\) path groups：

- group A 计算更新；
- group B 评估 local generalization。

比较：

- OMPI full；
- diagonal TPO；
- ordinary response TPO；
- OMPI \(K=1\)；
- uniform responsibility；
- best-path weighted CE。

所有 update 做相同 parameter-norm trust step，只更新冻结的 rank-8 LoRA（最后 4 blocks + LM head），不做多步训练。

### Gate 1B PASS 阈值

两个主要 checkpoint 分别满足：

1. OMPI 与 strongest simple baseline 的 gradient cosine \(\le0.95\)，证明不是数值同一更新；
2. 在 group B 上，reward-tilted held-out candidate CE gain 比 strongest baseline 高至少 **15%**；
3. 差值的 prompt-bootstrap 95% CI 下界 \(>0\)；
4. correct-candidate marginal probability gain 至少高 **10%**；
5. top-2 responsibility truncation 与 full gradient cosine \(\ge0.90\)；
6. projected training wall-clock overhead \(\le1.5\times\) diagonal TPO。

### Gate 1B [KILL CONDITION]

- gradient cosine \(>0.98\)；
- local utility improvement \(<5\%\)；
- K=1 / diagonal / best-path baseline 在误差范围内相同；
- advantage 只存在于 group A，不泛化到 independent group B；
- top-2 truncation cosine \(<0.80\)，导致训练成本不可压缩；
- 需要 answer labels、PRM 或额外 free-generation 才通过。

### 预计资源

**[INFERENCE]** 基于官方 1B/1.5B 配置，而非实测：

- 每 checkpoint：1×A100 80GB 或 H100 80GB；
- 15–40 GPU-hours；
- 两个 checkpoint 合计 30–80 GPU-hours；
- 250–500 GB 临时存储；
- 不保存全词表 logits，只保存 sequence log-likelihood、responsibility、manifest 和必要 replay tensors。

## 10.8 第一个训练 gate

### 冻结目标

只做 1B/1.5B pilot；不先上 7B。  
每个方法 3 seeds，使用官方 sampler 和相同 rollout count。

### 训练设置

- 起点：官方 1B latent-SFT / latent-GRPO compatible checkpoint；
- train：GSM8K-Aug；
- \(K=8\) rollouts；
- 500 optimizer updates 或 official low-task token budget 的 20%，取较小者；
- OMPI 使用 checkpoint gate 冻结的 top-\(r\)、\(\beta\)、refresh frequency；
- 不允许 pilot 后调 PASS threshold。

### 必须比较的 baseline

1. released Latent-GRPO；
2. same-sampler GRPO/RLOO；
3. same-sampler TPO；
4. OMPI \(K=1\)；
5. diagonal-only OMPI；
6. reward-weighted SFT / weighted CE；
7. compute-matched TPO：把 OMPI 的额外 GPU-hours换成更多 ordinary prompts；
8. 若代码可 source-faithful 运行：SofT-GRPO / LEPO objective。

### 双重匹配

- **rollout-matched：** 相同 prompt 与 rollout；
- **GPU-hour-matched：** baseline 获得与 OMPI 相同总训练 GPU-hours。

prompt-matched 但计算不匹配的结果只能作 secondary。

### 冻结 PASS 阈值

三 seeds 聚合后，全部满足：

1. rollout-matched macro pass@1（GSM8K、GSM8K-Hard、SVAMP、MultiArith）比 strongest baseline 至少 **+2.0 percentage points**；
2. GPU-hour-matched macro pass@1 至少 **+1.5 points**；
3. stratified bootstrap 95% CI 下界 \(>0\)；
4. 任一单任务回退不超过 **0.5 point**；
5. pass@8 回退不超过 **1.0 point**；
6. median response length 与 baseline 差不超过 10%；
7. OMPI full 比 \(K=1\) 和 diagonal-only 分别至少高 **1.0 point**；
8. off-diagonal responsibility mask ablation 至少损失 **1.0 point**；
9. rollout-matched wall-clock不超过 TPO 的 \(1.5\times\)；
10. 无 KL explosion、mode collapse 或仅靠更长输出获得的收益。

### [KILL CONDITION]

任一成立即停止：

- strongest simple baseline 与 OMPI 差 \(<0.5\) point；
- GPU-hour matching 后优势消失；
- 只有 1/3 seed 正；
- 仅 GSM8K 正、OOD 全部无增益；
- full 与 \(K=1\)/diagonal 无差；
- exact-string dedup 去掉后结果反而更好且无法解释；
- training overhead \(>2\times\)；
- 需要额外 verifier/process labels；
- sampler 改变后才有收益，但同 sampler objective ablation 不支持 OMPI。

## 10.9 所有 gate 的冻结决策表

| Gate | PASS | KILL | 允许的下一步 |
|---|---|---|---|
| Gate 0 CPU | math controls 全过；aliasing MSE ≥30% 降低 | 控制失败或只剩 TPO+IWAE 拼装 | 实现 source-faithful cross-score |
| Gate 1A checkpoint | 两 checkpoint correct ESS/entropy/off-source mass 达标 | responsibility collapse / signal 单架构 | local LoRA utility |
| Gate 1B checkpoint | held-out local gain ≥15%，top-2 可近似 | TPO/K=1 同效或成本不可压 | 1B/1.5B 训练 pilot |
| Gate 2 training | rollout 与 GPU-hour matching 均胜，3 seeds，机制 ablation 成立 | 任何核心阈值失败 | 扩第二架构/高难任务 |
| Gate 3 confirmatory | 两架构、两 task families、预注册 held-out 复现 | 只在单模型/单数据集 | 完整 paper |
| 7B scale | 只在小模型完成后 | 不作为救活小模型失败的手段 | 扩展性证据 |

## 10.10 数据集、模型、baseline、ablation

### 数据集

**低成本机制与 pilot：**

- GSM8K / GSM8K-Aug；
- GSM8K-Hard；
- SVAMP；
- MultiArith；
- MATH-500 子集。

**confirmatory：**

- DAPO-Math-17k train；
- MATH-500；
- AIME-2024；
- AIME-2025；
- GPQA（只在 7B、格式与 verifier 可靠时）。

### 模型

- official Latent-GRPO LLaMA3.2-1B；
- official SofT-GRPO 1.5B；
- controlled LEPO implementation from same latent initialization；
- optional official Latent-GRPO Qwen2.5-Math-7B，仅 confirmatory。

### Baseline 分层

**必须：**

- GRPO、RLOO；
- DAPO/Dr.GRPO-style strong recipe；
- Latent-GRPO；
- SofT-GRPO；
- LEPO；
- TPO；
- reward-weighted CE/SFT；
- compute-matched extra rollouts。

**不能当 baseline 缺失借口：**

LEPO 没有公开 trained checkpoint 时，必须从相同 initialization 做小规模 controlled training；不能用 public base model 代替并声称方法比较。

### Ablation

1. \(K=1,2,4,8\)；
2. \(M=2,4,8\)；
3. diagonal only；
4. uniform responsibilities；
5. best-path responsibility；
6. exact-string dedup on/off；
7. top-\(r\), \(r=1,2,4,K\)；
8. frozen replay vs pathwise；
9. proposal refresh frequency 1/2/4 optimizer epochs；
10. target anchor \(p^{old}\) on/off；
11. reward temperature \(\beta\) 的预注册小网格；
12. objective × sampler 2×2；
13. same total prompts / rollouts / GPU-hours；
14. answer-only class grouping仅 secondary，不可替代主结果。

## 10.11 预计 GPU、显存、时间与存储

以下均为 **[INFERENCE]**，根据官方脚本的 8-GPU、`n=8`、1B/7B context 设置和 all-pairs 额外 forward 粗估；必须在 pilot 后用真实 profiler 更新。

| 阶段 | 推荐硬件 | 单次墙钟 | 约 GPU-hours | 存储 |
|---|---|---:|---:|---:|
| CPU exact gate | 8–16 CPU cores | <2 h | 0 | <1 GB |
| 单 checkpoint Gate 1 | 1×A100/H100 80GB | 15–40 h | 15–40 | 100–250 GB |
| 两 checkpoint Gate 1 | 1×80GB 顺序运行 | 30–80 h | 30–80 | 250–500 GB |
| 1B baseline 单 seed pilot | 8×A100/H100 80GB | 24–48 h | 192–384 | 0.5–1 TB |
| 1B OMPI 单 seed pilot | 8×A100/H100 80GB | 36–72 h | 288–576 | 0.8–1.5 TB |
| 5–6 方法 × 3 seeds pilot | 8-GPU jobs | — | 4,000–9,000 | 2–4 TB（滚动清理） |
| 7B 单 seed confirmatory | 8×H100 80GB 或 16×A100 80GB | 72–144 h | 576–2,304 | 2–4 TB |
| 7B 完整方法矩阵 | 多 job | — | 10,000–25,000 | 4–8 TB |

成本压缩优先级：

1. 先用 Gate 1 KILL；
2. top-2 responsibilities；
3. 只存 sequence scores，不存 full logits；
4. candidate dedup；
5. 先 1B/1.5B，不以 7B 搜现象；
6. 只在预注册 pilot 通过后运行 confirmatory。

## 10.12 如果核心假设失败，是否有诚实降级论文

### Gate 1A 失败

**没有 method-paper 降级。**  
结论是完整 response likelihood 使 latent paths 几乎可识别，cross-path marginalization 无实际自由度。可以归档为 negative result，但不应包装为 AAAI 方法。

### Gate 1A 通过、Gate 1B 失败

可形成严格的 analysis：存在 path aliasing，但 responsibility gradient 没有局部效用。仍不是本项目要求的 method paper。

### Gate 1B 通过、training 失败

可写“local observable-marginal advantage 不转化为 online improvement”的负结果，适合作为 workshop、negative-results track 或更长期 journal analysis；不能改名成 benchmark。

### 1B/1.5B 通过、7B 失败

若两个独立 latent architecture、两类任务、三 seeds 均成立，仍可能形成有限规模 method paper；必须诚实限制 scalability claim。若只有一个架构成立，不建议冲 AAAI main。

### 任何失败后的禁止 salvage

- 不改成 entropy bonus；
- 不改成更多 rollouts；
- 不改成 process reward；
- 不放宽 ESS/accuracy thresholds；
- 不把 same data 的新协议称为 confirmatory；
- 不用 7B 搜索来救活 1B gate。

## 10.13 论文标题与摘要式贡献

### 首选标题

**Optimize What the Verifier Sees: Marginal Policy Improvement over Private Latent Reasoning Paths**

备选：

- **Private Thoughts, Public Rewards: Observable-Marginal RL for Latent Reasoning**
- **Beyond Path-Local GRPO: Cross-Path Responsibility for Latent Reasoning Policies**

### 摘要式贡献

1. **Policy-object diagnosis.**  
   证明/展示现有 latent RL 在 continuous execution、proxy control 和 path-local surrogate 之间存在不同 contract；但不把诊断本身当方法贡献。

2. **Observable-marginal policy improvement.**  
   提出 OMPI：对 naturally sampled private paths 计算 visible response marginal，在 finite candidate set 上做 reward-tilted target fitting。

3. **Cross-path credit.**  
   推导 \((p_j-q_j)\rho_{ij}\) responsibility update，不需要 latent-action density，可用于 mixed-measure sampler replay或 clean reparameterizable sampler。

4. **Theory.**  
   给出 private-path refinement invariance、fixed-proposal target 和 finite-candidate error bound；清楚标注 TPO/IWAE 继承部分。

5. **Empirical mechanism.**  
   在至少两种 latent architecture 上证明高 reward responses 有非退化 cross-path posterior，并在 rollout/GPU-hour matching 下优于 TPO、GRPO、Latent-GRPO、LEPO、SofT-GRPO。

### 摘要不能声称

- “首次 marginalize reasoning traces”；
- “首次 reward posterior latent reasoning”；
- “exact policy gradient for official Latent-GRPO”；
- “monotonic improvement in full response space”；
- “free performance without extra compute”。

## 10.14 主图设计

### Figure 1：三联主图

**左：Official contract。**

展示同一 latent step 的三条线：

- executed mixture；
- selected support / proxy token；
- path-local training surrogate。

用箭头标出 reward 只在 visible response 上。

**中：OMPI all-pairs credit matrix。**

矩阵行是 latent paths \(z_i\)，列是 distinct responses \(y_j\)；单元格为 \(\ell_{ij}\)，列 softmax 得 \(\rho_{ij}\)，列 logsumexp 得 observable marginal logit，顶部 reward tilt 得 \(q_j\)。

**右：falsification + gain。**

上半：correct response responsibility ESS 分布，画出 KILL 线 1.20、PASS 线 1.50。  
下半：GPU-hour-matched pass@1 或 held-out target gain，对比 TPO、K=1、diagonal、OMPI full。

### 必须的辅助图

- gradient cosine vs ESS scatter；
- top-\(r\) compute/accuracy Pareto；
- sampler × objective 2×2；
- negative controls；
- training curves带三 seeds，而不是只报 best checkpoint。

---

## 11. 备选 PCMC 的最低成本执行顺序

PCMC 不应与 OMPI 并行消耗大量 GPU。只在 OMPI Gate 1A KILL 或资源独立时运行。

### PCMC Gate A：causal non-closure

每 prompt/path 采 source-native \(S,q\)，比较：

1. arithmetic soft execution；
2. 按 \(q\) randomized hard component 的期望；
3. top-1；
4. temperature-sharpened mixture。

计算 one-step JS closure gap、短 continuation reward 与 final correctness。

**PASS：**

- high-gap quartile 中 randomized-hard 的 expected correctness 比 arithmetic 至少高 3 points；
- closure gap 与 reward loss 的 prompt-level Spearman \(\rho\ge0.25\)，95% CI 下界 \(>0.10\)；
- 结果在两个 checkpoint 同方向。

**KILL：**

- hard mixture 不优；
- closure gap 只相关于 entropy/length，控制后消失；
- top-1 同效；
- 只有人工 high-entropy actions 有现象。

### PCMC Gate B：oracle barycenter existence

对 50–100 个 natural mixtures，直接优化一个自由 latent vector \(u\)，目标匹配 one-step branch-mixture distribution。

**PASS：**

- median KL 比 arithmetic 减少至少 60%；
- median final KL \(\le0.05\) nats/token；
- 优化后的 \(u\) 不要求极大 norm 或离开 natural hidden radius；
- 用同一低秩 adapter family 可 amortize 至少 50% 的 oracle gain。

**KILL：**

- oracle 也无法 closure；
- 只能用 unrestricted per-example high-dimensional optimization；
- closure 后 final reward不改善；
- generic same-parameter MLP 已完全相同。

只有 A+B 同时通过，才设计 PCMC training；否则停止。

---

## 12. 最终研究决策

### 当前授权

- **授权：** OMPI Gate 0；
- Gate 0 通过后，授权 OMPI Gate 1A；
- Gate 1A/1B 均通过后，才授权 1B/1.5B training pilot；
- **不授权：** 当前直接启动 full RL training；
- **不授权：** 重新打开任何仓库已 KILL likelihood/geometry/horizon/exploration方向。

### 当前项目状态的最诚实表述

> **当前没有值得直接训练的 idea；有一个值得用 checkpoint inference 尽快证伪的首选 idea。**

### 若两个最终 gate 都失败，下一轮应该搜索什么

不是继续在 sampler likelihood 上打转，而应搜索：

1. **可观测 objective 与 private computation 的其他可估计接口**：尤其不需要完整 response likelihood 的 outcome-class marginal、amortized response posterior或 verifier-compatible sufficient statistic；
2. **具有可证明因果 harm 的 transition-contract failure**：不是单纯 metric mismatch；
3. **无需额外 rollout 的 natural counterfactual reuse**：例如已有 KV/cache 上的精确 conditional reuse，但必须与 CVT-RL/VinePPO/IBPO 相减；
4. **能在多个 latent architecture 上复现的 shared phenomenon**，而不是为 Latent-GRPO 特定代码制造问题。

若这些搜索也没有非退化现象，应停止 AAAI method-paper 目标，把仓库定位为高质量 negative-results / policy-contract audit 资产。


---

## 13. 文献检索方法、版本边界与不确定性

### 13.1 检索时间与优先级

**检索截止：2026-07-17。** 检索顺序是：

1. 论文原文 HTML/PDF、正式 proceedings；
2. 作者或官方项目页；
3. 官方 GitHub 源码与 released configs/checkpoints；
4. 仅对外围背景工作使用 abstract-level 元数据。

对两个最终候选的核心碰撞，实际检查了方法段而不是只读标题或摘要：

- **TPO**：finite candidate distribution、reward-tilted target、cross-entropy 与 \(p-q\) 梯度；
- **JEPO**：CoT 作为 latent variable、Jensen evidence objective，以及 verifier-free / unverifiable-data policy optimization；
- **LaCoT**：amortized latent-rationale posterior、Reference-Guided GFlowNet 与 marginal-likelihood inference；
- **Latent Thought Flow**：Gaussian reparameterized variable-length latent trajectory、reward-proportional target、continuous SubTB、answer loss和 reference prior；
- **Soft Concept Mixing**：full-distribution weighted embedding、与 contextual hidden state 相加、GRPO 训练；
- **Probabilistic Mixup**：conditional-density fusion，而非仅表示插值。

### 13.2 版本说明

- arXiv `2503.19618` 的早期标题是 *Learning to chain-of-thought with Jensen's evidence lower bound*；正式 NeurIPS 2025 版本标题为 *Beyond Verifiable Rewards: Scaling Reinforcement Learning in Language Models to Unverifiable Data*，方法名 JEPO。本报告按正式版本定位碰撞。
- 2026 年多篇工作仍是近期预印本；其公式、代码、接受状态或后续版本可能继续变化。
- 任何“未发现直接同构方法”的判断都只是截至检索日的检索结论，不是可证明的全球首创性。

### 13.3 本报告没有完成的事项

**[OPEN QUESTION]** 本报告没有实际下载并运行所有外部方法 checkpoint，也没有复现 TPO、LaCoT、LTF、SCM 的训练结果。对这些工作的结论限于论文方法、官方材料和可用源码所支持的范围。

**[OPEN QUESTION]** OMPI 当前只有独立公式设计与 gate protocol，没有真实 checkpoint cross-path matrix；因此“responsibility 非退化”“gradient variance 更低”“matched-compute 更好”全部仍是假设。

**[OPEN QUESTION]** PCMC 当前没有证明 natural soft actions 的 non-closure 会造成因果 reward 损失，也没有证明 single-forward barycenter 存在。

**[COLLISION RISK]** 即使 Gate 1A 为正，OMPI 仍可能被后续或遗漏文献完全覆盖；进入训练前必须进行一次以 “latent-variable TPO / marginal policy improvement / nuisance trajectory marginalization / response-class policy optimization” 为关键词的更新检索。

### 13.4 独立性声明

本报告没有继承仓库旧排名。实际排序受以下新证据主导：

- exact-density replacement 的冻结 utility 结果为负；
- SWITCH V32/FCTR calibration 为负；
- forward-horizon sacrificial gate 为负；
- mixed-measure 与 coupled-group exact red team 为负；
- TPO、JEPO、LaCoT、LTF、SCM 对剩余空间的进一步压缩。

因此，本报告没有把仓库当前或历史“active”标签当作正面先验。

---

## 14. 核心参考文献与源码索引

### 14.1 本项目与直接 latent-policy 对象

- [Latent-GRPO paper](https://arxiv.org/abs/2604.27998)
- [Latent-GRPO official code](https://github.com/DJC-GO-SOLO/Latent-GRPO/tree/c0994fb781a2d180662bb522d8ff3e8638dcf56d)
- [Latent-SFT / Chain of Superposition](https://arxiv.org/abs/2510.15522)
- [SofT-GRPO](https://arxiv.org/abs/2511.06411)
- [LEPO](https://arxiv.org/abs/2604.17892)
- [NF-CoT](https://arxiv.org/abs/2606.06447)
- [SWITCH](https://arxiv.org/abs/2606.13106)
- [Latent Thought Flow](https://arxiv.org/abs/2606.16222)

### 14.2 OMPI 的核心碰撞

- [Target Policy Optimization](https://arxiv.org/abs/2604.06159)
- [Beyond Verifiable Rewards / JEPO, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6bd67a424dc59481e1e5a5061ffc8dfe-Abstract-Conference.html)
- [Latent Chain-of-Thought for Visual Reasoning / LaCoT](https://arxiv.org/abs/2510.23925)
- [Importance Weighted Autoencoders](https://arxiv.org/abs/1509.00519)
- [VIMCO](https://arxiv.org/abs/1602.06725)
- [Doubly Reparameterized Gradient Estimators](https://arxiv.org/abs/1810.04152)
- [Reweighted Wake-Sleep](https://arxiv.org/abs/1406.2751)
- [Stochastic Computation Graphs](https://arxiv.org/abs/1506.05254)

### 14.3 PCMC 的核心碰撞

- [Soft Thinking](https://arxiv.org/abs/2505.15778)
- [Text Generation Beyond Discrete Token Sampling / MoI](https://arxiv.org/abs/2505.14827)
- [Soft Concept Mixing](https://arxiv.org/abs/2511.16885)
- [Mixup Regularization: A Probabilistic Perspective](https://proceedings.mlr.press/v286/el-laham25a.html)
- [Manifold Mixup](https://arxiv.org/abs/1806.05236)
- [On the Jensen Gap](https://arxiv.org/abs/1712.05267)

### 14.4 RLVR、信用与序列优化

- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)
- [RLOO / Back to Basics](https://arxiv.org/abs/2402.14740)
- [DAPO](https://arxiv.org/abs/2503.14476)
- [Dr.GRPO](https://arxiv.org/abs/2503.20783)
- [GSPO](https://arxiv.org/abs/2507.18071)
- [RUDDER](https://arxiv.org/abs/1806.07857)
- [VinePPO](https://arxiv.org/abs/2410.01679)
- [PRIME](https://arxiv.org/abs/2502.01456)
- [CVT-RL](https://arxiv.org/abs/2606.05263)
- [IBPO](https://arxiv.org/abs/2605.16302)
- [BiPACE](https://arxiv.org/abs/2606.25556)

### 14.5 仓库中决定本报告排序的负结果

- [Policy Contract Audit experiment log](https://github.com/L-Dramatic/latentgrpo/blob/9ccd18295941b59d4862ba5a790f7a44c4b9fae2/research/policy_contract_audit/EXPERIMENT_LOG.md)
- [Top-K Concrete experiment log](https://github.com/L-Dramatic/latentgrpo/blob/9ccd18295941b59d4862ba5a790f7a44c4b9fae2/research/topk_concrete/EXPERIMENT_LOG.md)
- [Score-Squashed Gumbel experiment log](https://github.com/L-Dramatic/latentgrpo/blob/9ccd18295941b59d4862ba5a790f7a44c4b9fae2/research/score_squashed_gumbel/EXPERIMENT_LOG.md)
- [Mixed-Measure decision](https://github.com/L-Dramatic/latentgrpo/blob/9ccd18295941b59d4862ba5a790f7a44c4b9fae2/research/mixed_measure_policy/DECISION.md)
- [Coupled Group Exploration exact gate](https://github.com/L-Dramatic/latentgrpo/blob/9ccd18295941b59d4862ba5a790f7a44c4b9fae2/research/coupled_group_exploration/EXACT_GATE.md)
- [SWITCH C2 attempt-5 postmortem](https://github.com/L-Dramatic/latentgrpo/blob/9ccd18295941b59d4862ba5a790f7a44c4b9fae2/research/coordinate_invariance/SWITCH_C2_ATTEMPT5_POSTMORTEM.md)
- [Forward-KL sacrificial discovery kill report](https://github.com/L-Dramatic/latentgrpo/blob/9ccd18295941b59d4862ba5a790f7a44c4b9fae2/research/behavioral_geometry/P1_SACRIFICIAL_DISCOVERY_KILL_REPORT_ZH.md)

---

## 15. 最终一句话结论

> **不要再修 latent likelihood。先用 OMPI Gate 0/1A 检验“同一高奖励可见响应是否由多条自然 latent paths 共同解释”；若责任分布退化，就停止训练。PCMC 只作为独立的 mixture-semantics 备选 gate。按当前证据，项目尚无可直接启动 AAAI 方法训练的方案。**
