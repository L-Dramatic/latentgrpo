# Family B（source-objective action）真实性审计

**日期：** 2026-07-16  
**审计范围：** 只读官方 source、现有 runtime 与已经通过的公式 contract；不读取数据集、不生成真实题目 rollout、不计算奖励、不训练。  
**结论：** `BLOCKED-NO-DATA-AND-NO-SOURCE-ACTOR-RUNTIME`  
**是否永久 kill：** 否；但在当前授权和当前运行时下，它不能被计作第二个自然 action family。

## 一句话结论

Family B 不是“从 Family A 换一个随机种子”，也不是“对当前 logits 求一个方便的梯度”。它必须从官方 Latent-GRPO 的真实 GRPO construction group 中产生：同一个题目的 8 条 source rollout、相应 reward、first-mask/winner advantage 规则、保存的 top-k/Gumbel record、old log-prob 与 response mask，共同确定 source objective 的方向。

当前禁止读取数据/做 calibration，而本机 WSL 环境也没有官方 actor 所需的 `flash_attn`、Ray 或 VERL runtime。因此，任何现在构造出的 `h + eta*g` 都会是人工 toy gradient，不能诚实地命名为 `LatentPolicy-SOURCE-GRAD`，更不能拿来凑第二个 family。

## 1. 官方 source 对 Family B 实际依赖什么

官方训练脚本的关键配置不是单样本单步梯度：

| Source 事实 | 对 Family B 的含义 |
|---|---|
| `rollout.n=8` | 同一 prompt 必须有 8 条 rollout，不能用一条任意合成记录代替 |
| `algorithm.adv_estimator=grpo` | action 方向依赖组内相对 reward，而非独立 token NLL |
| `max_response_length=128`、temperature `0.6`、top-p `0.95`、top-k `30` | construction rollout 的完整解码政策必须冻结并可复现 |
| latent top-k `10`、Gumbel temperature `1`、one-sided noise、id `524` | 首步 latent 记录必须来自同一个官方 source sampler law |
| actor update 所需张量 | `responses`、`input_ids`、`attention_mask`、`position_ids`、`old_log_probs`、`advantages`、`rollout_topk_ids`、`rollout_topk_gumbels`、`gumbel_temperature`、`token_level_rewards` |
| first-mask/winner advantage | source 会在组内用特定规则保留/置零 advantage；随意设正负 advantage 会改变 action 方向 |

源码中的 source Gumbel surrogate 与 PPO loss 已有 CPU formula contract；它们仅说明公式可被阅读与单元测试，**不产生真实 source group 的 reward/advantage，也不构成真实 actor execution**。

## 2. 当前环境事实

实际检查的 WSL runtime 为：

- PyTorch `2.6.0+cu124`，CUDA 可用；
- SGLang 可导入；
- `flash_attn`：不存在；
- Ray：不存在；
- VERL：不存在；
- 本机只有一张 8GB RTX 4060 Laptop GPU。

而官方端到端训练入口默认配置 `GPUS=0,1,2,3,4,5,6,7`，`trainer.n_gpus_per_node=8`，并通过 Ray/VERL/FSDP actor 执行其 source policy update。

这不表示“原始 scalar surrogate 数学上只能在 8 张卡计算”。对于一个未来、已授权的 construction group，首步 `g^ell` 可以研究是否能从已保存的 source records 以低算力重放；但它表示：

1. 当前不能声称已复现官方 actor 的真实运行时和 Flash-Attention 路径；
2. 当前不能生成 source-required 的真实 group advantage；
3. 任何把公式 contract 直接代入人工 advantage 的做法都只能是 toy control，不能计作 Family B。

## 3. 为什么不能用合成 advantage “先试试”

source Gumbel likelihood 的反向符号会随 advantage 与 `raw_diff` 的符号改变。人为设置 `advantage=+1/-1` 会直接选择要走的反向分支；再把该梯度通过 `W^T` 映射回 hidden state，会得到一个看似很漂亮、但由实验者自行指定的 action。

这会同时违反三件事：

- 它不是算法自然产生的 action；
- 它不能独立于风险结果被解释为 source objective；
- 它会把 Family B 降成任意 activation-edit stress probe。

因此本审计明确禁止把 synthetic loss、answer NLL、任意 reward sign、或 Family A 的 seed 差异替换为 Family B。

## 4. Family B 未来唯一可接受的低算力路径

只有在未来获得**明确的 construction/calibration 授权**后，才可尝试一个缩小但源等价的工程 gate：

1. 仅在 construction split 中选定一个预注册的 source prompt group；
2. 用冻结的官方 decode 配置生成该 prompt 的 8 条 rollout，并保存每条 latent top-k/Gumbel record、mask、old log-prob 与结束状态；
3. 用该组真实 reward 经过原始 include-overlong / first-mask-winner GRPO 规则生成 advantage；
4. 验证 first-update self-ratio、Gumbel likelihood forward 值和 backward sign 都与可运行的官方 source 路径一致；若仍无 Flash-Attention runtime，必须明确判为“未验证 kernel equivalence”，不能写 exact；
5. 从真实首步 action 的 source objective 得到 `g^ell`，用已检查的 biasless tied head 映射到 final-RMSNorm state；
6. 只将它作为 **objective-derived hidden-state counterfactual**，不称 optimizer step、参数更新或 pathwise rollout gradient；
7. 在它能作为第二 family 之前，审计 action 与 Family A 的方向角、support/proxy 变化、source-stop、自然退出与有效维数。

若该流程在单卡下无法完成，正确结论是 Family B `NO-GO-LOW-COMPUTE`，而不是降低 group size、伪造 reward、或静默改用常规 token NLL。

## 5. 对当前项目的实际影响

| 问题 | 当前答案 |
|---|---|
| Family A 是否有真实 source-native action？ | 工程上是；但尚无自然退出/语义 tail 证据 |
| Family B 是否已存在？ | 否，仅有 source formula contract |
| Family B 是否可被当前无数据预检构造？ | 不可；那会变成 toy gradient |
| Family B 是否被永久否定？ | 没有；需要未来明确 construction 授权与 source group/runtime 复现 |
| 当前能否声称两个 natural family？ | 不能 |
| 当前是否可进入 P1 calibration？ | 不能 |

## 当前最诚实的项目状态

无 JVP 路线保留了一个可运行的 Family A 前向动作接口，但强论文所需的“两个真实自然 family + 干净现象 + 合法方法包”现在还差得很远：

- Family B 被 source provenance 与授权门阻断；
- PaTR 被数学/合约审计阻断为当前方法；
- Family A 的自然退出与真实语义 tail 尚未资格化。

在获得新授权前，继续扩大实验只会制造更多看似有数值、实则不能用于论文主张的结果。当前正确动作是保留这些可复现的工程证据，并将主线维持在 `HOLD`。
