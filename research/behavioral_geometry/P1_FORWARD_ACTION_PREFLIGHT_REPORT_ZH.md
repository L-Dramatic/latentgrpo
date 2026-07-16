# P1 无 JVP Family A 前向 action 预检报告

**日期：** 2026-07-16  
**状态：** `PASS-ENGINEERING-FAMILY-A`  
**项目决策：** `HOLD / NO-GO-CAL`（保持不变）  
**范围：** 真实本地 Latent-GRPO 1B checkpoint 的纯合成、无反传工程预检；不是 P1 现象实验。

## 一句话结论

官方 Gumbel latent sampler 在真实 checkpoint 上确实能稳定给出多个、会被真实 SGLang request 接受并消费的**连续首步 latent action**；这些 action 可以在不使用 JVP/Fisher/反传的情况下，进入独立缓存的前向分支，并完成“只从参考分支采样、候选分支强制跟随同一可见历史”的链式 KL 计算。

这只证明无 JVP 路线的**工程地基可用**，完全不证明短 horizon 会误判长程风险，更不授权校准、数据集实验、方法实现或训练。

## 1. 本次到底测了什么

固定使用合成标记 `0 0 0 <think>`、官方 Top-10 sampler、固定四个 Gumbel seed：

`2026071601, 2026071602, 2026071603, 2026071604`。

每个 seed 的首步 action 都经过以下真实链路：

1. 将真实 checkpoint 的首步 logits 输入**未修改的官方 sampler**；
2. 得到官方的 top-k id、Gumbel 分数、混合概率与 proxy；
3. 将 proxy 放进真实 `Req`，严格按 `append -> check_finished -> update_latent_info` 执行；
4. 只有没有被通用停止规则截断、没有触发 524 结构退出、且 request 保留原始软混合时，才视为“实际消费的连续 latent action”；
5. 按官方 `weighted_forward` 的数值规则（FP32 概率加权、再转回 BF16）构造 action embedding；
6. 参考与候选 action 分别从独立 prefill/cache 出发，重算三步确定性 source suffix；随后**仅为测试可见 token 的 sample/force 管线**，统一强制输入 524 边界；
7. 可见阶段只从参考分支抽样，候选分支对完全相同 token 历史做 teacher forcing，并逐步计算全词表方向 KL。

全程没有读取数据集、题目、答案、奖励、校准集、测试集，也没有训练或反向传播。

## 2. 通过了什么

| 检查 | 结果 | 含义 |
|---|---|---|
| 四个固定 Gumbel seed 的首步 action | 全部通过 | 都没有被 request 停止，都保持 latent mode，均被作为连续软混合实际消费 |
| 首步 proxy | 全部为 `12` | 没有因 524/EOS/stop 被伪造成普通 action |
| 连续 action 差异 | seed 1601 与 1602 的 L2 距离为 `0.13047` | 虽然 proxy 相同，混合的 support/权重不同，连续 latent vector 确实不同 |
| request 语义 | 通过 | action 不是绕过 scheduler 直接塞进模型的伪 action |
| 参考/候选 KV cache | 完全不共享存储 | 不存在候选分支偷偷复用参考分支状态的缓存污染 |
| 共享历史 teacher forcing | 通过 4 个合成可见 token | 候选分支逐步跟随参考分支抽到的同一历史，前向 KL 可计算且有限 |
| 显存 | 峰值约 2.50 GB，结束后可用约 4.91 GB | 在现有 8GB 卡上可运行，不需要 FP32 反传 |

本次合成 plumbing control 的逐步 KL 为：

`[0.000073, 0.002334, 0.536379, 0.001264]`。

**这些数值不能作为任何 P1 结果引用。** 它们只表明两个前向分支确实不同，并且链式 KL 的 sample/force 机械流程没有断裂。

## 3. 一个重要的数值修正

官方 `weighted_forward` 不是“先把概率降为 BF16，再做加权”。它用 FP32 概率与查表向量相乘、累加，再转回 embedding dtype。

在四个固定 action 上，“旧式 BF16-first 写法”与官方算术的最大分量差均为 `0.00048828125`。数值很小，但它是真实、可测的实现差异。因此：

- 后续无 JVP 前向路线必须使用本报告的 FP32 加权规则；
- 旧 JVP 的失败结论仍然保留，不能因为这里修正了 action 构造而重解释为可用；
- 不能把任何旧的 BF16-first helper 当成官方端到端 action 的精确实现。

## 4. 没有通过、也没有声称通过的内容

### 4.1 没有自然退出到可见阶段

在预先固定的三步确定性 source suffix 中，参考和候选都持续产生 proxy `12`，没有自然发出 524。因此为了验证可见 sample/force 的工程闭环，测试脚本在这个固定小深度后**人为输入了 524**。

这意味着：

- 524 的真实 request 语义此前已经独立验证过；
- 本次验证了“一个 source-native 首步 action 后，可接入正确的 524 硬边界与可见 teacher forcing”；
- 但没有验证该合成上下文中的**自然 latent-to-visible 转换**，更没有验证真实问题上的语义 tail。

所以不能把这次运行称为“自然 source rollout 完整通过”，更不能把它当作 Horizon Gap 的正证据。

### 4.2 Family A 仍然是单一机制家族

四个 action 来自同一个官方 Gumbel exploration 机制；不同 seed 不是两个 family。即使未来 Family A 出现漂亮的长程差异，它也只能支撑单家族诊断，不能独自达到 P0 强主张所要求的两个自然 update family。

### 4.3 没有任何现象、方法或 AAAI 结论

本报告没有回答：

- 短 horizon 是否发生风险排序翻转或 false-safe；
- 该现象是否有 latent recursion 特异性；
- PaTR 是否优于直接 Monte Carlo 或 MLMC；
- 是否存在跨机制复现；
- 是否值得启动 `GO-CAL`。

这些问题在当前状态仍全部是 `NO-GO`。

## 5. 由此得到的严格下一步

1. 对 Family B 做**只检查 source objective 是否能真实生成首步 action**的工程预检；禁止把其目标梯度或 surrogate 偷换为递归 JVP 风险分数。
2. 对 Family A 的正式 P1 方案单独审计“自然退出、可见语义 tail、动作 support、分层独立性”是否在低算力下可完成。合成 marker 的强制 524 不能替代这一门。
3. 在 Family B 与 formal P1 protocol 都通过审查之前，不实现 PaTR，不启动校准或任何数据实验。

## 可复现入口

- 脚本：`p1_forward_action_preflight.py`
- 运行环境：WSL 用户环境 `/home/lixingshuo/.venvs/latentgrpo-py311/bin/python`
- 实际结果：`PASS-ENGINEERING-FAMILY-A`
- 冻结 P0 哈希：本次运行后再次核对，五个冻结文件均未变化。
