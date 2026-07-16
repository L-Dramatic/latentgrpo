# P1 Forward-KL Source-Action 预检：自提示词与验收契约

**状态：** 本阶段执行提示词；只授权无数据、无训练的工程预检。  
**适用方向：** no-JVP redesign / Family A source-native action。  
**禁止外推：** 不得据此声称 Horizon Gap、风险证书、PaTR 有效、真实更新有效或 `GO-CAL`。

## 自提示词

> 你是本项目的首席研究工程师和最严格的内部审稿人。你的任务不是让实验“跑出一个数”，而是判断当前 Latent-GRPO checkpoint 是否支持一个定义清楚、可复现、不会把人工控制误报为科学证据的 continuation-risk evaluator。
>
> 只使用本地冻结 checkpoint、固定官方源码、合成 prompt 和预先声明的随机种子。禁止访问数据集、奖励、答案、校准集、测试集，禁止训练、JVP、Fisher、Jacobian 和事后改参数。
>
> 必须同时满足：官方 Gumbel action 被真实 request 顺序消费；reference 使用无噪声官方 action，candidate 使用固定种子的官方 noisy action；每条 latent closure 使用自己的 request 与 KV cache；每一步执行 append -> check_finished -> update_latent_info；自然 524、token/string/length stop、timeout 和 abort 均作为显式 endpoint；只有双方自然进入 visible mode 时，visible continuation KL 才能成为未来科学 estimand 的候选证据。
>
> 人工强制 524 只允许验证 sample/teacher-force plumbing。其结果必须标记为 control，不得进入现象判断。对同一 action 的两个独立分支必须得到零 endpoint divergence、逐步零 KL 和零累计 KL。对不同 action 必须分别在 reference histories 和 candidate histories 上计算 forward/reverse chain-rule KL；独立 rollout 对独立 rollout 的 logits 比较不被接受。
>
> 先运行精确小词表测试，验证 identity、方向不对称、chain-rule equality、support mismatch 和 endpoint atom 语义；再运行真实 WSL checkpoint。任何 cache 共享、identity 非零、非法 token、request 顺序错误、非有限 full-softmax KL 或自然 endpoint 被人工替换，都判失败。
>
> 最终只能给出以下决策之一：`PASS-NATURAL-FORWARD-KL-PREFLIGHT`、`PASS-PLUMBING-HOLD-NATURAL-CLOSURE`、`FAIL-FORWARD-KL-PREFLIGHT`、`BLOCKED`。无论哪一种，都保持 `NO-GO-CAL`，直到单独冻结并授权牺牲性发现协议。

## 冻结对象

1. checkpoint：`_models/Latent-GRPO-Llama-1B`；
2. runtime：`Python 3.11.13 / torch 2.6.0+cu124 / transformers 4.51.1 / pinned SGLang 0.4.6.post1`；
3. source latent-end id：`524`；模型词表行：`0..128255`；
4. reference first action：官方 sampler、`add_noise_gumbel_softmax=False`；
5. candidate first actions：官方 sampler、四个预声明 Gumbel seeds；
6. future latent closure：每条分支自己的 cache，官方无噪声 sampler，自然上限 `32` latent steps；
7. visible engineering law：全词表 softmax、temperature `1.0`；这不是最终论文的 deployment law；
8. plumbing control：最多三步无噪声 latent suffix 后可人工消费 `E_524`，必须显式标记；
9. identity tolerance：逐步和累计 KL 的绝对值不超过 `1e-8`，且 endpoint atom 相同；
10. 所有随机种子、prompt、horizon 和阈值在运行前固定。

## 四道门

### G1：Source authenticity

- prompt 由 checkpoint 自带 chat template 构造并以 `<think>` 结束；
- deterministic reference 与 noisy candidate 均来自未修改的 pinned sampler replay；
- proposal 必须通过真实 `Req` 的持久状态执行，而不是每步新建 request；
- `524` 不得出现在 EOS、显式 stop 或 additional-stop 集合中。

### G2：Endpoint integrity

- endpoint 至少区分 `NATURAL_VISIBLE`、`FINISH_LENGTH`、`FINISH_TOKEN`、
  `FINISH_STRING`、`LATENT_TIMEOUT`、`EXECUTION_ABORT` 与
  `FORCED_VISIBLE_CONTROL`；
- endpoint 不同的确定性分支具有扩展实数 `+inf` endpoint KL；
- 双方均不是自然 visible 时，不得报告科学 visible continuation KL；
- 人工边界只能产生 plumbing control。

### G3：Estimator identity and directionality

- identity branches 使用独立 cache，但 logits、forward KL 和 reverse KL 为零；
- `KL(P_ref || P_cand)` 只在 reference sampled histories 上计算；
- `KL(P_cand || P_ref)` 只在 candidate sampled histories 上计算；
- EOS emission 计入该步，随后进入 absorbing state；
- full-softmax 密度不得 epsilon-floor、截断或静默 clamp。

### G4：Runtime and cost

- 记录每个阶段 wall time、forward step 数、峰值 allocated/reserved VRAM；
- 先完成 `H=4` 工程门；只有自然 pair 可估计时才运行一个 `H=64, R=1`
  科学路径性能单元；否则只报告 plumbing 成本，不外推 discovery 预算；
- 远程 GPU 不在本阶段授权范围内。

## 决策规则

- **PASS-NATURAL-FORWARD-KL-PREFLIGHT：** G1-G4 通过，reference 和至少一个
  distinct candidate 均自然进入 visible mode，identity 为零，双向路径可计算。
- **PASS-PLUMBING-HOLD-NATURAL-CLOSURE：** source action、request、cache、identity
  和人工 plumbing 通过，但冻结的自然 closure 没有形成可比较 visible pair。
- **FAIL-FORWARD-KL-PREFLIGHT：** 任一真实性、cache、identity、密度或 endpoint
  语义门失败。
- **BLOCKED：** runtime、模型或本地 CUDA 不可用，且没有得到科学结果。

下一阶段只有在 `PASS-NATURAL-FORWARD-KL-PREFLIGHT` 后才能另行冻结
`8 prompts x 4 actions x 4 histories` 牺牲性发现协议；本文件本身永远不授权该实验。
