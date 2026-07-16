# P1 Forward-KL Source-Action 预检报告

**日期：** 2026-07-16  
**执行范围：** 本地冻结 checkpoint、固定官方源码、source-format 合成 prompt、无数据、无训练  
**决策：** `PASS-NATURAL-FORWARD-KL-PREFLIGHT / GO-DISCOVERY-PROTOCOL-FREEZE / NO-GO-DISCOVERY-EXECUTION / NO-GO-CAL`

## 1. 这次通过了什么

本次结果证明当前 checkpoint/runtime 可以支持 no-JVP redesign 所需的最小
source-faithful evaluator：

1. deterministic reference 和四个固定种子的 noisy candidate 均由 pinned
   官方 sampler 产生，并被真实 `Req` 顺序消费；
2. 每条 latent closure 使用持久 request 和独立 KV cache，每一步严格执行
   `append -> check_finished -> update_latent_info`；
3. reference 与首个 distinct candidate 都在冻结的 32-step cap 内自然产生
   `524` 并进入 visible mode；
4. visible continuation 在同一 sampled history 上执行 sample/teacher-force；
5. identity action 的独立 cache 双分支得到逐步和累计严格零 KL；
6. forward 与 reverse KL 使用各自方向的 freshly rebuilt closure，未复用已经
   增长过的 cache；
7. 一个完整 64-step 双分支成本控制能在本地 RTX 4060 Laptop GPU 上运行。

这不是 Horizon Gap 结果。当前只使用一个合成 prompt、一个 candidate 和工程
horizon，不能支持现象、泛化、风险阈值或方法收益声明。

## 2. 冻结输入与运行时

| 项目 | 值 |
|---|---|
| checkpoint | `_models/Latent-GRPO-Llama-1B` |
| model rows | `0..128255` |
| source latent-end id | `524` |
| prompt | checkpoint chat template + 一个手写合成算术问题 |
| reference action | 官方 no-noise top-10 mixture |
| candidates | 官方 one-sided Gumbel action，4 个预声明 seed |
| future closure | 官方 no-noise sampler，自然 cap `32` |
| engineering visible law | full-vocabulary softmax，temperature `1.0` |
| runtime | Python 3.11.13 / torch 2.6.0+cu124 / transformers 4.51.1 / pinned SGLang |

合成问题不来自任何数据集；没有读取答案、reward、calibration 或 held-out 数据。
prompt 使用与官方 eval 相同的 `apply_chat_template(..., add_generation_prompt=True)`
和 `add_special_tokens=False` 路径，并以 `<think>` 结束。

## 3. Source action 与自然 endpoint

deterministic reference 的首 action proxy 为 `18`。四个 noisy candidate 的
proxy 分别为 `220, 220, 24, 2511`，相对 reference action embedding 的 L2
距离分别为：

| seed | proxy | L2 to reference | 被 source 消费 |
|---:|---:|---:|:---:|
| 2026071601 | 220 | 0.126592 | 是 |
| 2026071602 | 220 | 0.202069 | 是 |
| 2026071603 | 24 | 0.359848 | 是 |
| 2026071604 | 2511 | 0.494894 | 是 |

首个 candidate 用于端到端门。自然 closure 为：

| 分支 | proxy trace | endpoint | latent steps |
|---|---|---|---:|
| reference | `18 -> 35124 -> 82460 -> 2304 -> 524` | `NATURAL_VISIBLE` | 5 |
| candidate | `220 -> 35124 -> 524` | `NATURAL_VISIBLE` | 3 |

两个 cache 的底层 storage 指针不重叠。endpoint atom 相同，endpoint KL 为
`0`，因此 visible continuation law 在这一工程样本上定义良好。

人工 `E_524` 仍保留为 plumbing control，但只要任一分支使用人工边界，该
pair 就被标记为非科学控制。若与自然 endpoint 混合，endpoint KL 按扩展实数
记为 `+inf`，不得用 epsilon 或平滑掩盖。

## 4. Identity 与双向同历史 KL

identity control 对同一 deterministic action 构造两个独立 cache 分支，并为
forward/reverse 各自重建 fresh closure：

| 检查 | 结果 |
|---|---:|
| cache storage disjoint | true |
| forward max per-step KL | 0.0 |
| forward total KL | 0.0 |
| reverse max per-step KL | 0.0 |
| reverse total KL | 0.0 |
| frozen tolerance | `1e-8` |

自然 reference/candidate pair 的工程 H=4 结果为：

| 方向 | total KL |
|---|---:|
| `KL(P_ref || P_candidate)` | 0.0001949150 |
| `KL(P_candidate || P_ref)` | 0.0007495184 |

这两个值只证明 estimator 能得到有限、非负且方向不对称的结果。它们既不是
effect size，也不能说明长 horizon 是否必要。

一次旧的内部运行曾在 forward 后复用同一个可能增长的 Transformers cache 做
reverse。fresh-cache 审计后结果发生变化，因此该旧数值作废；本报告只保留每个
方向重新闭合 latent path 后的结果。

## 5. H=64 单单元成本

预声明的 sampled `H=64, R=1` history 在第 8 个 visible token 自然产生 EOS。
该 emission 计入 KL，随后进入 absorbing state；不允许重抽 seed 来获得更长
路径。它只能提供早停路径成本：

| 项目 | 结果 |
|---|---:|
| requested horizon | 64 |
| valid visible steps | 8 |
| sampled path terminated by EOS | true |
| sampled forward total KL | 0.0001981801 |
| sampled path with fresh closures | 1.360 s |

为测量完整 64-step 成本，另运行固定非 EOS token `220` 的重复 teacher-force
控制。它不是随机 law，不是风险结果，其 KL 数值禁止用于论文：

| 项目 | 结果 |
|---|---:|
| full visible steps | 64 |
| 64-step paired sample/force core | 4.795 s |
| throughput | 13.35 visible steps/s |
| complete benchmark block | 6.541 s |
| peak allocated VRAM | 2.746 GiB |
| peak reserved VRAM | 2.818 GiB |
| total one-shot process including model load | 76.27 s |

模型加载约 46.5 秒，是一次性进程的主要开销；批量 discovery 必须常驻模型，
不能把每个 action 都按 76 秒外推。另一方面，固定 token 控制的 KL 很大且路径
完全人工，进一步说明成本控制绝不能混入科学现象统计。

## 6. 可复现证据

- 执行提示词与 gate：`P1_FORWARD_KL_PREFLIGHT_SELF_PROMPT_ZH.md`；
- endpoint/KL 纯契约：`p1_forward_kl_contract.py`；
- 真实执行器：`p1_forward_action_preflight.py`；
- 原始机器可读结果：`results/P1_FORWARD_KL_PREFLIGHT_20260716.json`；
- 新增契约测试：`tests/test_p1_forward_kl_contract.py`。

执行命令：

```bash
/home/lixingshuo/.venvs/latentgrpo-py311/bin/python -m \
  research.behavioral_geometry.p1_forward_action_preflight \
  --benchmark-horizon 64 \
  --output /mnt/e/LantentGRPO/research/behavioral_geometry/results/P1_FORWARD_KL_PREFLIGHT_20260716.json
```

P1 相关测试结果为 `46 passed, 8 subtests passed`。仓库全量 Windows test
collection 另被旧环境的依赖冲突阻塞：当前全局 `huggingface-hub==1.2.4`，而
已安装 `transformers` 要求 `<1.0`。三个旧 policy-contract 测试在 import 时失败；
这不是本次 P1 测试失败，也不能被报告为全仓测试通过。

## 7. 绑定决策

### 允许

- 冻结一个新的、纯 discovery 的小规模协议；
- 设计 source-format 合成/公开 prompt 单元、action bank、history seed、endpoint
  统计、forward/reverse/event audit 和固定长视野基线；
- 在协议审核通过后使用本地 GPU 运行牺牲性 discovery。

### 仍不允许

- 直接启动报告建议的 `8 x 4 x 4`，因为 prompt bank、独立统计单位、EOS/timeout
  处理和 baseline 尚未冻结；
- calibration、held-out test、PaTR、Fisher/JVP、训练或真实 optimizer update；
- 把本报告中的 H=4/H=8 数值称为 Horizon Gap 证据；
- 租用远程 GPU。

下一唯一任务是 `P1_SACRIFICIAL_DISCOVERY_PROTOCOL`。它必须先通过 protocol
lint 和 dry-run，才可把当前 `NO-GO-DISCOVERY-EXECUTION` 改为一次性、本地、
有硬预算的 `GO-SACRIFICIAL-DISCOVERY`。
