# P1 Sacrificial Forward-KL Discovery：终止报告

**日期：** 2026-07-16  
**协议：** `p1-sacrificial-forward-kl-v1-20260716`  
**冻结配置 SHA-256：** `d47b5089866499e29d09e8b6d2f2ea7b10c8ab3722f943dd95657744a7c2a7a0`  
**决策：** `KILL-CURRENT-FORWARD-HORIZON-GAP-MAINLINE / NO-FURTHER-GPU / NO-GO-CAL`

## 1. 一句话结论

在 8 个 source-format prompts、每题 4 个自然 Gumbel actions、forward/reverse
各 4 条独立 histories 的冻结 discovery 中，**短 horizon 没有错排 action，且
几乎全部可观测 KL 在 H=8 前已经累积完成**。当前 Family A 不支持值得继续投入
AAAI main 方法研发的 Forward Horizon Gap；按照预注册规则终止该主线。

## 2. 执行完整性

| 项目 | 结果 |
|---|---:|
| action/endpoint records | 40 / 40 |
| directional path records | 256 / 256 |
| duplicate records | 0 |
| error/OOM/invalid records | 0 |
| natural-visible prompt-action pairs | 32 / 32 |
| natural-visible rate | 1.0 |
| forward histories | 128 |
| reverse histories | 128 |
| manifest measured runtime | 375.78 s |
| process runtime including checkpoint hashing/startup | 466 s |
| peak allocated VRAM | 2,612,642,304 bytes |
| peak reserved VRAM | 2,694,840,320 bytes |

checkpoint、source commit、config hash 和 runtime 均在 manifest 中固定。运行没有
恢复、补样本、改 seed、换 prompt 或延长预算。

## 3. Endpoint 与终止长度

全部 40 个 reference/candidate action 都自然进入 visible mode。latent closure
长度分布为：

| latent steps | action count |
|---:|---:|
| 2 | 5 |
| 3 | 5 |
| 5 | 5 |
| 6 | 18 |
| 8 | 7 |

全部 256 条 visible histories 都由模型 EOS `128009` 自然终止：

| valid visible steps | path count |
|---:|---:|
| 8 | 68 |
| 11 | 144 |
| 14 | 6 |
| 15 | 34 |
| 16 | 4 |

没有一条路径需要 H=32 或 H=64。EOS emission 的 KL 已计入最后一步，其后按
冻结协议进入 absorbing zero increments。因此 `D_64` 的平台化不是截断 bug，
而是该 visible law 下的真实有限路径结果。

## 4. 冻结主指标

| 指标 | 结果 | GO 门 |
|---|---:|---:|
| robust forward flip prompts | 0 / 8 | >= 2 |
| robust reverse flip prompts | 0 / 8 | sensitivity only |
| D64 risk-eligible actions | 14 / 32 | descriptive |
| eligible action late-mass rate | 0.0 | >= 0.25 |
| median D8/D64 | 0.999839 | <= 0.80 |
| forward H64 rank stability | 0.989583 | >= 0.60 |
| natural-visible rate | 1.0 | >= 0.80 |

结果不是“信号太吵所以看不出来”。H64 排名稳定度接近 0.99，说明 action 风险
排序在不同 histories 上高度稳定；只是 H=8 与 H=64 给出了同样排序。14 个超过
`D64 >= 1e-5` 数值门槛的 actions 中，没有任何一个满足冻结的 25% late-tail
比例。

预注册 KILL 条件为：无 robust flip、late-mass rate `<0.10`、median
`D8/D64 >=0.90`。三项分别为 `0`、`0.0`、`0.999839`，明确触发 KILL。

## 5. 对 idea 的含义

### 被否定的内容

- 当前 source-native Gumbel Family A 上，“H=8 会系统性低估 H=64 continuation
  risk”没有得到支持；
- 短 horizon 没有发生 leave-one-history 稳定的 action ranking reversal；
- PaTR/ACRT 在当前项目上缺少最基本的 delayed-risk 现象前提；
- 不应进入 calibration、held-out、adaptive method、真实 optimizer update 或训练。

### 没有被否定的内容

- endpoint-aware chain-rule KL 实现是正确且可复用的工程资产；
- 自然 Gumbel action、persistent Req、fresh cache 和双向 same-history evaluator
  都已经通过；
- 该实验不是对所有 latent models、所有真实 optimizer updates 或无限未来的
  数学否定。

但项目目标是高竞争力 AAAI 方法论文，而不是证明一个无限宽泛的存在性命题。
在最贴近当前 checkpoint/source 的低成本自然 family 上，冻结 discovery 给出
清晰负结果；继续换 family、扩 prompt 或上训练会违反预注册 stopping rule，且
会把研究变成事后寻找现象。

## 6. 决策边界

立即生效：

1. 终止当前 `Forward Horizon Gap -> PaTR/ACRT` 主线；
2. 不追加 prompt、action、history、horizon 或 GPU 预算；
3. 不租远程 GPU，不运行 calibration、held-out 或训练；
4. Family B 和 PaTR 的历史审计继续保留，但不再作为当前 next step；
5. 所有代码、协议、raw records 和负结果保留为可复用资产；
6. 若未来重启，必须由新的外部证据、不同机制和全新 claim contract 触发，不能
   把本次 v1 扩样当作“修复”。

下一科研动作应回到已存档 idea 的重新选拔，而不是继续优化本方向。

## 7. 可复现文件与哈希

- `results/p1_sacrificial_discovery_v1/records.jsonl`  
  SHA-256 `6036fc4ca074d9af190e9191ac744902f7ff07ac63a0e631f57b89545d055df2`
- `results/p1_sacrificial_discovery_v1/manifest.json`  
  SHA-256 `516499216eda6d8a3314cd09835597ef5b584c74f5c4d19041f3ff42af158afe`
- `results/p1_sacrificial_discovery_v1/summary.json`  
  SHA-256 `c5fdc5ac68eeee6e59329ff2ab09f617dee362f561a5dc03d700c1f6be5d9821`

独立只读复算了 decision、endpoint rate、eligible count、late-mass rate、
median ratio、rank stability 以及 forward/reverse flip maps；全部与存储的
`summary.json` 完全一致。
