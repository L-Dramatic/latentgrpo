# P1 Sacrificial Forward-KL Discovery v1

**冻结日期：** 2026-07-16  
**协议 ID：** `p1-sacrificial-forward-kl-v1-20260716`  
**配置：** `configs/p1_sacrificial_discovery_v1.json`  
**授权：** `GO-SACRIFICIAL-DISCOVERY`，仅限一次本地运行  
**仍禁止：** calibration、held-out、PaTR、Family B 伪造、训练、远程 GPU、论文结论

## 1. 自提示词

> 你是该方向的执行者和反方审稿人。运行目的不是寻找漂亮 KL，而是用一次不可扩张、可恢复、有硬预算的实验判断 Forward Horizon Gap 是否值得继续。严格使用冻结配置，不增加 prompt、seed、action、history、horizon 或预算；不查看答案、reward 或 benchmark；不使用 JVP/Fisher；不把同一 Gumbel family 的 seeds 说成多个机制。
>
> 每个 prompt 先构造一个官方无噪声 reference action 和四个全新固定 seed 的官方 noisy actions。每条分支使用自己的持久 Req 和 fresh KV cache，未来 latent suffix 使用官方无噪声 law，最多 32 步，只接受自然 endpoint。forward KL 的每条 reference history 只采样一次并共享给四个 candidate teacher-force；reverse KL 对每个 candidate 独立采样并 teacher-force reference。可见 law 是 full-softmax temperature 0.6；EOS emission 计入当前步，之后为 absorbing zero increment。
>
> 每个完成单元立即追加 JSONL、flush、fsync；重启只跳过完整 key，不重算或覆盖。超过一小时在下一个单元前停止并保存 manifest。运行完成后只按预注册规则输出 `KILL`、`HOLD-INSUFFICIENT` 或 `GO-REWRITE-METHOD-CONTRACT`，不得人工挑选 prompt、history 或 direction。

## 2. 研究对象

对于 prompt `g`、deterministic reference action `a0` 与 noisy candidate `a`：

\[
D_H^\rightarrow(g,a)
=\mathbb E_{Y\sim P_{a0}}
\sum_{t=1}^{H}
\mathrm{KL}(\pi_{a0,t}(\cdot\mid Y_{<t})\|\pi_{a,t}(\cdot\mid Y_{<t})).
\]

reverse sensitivity 为：

\[
D_H^\leftarrow(g,a)
=\mathbb E_{Y\sim P_a}
\sum_{t=1}^{H}
\mathrm{KL}(\pi_{a,t}(\cdot\mid Y_{<t})\|\pi_{a0,t}(\cdot\mid Y_{<t})).
\]

两者不能用独立 rollout logits 对齐估计。`0.5*(forward+reverse)` 只能称为
对称 KL 描述，不能称为 JS divergence。

## 3. 冻结样本与独立性

- 8 个手写、非数据集、source-format reasoning prompts；
- 每个 prompt：1 个 deterministic reference + 4 个 noisy candidates；
- forward：4 个 reference histories，四个 candidates 共享每条 history；
- reverse：每个 candidate 4 个 candidate histories；
- horizons：`{1,3,8,16,32,64}`；
- prompt 是最高层 discovery 单位；action/history 只是在 prompt 内嵌套；
- action seeds 与工程预检 seeds 完全不同；
- forward/reverse history namespaces 互不重叠。

配置中的 seeds 是 base seeds。第 `i` 个 prompt 的 action 与 history seed 统一加
`i*10000`；reverse 还为第 `j` 个 candidate 加 `j*100`。protocol lint 必须证明
所有 materialized seeds 全局唯一，且 action/forward/reverse 三个 namespace 无交集。

总计计划写入 `8 x 4 x 4 x 2 = 256` 个 directional path records，另有
40 个 action/endpoint records。不得把 256 当成 256 个独立科学样本。

## 4. Source 与 endpoint 规则

1. prompt 必须经 checkpoint chat template 构造并以 `<think>` 结束；
2. `524` 必须不在 EOS、stop-token、additional-stop 或 stop-string 集合；
3. reference first action 使用官方 no-noise sampler；candidate 使用冻结 Gumbel seed；
4. 每条 closure 持久维护一个 Req，逐步 append/check/update；
5. 自然 endpoint 显式区分 visible、length、token、string、timeout、abort；
6. 只有 reference/candidate 都为 `NATURAL_VISIBLE` 才产生 path record；
7. endpoint mismatch 按 `+inf` structural risk 记录，不平滑、不删除；
8. 本实验禁止人工 `524`。

## 5. 可见 law 与 EOS

- full vocabulary rows `0..128255`；
- temperature `0.6`；
- `top_k=-1, top_p=1.0`；
- 不 epsilon-floor，不读取 backend 的截断 logprob；
- EOS 或冻结 additional-stop token 的 conditional KL 和 emission 计入该步；
- 任一终止 emission 后所有 horizon 增量为 0，因此短路径的 `D_64` 合法等于其终止前累计值；
- tokenizer-only id `128256` 永远非法。

## 6. 预注册描述量

每个 prompt-action-direction 计算四条 history 的平均累计 KL：

`D_1, D_3, D_8, D_16, D_32, D_64`。

主要 discovery 量：

1. `Tail_8,64 = D_64 - D_8`；
2. eligible action：`D_64 >= 1e-5`；
3. late-mass action：eligible 且 `Tail_8,64 / D_64 >= 0.25`；
4. `D8/D64` 比例；
5. prompt 内 candidate ranking 在 H=8 与 H=64 的 pairwise flip；
6. natural-visible endpoint rate；
7. 单 history 的 H=64 pairwise ranking 与四-history mean ranking 的一致率。

### Robust ranking flip

对同一 prompt 的任意 candidate pair `(a,b)`，若 full mean 下
`sign(D8(a)-D8(b))` 与 `sign(D64(a)-D64(b))` 相反，并且删除四个共享 forward
history 中任意一个后，两个符号仍保持各自方向，则该 prompt 存在 robust flip。
零差值不算 flip。不用事后 effect-size 阈值替换此规则。

reverse 只作 support/mode sensitivity，不作为 primary GO 计数；若 forward 与
reverse 叙事冲突，后续 method contract 必须显式降级为方向性风险，不能宣传
对称安全。

## 7. 唯一决策规则

### `GO-REWRITE-METHOD-CONTRACT`

必须全部满足：

1. natural-visible endpoint rate `>= 0.80`；
2. 至少 2/8 prompts 有 robust forward ranking flip；
3. eligible actions 中 late-mass rate `>= 0.25`；
4. eligible actions 的 median `D8/D64 <= 0.80`；
5. H=64 pairwise rank stability `>= 0.60`；
6. 运行完整、未超预算、无 cache/request/identity 错误。

该结果只允许重写新的独立 method claim contract，并规划 actual update；不自动
恢复 PaTR，不授权 calibration。

### `KILL`

满足任一硬条件：

1. natural-visible rate `< 0.80`；
2. 没有 robust flip，late-mass rate `< 0.10`，且 median `D8/D64 >= 0.90`；
3. source/request/cache/density 实现失败或不能复现；
4. 一小时内不能在当前顺序实现上完成，且 measured cost 没有清楚的批处理修复空间。

### `HOLD-INSUFFICIENT`

其余情况，包括 rank stability `<0.60`、eligible action 太少、部分指标接近门槛、
预算停止但已有记录可恢复。HOLD 不允许增加当前 discovery 的样本；只能先做盲态
工程优化或放弃。

## 8. Baselines 与禁止项

本阶段只比较固定 horizons：H1/H3/H8/H16/H32 与直接 H64 oracle。action L2、
latent length、proxy trace 与 endpoint 只作诊断。不得实现 adaptive stopping、
PaTR、MLMC、Fisher/JVP 或学习型 predictor，因为 P1 现象尚未过门。

## 9. 预算、恢复与监控

- 本地 RTX 4060 Laptop GPU；远程 GPU 不启动；
- wall-clock hard cap：3600 秒，不含代码编译与纯 CPU lint；
- 模型在一个进程中常驻；单元顺序由配置固定；
- `records.jsonl` 每个单元 flush + fsync；
- `manifest.json` 原子替换，记录 config/source/checkpoint hash、完成 key、时间和显存；
- 重启时只接受相同 protocol/config hash；
- 异常、OOM 或 Ctrl-C 写入 manifest 后退出；已完成记录不得覆盖；
- 完整结束后写 `summary.json` 和中文决策报告。

协议冻结后，任何 config 内容变化都必须改 protocol ID，当前 v1 不再执行。
