# P1 no-JVP 决策补充：自然 closure 新证据后的重新裁决

**日期：** 2026-07-16  
**上游历史决策：** `P1_NO_JVP_FINAL_DECISION_ZH.md` 中的 `RECOMMEND-KILL-AS-AAAAI-MAIN`  
**新证据：** `P1_FORWARD_KL_PREFLIGHT_REPORT_ZH.md`  
**当前裁决：** `DO-NOT-PROMOTE-AS-METHOD / GO-ONE-SACRIFICIAL-DISCOVERY-AFTER-PROTOCOL-FREEZE / NO-GO-CAL`

## 为什么必须补充，而不是删除旧决策

旧决策在当时证据下是合理的。它基于四项主要判断：

1. Family A 的旧合成 marker 在三步内没有自然退出，只能人工消费 `524`；
2. Family B 没有真实 source group 和 actor/runtime，不能算第二自然 family；
3. PaTR 缺少对原始 KL 的合法 optional-stopping 界，不能叫 certificate；
4. 在前三项都未解决时直接投入数据、训练或大规模 calibration，产出比过低。

新预检只改变第一项，而且改变得足够明确：使用官方 eval 的 source-format chat
template 后，deterministic reference 在 5 步、首个 noisy candidate 在 3 步自然
产生 `524`，两个 endpoint 都是 `NATURAL_VISIBLE`；fresh-cache identity、forward、
reverse 和 H=64 成本链路均通过。

因此，“Family A 当前连自然 visible continuation 都没有”已经不是有效的 KILL
理由。旧 marker 的失败现在应解释为 prompt-format engineering failure，而不是
checkpoint 的自然 closure failure。

## 哪些负面判断仍然成立

- Family B 仍是 `BLOCKED-NO-DATA-AND-NO-SOURCE-ACTOR-RUNTIME`；
- 四个 Gumbel seeds 仍属于一个机制 family，不能冒充跨机制复现；
- PaTR 仍是候选直觉，不是已成立方法，更不是 certificate；
- 单个合成 prompt 的有限 KL 不构成 Horizon Gap；
- 当前不能写 AAAI main method claim，不能启动 calibration、held-out、训练或
  真实 optimizer-update 实验。

## 为什么仍值得做一次 discovery

当前最重要且最便宜的未知量已经从“链路能否运行”变成：

> 在多个 source-format prompt 与自然 Gumbel actions 上，短 horizon 是否真的会
> 错排 action，或者遗漏足以改变决策的 late continuation mass？

完整 H=64 双分支成本控制约 4.8 秒，峰值显存低于 3 GiB。一个 8-prompt、
4-candidate、4-history 的顺序 discovery 在本地卡上的预计量级远小于一次训练，
且其结果能直接关闭或重新打开主线：

- 若没有 ranking flip、late mass 很小、H=8 已稳定，则永久 KILL 的证据会比旧
  marker 更强；
- 若多个 prompt 出现 leave-one-history 稳定的 ranking flip/false-safe，才值得
  重写独立 method contract，并把 actual update 作为后续高门槛；
- 无论哪种结果，都不能直接声称 PaTR 或 AAAI 方法成立。

这是一项高信息增益、硬预算、可随时停止的牺牲性实验，不是 sunk-cost 式扩张。

## 绑定边界

1. 先冻结 `P1_SACRIFICIAL_DISCOVERY_PROTOCOL`，通过 lint 与 no-data dry-run；
2. 协议必须使用 source-format prompt、自然 endpoint、fresh cache、shared-history
   forward/reverse KL、显式 EOS absorbing semantics 和 prompt-level 聚合；
3. 运行时间上限为本地 GPU 一小时；超过即停止并保留 checkpoint；
4. 结果只能是 `KILL`、`HOLD-INSUFFICIENT` 或 `GO-REWRITE-METHOD-CONTRACT`；
5. `GO-REWRITE-METHOD-CONTRACT` 仍不等于 `GO-CAL`；
6. 不启动远程 GPU，不实现 PaTR，不构造伪 Family B。

旧 `RECOMMEND-KILL-AS-AAAAI-MAIN` 继续作为当时证据的历史记录；本补充只在
上述一次 discovery 的授权边界内覆盖其“不要再测 Family A 现象”的部分。
