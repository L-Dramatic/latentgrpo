# SWITCH C2 AutoDL 运行手册

## 当前结论

现在仍未训练，也未下载 24.2 GB 的 Qwen3-8B + SWITCH LoRA 权重。本地 RTX
4060 已完成 C1 微分与坐标传输验证，但其 8 GB 显存不能运行 SWITCH C2。

C2 是训练前的科学闸门，不是正式训练。它依次执行：

1. 8 题 paper-final checkpoint 身份等价检查；
2. 按固定顺序扫描全部 500 道 MATH-500，选定 16 个 calibration 和 32 个 test；
3. calibration 只选择一个 probe 尺度、更新强度和最强简单基线；
4. 在未调参的 32 题 test 上作最终 go/no-go 判断。

只有第 4 步全部通过，才开发高效 V32/FCTR estimator；estimator 通过匹配算力
复核后，才进入 GRPO 训练。

## 推荐实例

- 首选：单卡 NVIDIA H20 96 GB；
- 次选：A100 80 GB 或 H100 80 GB；
- 不建议：48 GB 及以下显存，JVP 和 64-token 微分 replay 的余量不足；
- CPU 内存：至少 64 GB，建议 128 GB；
- 工作区可用磁盘：至少 100 GB；
- 镜像：Ubuntu 22.04、Python 3.10、PyTorch 2.5 或更高、CUDA 12.x。

默认脚本要求可见显存至少 78 GiB、空闲磁盘至少 70 GiB。不要为了在小卡上
运行而降低 horizon、样本数、probe 数或门槛；那会产生另一个实验。

## 启动

仓库同步到实例后，在仓库根目录执行：

```bash
bash research/coordinate_invariance/run_switch_c2_autodl.sh prepare
```

`prepare` 会创建隔离环境、锁定官方 SWITCH 源码、下载固定版本 MATH-500、
检查 GPU/磁盘/BF16，并运行 SWITCH/C2 专项测试。它不会下载大模型权重。

准备通过后，建议在 `tmux` 中分阶段运行：

```bash
bash research/coordinate_invariance/run_switch_c2_autodl.sh identity
bash research/coordinate_invariance/run_switch_c2_autodl.sh eligibility
bash research/coordinate_invariance/run_switch_c2_autodl.sh calibration
bash research/coordinate_invariance/run_switch_c2_autodl.sh test
```

也可以一次托管执行：

```bash
bash research/coordinate_invariance/run_switch_c2_autodl.sh all \
  2>&1 | tee artifacts/coordinate_invariance/switch_c2_autodl.log
```

首次 `identity` 会下载并逐文件校验基础模型和 adapter。后续阶段复用同一缓存。

## 断点与产物

每个长阶段逐题追加 JSONL journal：

```text
artifacts/coordinate_invariance/journals/
```

journal 同时绑定 config hash 和 implementation hash。实例中断后重跑相同命令会
从最后一条完整记录继续；代码或 config 变化后不会错误复用旧 journal。

正式产物为：

```text
artifacts/coordinate_invariance/switch_checkpoint_identity_smoke_v1.json
artifacts/coordinate_invariance/switch_c2_eligibility_v1.json
artifacts/coordinate_invariance/switch_c2_calibration_v1.json
artifacts/coordinate_invariance/switch_c2_test_v1.json
```

不要手工编辑 journal 或正式 JSON。若工程代码确需修复，保留旧文件，使用修复后
自动生成的新 implementation-key journal。

## 停止规则

- `identity` 失败：停止，只排查源码、权重、tokenizer 或 replay 等价性；
- eligibility 不足 48 题或 test 多样性不足：停止，不换样、不放宽条件；
- calibration 数值/传输控制失败：停止，不进入 test；
- test 任一科学 gate 失败：FCTR method 线停止，保留为负结果；
- test 全部通过：只授权 estimator 开发，不直接把局部 oracle 当成训练方法。

脚本以非零状态退出就是硬停止信号。不要通过 `FORCE=1`、降低显存门槛或删除
失败字段绕过科学判据；`FORCE=1` 仅用于在当前实现下有意重跑已通过阶段。
