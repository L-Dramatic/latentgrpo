# PCMC A0 托管执行手册

## 当前授权边界

当前只允许运行检查点推理 Gate A0，不允许 A1、Gate B 或训练。A0 完成后
无论结果是推进还是 KILL，机器都必须关机，后续阶段重新审计后再单独授权。

## 远端持久目录

- 仓库：`/root/autodl-tmp/latentgrpo`
- 运行状态：`/root/autodl-tmp/pcmc-a0-ops/state`
- 日志：`/root/autodl-tmp/pcmc-a0-ops/logs`
- 结果：`/root/autodl-tmp/pcmc-a0-ops/results`
- 证据：`/root/autodl-tmp/pcmc-a0-ops/evidence`

所有进度均写入数据盘。GPU 重启后重新启动同一入口，会校验协议哈希，并只补
缺失题目；已完成记录不可覆盖。

## 启动

```bash
cd /root/autodl-tmp/latentgrpo
bash research/policy_conditional_mixture_closure/ops/start_pcmc_a0_managed.sh
```

启动脚本会分别创建主控制器和独立看门进程。不要手工并发启动第二份作业，
文件锁会拒绝重复执行。

## 状态检查

```bash
cat /root/autodl-tmp/pcmc-a0-ops/state/managed.status
tail -n 80 /root/autodl-tmp/pcmc-a0-ops/logs/pcmc_a0_managed.log
tail -n 20 /root/autodl-tmp/pcmc-a0-ops/logs/pcmc_a0_gpu_metrics.csv
```

状态 `RUNNING` 表示正在执行；`SHUTDOWN_PENDING` 表示结果已经整理并进入重复
关机请求。平台关机是否真正完成必须从本地连续检查 SSH 后端不可达来确认，
不能只看网页 TCP 代理端口。

## 异常策略

- 单题结果采用原子分片，进程中断不会损坏此前进度。
- 协议、模型、数据、源码或预检哈希不匹配时立即停止。
- 剩余磁盘低于 50 GiB、GPU 连续异常、心跳超过 10 分钟或总运行超过 10
  小时时，看门程序终止主作业、同步证据并持续请求关机。
- 单检查点 A0 最长 4 小时。触顶后保留进度并关机，不把不完整结果解释为
  科学结论。

## 完成产物

最终证据包位于：

`/root/autodl-tmp/latentgrpo/artifacts/pcmc_gate/pcmc_a0_return_bundle.tar.gz`

其中包括资产校验、工程预检、1000 条 A0 记录、两个运行清单、A0 决策、
协议、日志和 GPU/磁盘监控。只有 `a0_decision.json` 的冻结规则判定可决定
下一步。
