# LatentGRPO 使用指南

## 概述

LatentGRPO (Latent Group Relative Policy Optimization) 是一种基于连续思想的强化学习方法，用于大语言模型的推理优化。该方法冻结所有LLM参数，只训练一个轻量级的投影模块。

## 核心特性

1. **连续思想生成**: 在LLM的潜在空间中生成K个连续思想向量
2. **多轨迹采样**: 通过噪声注入生成G条不同的推理轨迹
3. **对比正则化**: 使用InfoNCE损失保持轨迹多样性
4. **群相对优势估计**: 无需单独的价值模型
5. **固定长度优势**: 消除离散CoT中的长度偏差

## 快速开始

### 1. 训练LatentGRPO模型

```bash
python main.py \
  --mode baseline \
  --baseline latentgrpo \
  --dataset gsm8k \
  --config small \
  --num_exps 1 \
  --device 0 \
  --train_max_contemp_tokens 5 \
  --eval_max_contemp_tokens 1 \
  --latentgrpo_epochs 10 \
  --latentgrpo_lr 1e-4 \
  --latentgrpo_wd 0.01 \
  --num_trajectories 4 \
  --batch_size 4
```

### 2. 主要参数说明

#### 基础参数
- `--mode baseline`: 使用baseline模式
- `--baseline latentgrpo`: 指定使用LatentGRPO方法
- `--dataset`: 数据集名称 (gsm8k, svamp, multiarith, commonsense_qa, coin_flip)
- `--config`: 模型配置 (small, mistral, qwen)
- `--num_exps`: 实验次数
- `--device`: GPU设备ID

#### LatentGRPO特定参数
- `--train_max_contemp_tokens K`: 训练时连续思想向量的数量 (默认: 5)
- `--eval_max_contemp_tokens`: 评估时连续思想向量的数量 (默认: 1)
- `--latentgrpo_epochs`: 训练轮数 (默认: 10)
- `--latentgrpo_lr`: 投影模块学习率 (默认: 1e-4)
- `--latentgrpo_wd`: 投影模块权重衰减 (默认: 0.01)
- `--num_trajectories G`: 多轨迹采样的轨迹数量 (默认: 4)
- `--contrastive_lambda λ`: 对比损失权重 (默认: 0.1)
- `--contrastive_temperature η`: 对比损失温度参数 (默认: 0.5)
- `--kl_beta β`: KL散度权重 (默认: 0.1)

### 3. 在不同数据集上训练

#### GSM8K (数学推理)
```bash
python main.py \
  --mode baseline \
  --baseline latentgrpo \
  --dataset gsm8k \
  --config small \
  --train_max_contemp_tokens 5 \
  --num_trajectories 4 \
  --latentgrpo_epochs 10
```

#### CommonsenseQA (常识推理)
```bash
python main.py \
  --mode baseline \
  --baseline latentgrpo \
  --dataset commonsense_qa \
  --config small \
  --train_max_contemp_tokens 5 \
  --num_trajectories 4
```

#### MultiArith (数学推理)
```bash
python main.py \
  --mode baseline \
  --baseline latentgrpo \
  --dataset multiarith \
  --config small \
  --train_max_contemp_tokens 5 \
  --num_trajectories 4
```

#### SVAMP (数学推理)
```bash
python main.py \
  --mode baseline \
  --baseline latentgrpo \
  --dataset svamp \
  --config small \
  --train_max_contemp_tokens 5 \
  --num_trajectories 4
```

#### CoinFlip (符号推理)
```bash
python main.py \
  --mode baseline \
  --baseline latentgrpo \
  --dataset coin_flip \
  --config small \
  --train_max_contemp_tokens 5 \
  --num_trajectories 4
```

### 4. 使用不同的LLM骨干网络

#### Small配置 (Llama-2-7B + Sheared-LLaMA-1.3B)
```bash
python main.py \
  --mode baseline \
  --baseline latentgrpo \
  --dataset gsm8k \
  --config small \
  --latentgrpo_epochs 10
```

#### Mistral配置 (Mistral-7B + mistral-1.1b)
```bash
python main.py \
  --mode baseline \
  --baseline latentgrpo \
  --dataset gsm8k \
  --config mistral \
  --latentgrpo_epochs 10
```

#### Qwen配置 (Qwen2.5-7B + Qwen2.5-0.5B)
```bash
python main.py \
  --mode baseline \
  --baseline latentgrpo \
  --dataset gsm8k \
  --config qwen \
  --latentgrpo_epochs 10
```

## 方法详解

### 连续思想生成 (Eq. 1)
```
h_k = LastHidden(LLM_φ([E_x; c_1; ...; c_{k-1}]))
c_k = Proj_θ(h_k), k = 1, ..., K
```
- 使用冻结的LLM生成隐藏状态
- 通过可训练的投影模块映射为连续思想向量

### 投影模块 (Eq. 2)
```
z_k = W_2 * σ(W_1 * h_k + b_1) + b_2
c_k = LayerNorm(z_k)
```
- 两层MLP + LayerNorm
- 参数量轻量，仅训练此模块

### 多轨迹采样 (Eq. 3)
```
c̃_1^{(i)} = c_1 + ε^{(i)}, ε^{(i)} ~ N(0, I_d)
```
- 在第一个思想向量注入高斯噪声
- 生成G条不同的推理轨迹

### 对比正则化 (Eq. 4)
```
L_cl = -Σ_{i=1}^G log(exp(τ_i · τ_i / η) / Σ_{j=1}^G exp(τ_i · τ_j / η))
```
- 使用InfoNCE损失保持轨迹多样性
- 防止轨迹表示塌陷

### 群相对优势 (Eq. 5)
```
Â_i = (r_i - mean({r_j})) / std({r_j})
```
- 基于组内归一化的相对优势估计
- 无需单独的价值模型

### 策略优化损失 (Eq. 6)
```
L_LatentGRPO = -(1/G) * Σ_{i=1}^G Â_i * log p_φ(a_i | τ̃_i(θ), x) + β * D_KL(π_θ || π_ref)
```
- 结合策略梯度和KL正则化
- 通过可微的连续思想向量端到端优化

### 总损失 (Eq. 7)
```
L = L_LatentGRPO + λ * L_cl
```
- 结合策略优化和对比正则化

## 输出结果

训练和评估结果保存在：
```
results/baseline/latentgrpo/{config}/{dataset}/
├── logs/              # 训练日志和TensorBoard数据
├── results/           # 评估结果 (JSONL格式)
└── saved_model_exp=*/ # 保存的模型检查点
```

### 评估结果格式
```json
{
  "numerical_accuracy": 0.95,
  "ave_sample_time": 1.23,
  "dataset": "gsm8k",
  "eval_temp": 0.7,
  "train_max_contemp_tokens": 5,
  "eval_max_contemp_tokens": 1,
  "exp_num": 0
}
```

## 超参数调优建议

### 1. 连续思想数量 (K)
- **训练时**: 5-10个思想向量
- **评估时**: 1-3个思想向量（提高推理速度）
- 数学推理任务可以使用更多思想向量

### 2. 轨迹数量 (G)
- **默认值**: 4条轨迹
- 增加轨迹数量可以提高多样性，但会增加计算成本
- 建议: 4-8条轨迹

### 3. 学习率和权重衰减
- **投影模块**: lr=1e-4, wd=0.01
- 可以根据数据集调整：
  - 数学推理: lr=1e-4
  - 常识推理: lr=5e-5

### 4. 对比损失权重 (λ)
- **默认值**: 0.1
- 如果轨迹多样性不足，可以增加到0.2
- 如果训练不稳定，可以降低到0.05

### 5. 训练轮数
- **默认值**: 10轮
- 小数据集: 5-10轮
- 大数据集: 10-20轮
- 根据验证集准确率早停

## 与其他方法对比

| 方法 | 参数量 | 训练目标 | 优势 |
|------|--------|----------|------|
| LatentGRPO | 仅投影模块 | RL + 对比 | 无需过程标注 |
| SoftCoT | 投影模块 | 监督学习 | 简单直接 |
| SemCoT | 多个模块 | 监督学习 | 语义对齐 |
| ICoT-SI | 全部参数 | 监督学习 | 训练充分 |

## 常见问题

### Q1: 为什么冻结LLM参数？
A: 减少计算成本，避免灾难性遗忘，专注于学习推理模式而非知识。

### Q2: 如何选择轨迹数量G？
A: G=4是较好的默认值。增加G可以提高多样性，但会线性增加计算时间。

### Q3: 训练不稳定怎么办？
A: 尝试：
- 降低学习率 (1e-4 → 5e-5)
- 减少对比损失权重 (0.1 → 0.05)
- 增加KL散度权重 (0.1 → 0.2)

### Q4: 评估时为什么使用更少的思想向量？
A: 提高推理速度。训练时使用更多思想向量以充分学习，评估时减少以提高效率。

### Q5: 如何处理长文本输入？
A: 增加`--max_seq_len`参数，但要注意GPU内存限制。

## 引用

如果您使用了LatentGRPO，请引用相关论文。

## 许可证

本项目遵循项目主许可证。