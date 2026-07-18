# BCMD Source Collision Audit

Audit date: 2026-07-18

Decision: `PASS_SOURCE_BOUNDARY / HOLD_CHECKPOINT_GATE`

This decision means only that no exact BCMD objective was found. It does not
authorize training.

## Pinned primary sources

| Source | Pin | Relevant result |
|---|---|---|
| Multiplex Thinking official code | Git commit `673b26025569be75d640064cd72cc36dde9193d0` | Public branch-and-merge implementation inspected locally. |
| Multiplex Thinking 7B | HF revision `1e660b7993c916b2f5902ca50ea2ca82c04c2f07` | Public 15.2 GB Qwen2 checkpoint; not yet downloaded. |
| [Multiplex Thinking paper](https://arxiv.org/abs/2601.08808) | arXiv `2601.08808` | Strongest direct method neighbor. |
| [CoT2](https://openreview.net/forum?id=sTPKDKn5ig) | ICLR 2026 | Continuous supervision and policy optimization for parallel exploration. |
| [MoT-G](https://openreview.net/forum?id=RAdC1K4JXq) | ICLR 2026 submission | Explicit token mixtures optimized with GRPO. |
| [Soft Tokens, Hard Truths](https://openreview.net/forum?id=9JjKTp8Jmy) | ICLR 2026 | Scalable continuous-CoT RL and discrete-inference baseline. |
| [SofT-GRPO](https://arxiv.org/abs/2511.06411) | arXiv `2511.06411` | Gumbel-reparameterized soft policy optimization. |

## Official-code findings

The official sampler draws `K` tokens, gathers their probabilities, optionally
replaces them by uniform weights, normalizes the weights, and passes the merged
embedding forward. Its registered `multiplex_thinking` policy loss applies a
clipped PPO ratio independently to the sampled-token log probabilities and sums
the losses across the multiplex width.

No call, target, or loss in the official method computes all hard-token branch
transitions and distills their weighted predictive mixture into the merged
transition. In particular, the custom policy loss contains no second branch
forward, branch teacher, distributional KL target, or closure term.

Pinned file hashes:

- `verl-latest/verl/trainer/ppo/core_algos.py`:
  `7c99c6190e29b366aa11574067265d2341aa5b44e35da7f3fd72458596a755be`;
- `sglang-0.4.9.post6/sglang/srt/layers/sampler.py`:
  `43a6916b8221fd932b74dfc99fce1b60ab2d3fe98a11378b4b832edaffe59501`;
- official `README.md`:
  `7b5e56192a838bf23f05ea8cfb13d1fbeeac5aeadeb06ccd4f8ede53af7772d7`.

## Novelty boundary

BCMD cannot claim the first stochastic soft-token RL method, first branch-and-
merge method, first superposition diagnosis, or first continuous-CoT
distillation method. Its only defensible novelty claim is narrower:

> first local branch-consistency teacher that explicitly trains a single merged
> soft transition to match the weighted predictive mixture of the same sampled
> hard branches under a nontrivial-support constraint.

This is conceptually related to ordinary ensemble distillation, so empirical
strength must carry the paper. A weak accuracy gain on one model is not enough.

## Remaining blocker

The second positive family is not established. The Multiplex 7B checkpoint must
first pass the frozen label-blind support and branch-JS gate. Because its full
weights are 15.2 GB, the scientific run should use a 24 GB or larger GPU without
quantization. The local 8 GB laptop GPU is not suitable for that check.

The gate was frozen before checkpoint download:

- protocol SHA-256:
  `ee139e3b23fea526a50c4ec04d84b64574190d330aba99848e05c04b57de5084`;
- 64-prompt manifest SHA-256:
  `56c3d97c706d5e9e568e225ae9f9f0715ca7d9b48c32b0d49bc9542164ef5745`;
- canonical selected-record SHA-256:
  `bba8f00c3eda7244d3a0b51e9fd6dbae72e4ea496dce676c97150bb19fff8c2a`.
