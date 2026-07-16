# Literature Collision Audit

Audit date: 2026-07-16.

| Work | Existing result | Collision |
|---|---|---|
| [Arithmetic Sampling (ICML 2023)](https://arxiv.org/abs/2210.15458) | Parallel diverse LLM decoding with unbiased, consistent expectations under the original model; reports lower reward-estimation variance. | Direct collision with sequence-level marginal-preserving diverse sampling for LLMs. |
| [Policy Learning and Evaluation with Randomized Quasi-Monte Carlo (AISTATS 2022)](https://proceedings.mlr.press/v151/arnold22a.html) | Replaces Monte Carlo policy samples with low-discrepancy randomized point sets and reduces policy-gradient/value variance. | Direct collision with the unbiased raw-score redesign. |
| [Adaptive Antithetic Sampling (ICML 2019)](https://proceedings.mlr.press/v97/ren19b.html) | Learns correlated samples while preserving unbiased Monte Carlo estimation. | Blocks a generic antithetic-variance claim. |
| [Group-Aware Policy Optimization (EMNLP 2025)](https://arxiv.org/abs/2511.12596) | Optimizes group-level diversity and coverage properties in LLM training. | Crowds broad group-diversity positioning, although it modifies rewards rather than the sampling copula. |
| [Smaller Models are Natural Explorers for Policy-Level Diversity in GRPO (2026)](https://arxiv.org/abs/2605.30789) | Improves GRPO exploration with temporally coherent small-model policies rather than token-level noise. | Raises the baseline bar and directly attacks incoherence from local randomization. |

## Remaining distinction

The narrow possible distinction was the interaction between an exact
marginal-preserving sampling copula and a group-relative LLM policy gradient.
The exact gate shows that this interaction is a liability: coupling invalidates
the independence used by within-group baselines. The unbiased fallback removes
the group-relative component and falls inside established RQMC/antithetic work.

## Collision verdict

Method novelty is insufficient for a primary AAAI paper after the C3 failure.
The result is still useful as a warning: marginally on-policy trajectories do
not guarantee an unbiased multi-trajectory gradient when rewards from one
trajectory are used as baselines for another.
