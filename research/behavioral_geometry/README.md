# Behavioral Continuation Geometry

This package tests whether a latent state must be judged by its multi-step
continuation distribution rather than by raw coordinates or only the next
token distribution.

## Estimator contract

For a directional comparison `KL(P || Q)`, the evaluator:

1. samples a continuation from `P`;
2. teacher-forces `Q` on exactly the same token history;
3. computes the categorical KL at every shared prefix;
4. sums the terms using the autoregressive chain rule.

This is a Monte Carlo estimator of finite-horizon joint continuation KL. It is
not the same as comparing logits along two independently sampled paths. If an
adapter reports a sampling temperature, logits are divided by that temperature
before KL computation so the density matches the distribution that generated
the trajectory. Deterministic decoding is rejected because it does not define
the required finite-support stochastic policy KL.

The first real-checkpoint experiment is an integration contract only. A later,
frozen phenomenon gate must show that multi-step geometry predicts a meaningful
failure that next-token KL and local coordinate baselines miss before any
amortized BCG method is built.
