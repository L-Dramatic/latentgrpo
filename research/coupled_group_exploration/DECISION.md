# SGGE Gate Decision

Decision date: 2026-07-16

## Verdict: KILL as a primary method direction

| Gate | Result | Reason |
|---|---|---|
| C1: exact trajectory marginals | PASS | Randomized stratification/antithetic uniforms can preserve each member's conditional uniform law. |
| C2: useful group coverage | PASS | The exact Bernoulli gate raises mixed-reward probability and can reduce raw-estimator variance. |
| C3: valid group-relative gradient | FAIL | Other-member rewards are correlated with the current score; the exact leave-one-out estimator is biased. |
| C4: independent novelty | FAIL | The unbiased fallback is covered by Arithmetic Sampling, RQMC policy gradients, and generic antithetic sampling. |

## Why the apparent signal is dangerous

Correlated sampling produces more nonzero-advantage groups, but that does not
by itself mean more accurate learning signal. In the exact gate the increased
mixed-group rate and positive gradient bias are two consequences of the same
dependence. A model experiment could therefore look promising while optimizing
a systematically rescaled or distorted objective.

## Preserved result

Keep the exact gate as a mandatory audit for any future proposal using coupled,
without-replacement, diverse, or low-discrepancy rollouts with GRPO/RLOO
baselines. Such work must report both per-trajectory marginal validity and
cross-trajectory score-baseline terms.

## Forbidden continuation

- No GPU or checkpoint inference for SGGE.
- No use of mixed-reward group frequency as a standalone success metric.
- No claim of unbiasedness from marginal on-policy sampling alone.
- No rebranding of Arithmetic Sampling or RQMC as a latent-GRPO method.
