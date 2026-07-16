# Coupled Group Exploration Audit

This directory records the zero-GPU gate for provisional **Stratified Gumbel
Group Exploration (SGGE)**.

The idea couples random numbers across rollouts in one GRPO group while keeping
each trajectory's marginal sampling law unchanged. In principle this can reduce
duplicate trajectories and increase the probability that a sparse binary reward
group contains both successes and failures.

The exact gate finds a binding conflict: the same cross-rollout dependence that
improves group coverage biases leave-one-out and group-centered policy-gradient
baselines. Removing the group-relative baseline restores unbiasedness, but then
the surviving method is a direct application of randomized quasi-Monte Carlo or
antithetic sampling, with Arithmetic Sampling already providing a language-model
decoding construction.

SGGE is therefore stopped as a primary AAAI method. Its exact gate remains a
required control for future correlated-rollout proposals.

`AUDIT_RESULTS.json` stores the frozen `p=0.3` values used in the decision.

No GPU run is authorized.
