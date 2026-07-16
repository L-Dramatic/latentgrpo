# Exact Bernoulli Gate

## Setup

Consider a one-step Bernoulli policy `A ~ Bernoulli(p)`, reward `R=A`, and logit
score `S=A-p`. The true policy gradient is `p(1-p)`.

For a two-rollout group, compare:

- IID: independent uniforms `U1, U2`;
- antithetic: `U2 = 1-U1`, with `Ai = 1[Ui < p]`.

Each antithetic member is exactly Bernoulli(p). Fresh independent antithetic
pairs at every autoregressive step would likewise preserve each trajectory's
conditional marginal law, even after the members' histories diverge.

## Coverage result

The probability that a group contains one success and one failure is

- IID: `2p(1-p)`;
- antithetic: `2 min(p, 1-p)`.

Thus coupling genuinely produces more reward-varying groups except at the
deterministic limits.

## Gradient result

The raw score estimator

`0.5 * (R1 S1 + R2 S2)`

remains unbiased under either joint law because it uses only each trajectory's
own reward and score. Antithetic coupling can reduce its variance.

The two-member leave-one-out/group-centered estimator is

`0.5 * ((R1-R2) S1 + (R2-R1) S2)`.

Under IID sampling its expectation is `p(1-p)`. Under antithetic sampling its
expectation is `min(p, 1-p)`. The relative bias is therefore

- `p/(p(1-p)) - 1 = p/(1-p)` for `p <= 0.5`;
- `(1-p)/(p(1-p)) - 1 = (1-p)/p` for `p >= 0.5`.

At `p=0.3`, the true gradient and IID leave-one-out expectation are `0.21`, but
the antithetic leave-one-out expectation is `0.30`, a `42.86%` positive bias.
The informative-group probability rises from `0.42` to `0.60`; the same
dependence causes both effects.

## Interpretation

Marginal correctness is insufficient. A baseline built from another coupled
rollout is correlated with the current trajectory score, so
`E[B_other * S_current]` need not vanish. Standard group reward normalization
adds further dependence and does not repair this cross term.

One can restore unbiasedness by using raw rewards, a baseline independent of
the coupled group, or an explicit joint-law correction. The first two abandon
the claimed GRPO group efficiency; the last is intractable for the proposed
autoregressive Gumbel copula. This triggers the frozen C3 kill condition.
