# Selection-Complete Latent Policy Optimization

This directory tests a narrow methodological claim about sparse soft-thought
policies: once Gumbel-perturbed logits are ranked and only the top-k entries are
executed, the selected scores are order statistics rather than independent
Gumbel variables. A likelihood that multiplies their marginal Gumbel densities
but omits the selection event is not a normalized density of the stored action.

The working method name is **Selection-Complete Latent Policy Optimization
(SC-LPO)**. The induced executed-action law is called **Top-K Concrete (TKC)**.

## Exact executed-action density

Let

```text
A_i = l_i + sigma G_i,       G_i ~ Gumbel(0, 1),
S = (s_1, ..., s_K),         A_s1 > ... > A_sK,
q_r = exp(A_sr / tau) / sum_m exp(A_sm / tau).
```

Define `eta = tau / sigma` and `alpha_i = exp(l_i / sigma)`. With density
measured over the ordered support and the first `K-1` simplex coordinates,

```text
p(S, q)
  = Gamma(K) eta^(K-1) product_r[alpha_sr q_r^(-eta-1)]
    / [sum_r alpha_sr q_r^(-eta)
       + q_K^(-eta) sum_(j not in S) alpha_j]^K.
```

This law has three useful checks:

- `K=1` is the categorical law `softmax(l / sigma)`;
- `K=V` is the ordinary Concrete distribution;
- adding a common constant to every logit leaves the density unchanged.

The exact density of the labeled selected scores is also implemented. It is the
product of selected Gumbel densities multiplied by the event that every
unselected score is below the K-th selected score.

## Scope boundary

The formula above assumes independent, unbounded Gumbel noise and a fixed
candidate vocabulary. The inspected official Latent-GRPO implementation at
commit `c0994fb781a2d180662bb522d8ff3e8638dcf56d` differs in three ways:

1. it constructs a policy-dependent top-p candidate set;
2. it clips Gumbel noise to `[-1.5, 3]`, creating boundary atoms;
3. it averages marginal selected-Gumbel log densities and applies an
   advantage-dependent straight-through gradient, while omitting the top-k
   selection event.

Therefore the current TKC implementation is not presented as the exact density
of that unchanged sampler. A clean method must either remove clipping and make
candidate support explicit, or derive and implement the full mixed
discrete-continuous clipped law.

## Evidence status

The frozen v1 CPU gate is recorded as **fail**, because one of five random
policy drifts produced only `0.0110` mean-ratio bias for the selection-omitting
surrogate, below the preregistered `0.03` minimum. Thresholds were not changed.

Other diagnostics support continuing the audit rather than claiming a method:

- known-distribution boundary error: `1.71e-13`;
- exact importance-ratio mean error: at most `0.00328`;
- exact score mean norm: at most `0.01275`;
- selection-omitting score mean norm: at least `0.8165`;
- exact versus official-style PPO clipping disagreement: `38.72%` on average;
- omitted-selection log-ratio correction RMS: at least `0.1516`.

An independent one-dimensional quadrature for `V=3, K=2` normalizes the density
to floating-point precision when the boundary is smooth.

The source-faithful official-default replay is also **fail** under its frozen
all-check rule. The one-sided shift changed policy log-ratios by only
`0.0052-0.0114` RMS, and one seed's clean selection correction was `0.0813`,
below the preregistered `0.1` minimum. Strong diagnostics instead came from
clipping, mean reduction, and support changes:

- `27.9%-35.1%` of selected components came from the upper clipping atom;
- clipping changed the complete ordered top-k support on at least `98.65%` of
  paired samples, although mean support overlap remained about `82%-84%`;
- `sum/K` versus joint-sum ratios changed PPO clipping on `36.53%` of samples;
- dynamic top-p support invalidated `7.69%` of old-policy actions on average;
- exact versus official-style clipping decisions differed on `37.59%` of clean
  unbounded samples.

The released Latent-GRPO paper explicitly describes its one-sided objective as
a surrogate rather than an exact probability density. The research claim must
therefore concern the value of sampler-likelihood consistency, not an
undisclosed algebraic mistake in that paper.

The subsequent FTK-PO factorized trust-region hypothesis also failed its frozen
gate. Although the joint law exactly decomposes into an ordered Plackett-Luce
support mass and a conditional-mixture density, only `8.89%` of 135 matched-KL
scenarios had two material KL components. Ordered support carried a median
`99.84%` of joint KL, and hidden component violations averaged only `2.03%`.
This stops the two-budget method while preserving the decomposition as a valid
theory and diagnostic result.

## Reproduction

```powershell
& .\_research_env\Scripts\python.exe -m unittest tests.test_topk_concrete_densities
& .\_research_env\Scripts\python.exe -m research.topk_concrete.toy_gate `
  --output artifacts\topk_concrete\topk_concrete_toy_v1.json
& .\_research_env\Scripts\python.exe -m research.topk_concrete.official_replay `
  --output artifacts\topk_concrete\official_replay_v1.json
& .\_research_env\Scripts\python.exe -m research.topk_concrete.factorized_trust_gate `
  --output artifacts\topk_concrete\factorized_trust_gate_v1.json
```

The gate commands intentionally exit with status 1 while any frozen check
fails. Their JSON artifacts are still written atomically.
