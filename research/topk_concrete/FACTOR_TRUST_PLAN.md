# FTK-PO Evidence Plan

## Research question

Does an exact sparse latent action contain operationally distinct support and
conditional-mixture updates that justify separate trust-region control, beyond
using a normalized joint likelihood with one KL budget?

## Method contract

The candidate method must use the exact joint Top-K Concrete importance ratio
for the reward surrogate. It may place separate dual KL constraints on the
Plackett-Luce support law and the conditional weight law. It must not use
`sum/K` as a likelihood, claim that independent component clipping is exact, or
retain clipped random variables while evaluating an unbounded-Gumbel density.

## Evidence ladder

### Gate A: mathematical validity

- joint log density equals support log mass plus conditional log density;
- ordered support masses sum to one in an enumerable case;
- integrating the joint density over each fixed support returns its
  Plackett-Luce mass;
- Monte Carlo component KL estimates obey the exact chain rule within sampling
  error.

The first three checks are implemented and pass. They establish validity, not
method usefulness.

### Gate B: frozen synthetic separability diagnostic

Use full-vocabulary, unbounded-Gumbel actions. Sweep broad, prespecified values
of `K`, temperature, logit concentration, drift direction, and five held-out
seeds. Calibrate each policy drift to the same target joint KL before measuring:

- support KL, conditional-mixture KL, and their shares of joint KL;
- correlation and sign cancellation between component log ratios;
- hidden component violations, where the joint log ratio lies inside the PPO
  band but at least one component lies outside;
- whether both components remain material across regimes rather than in one
  cherry-picked setting;
- Monte Carlo ratio normalization and KL-chain residuals.

All metrics and thresholds live in a versioned JSON file written before the
first reported run. The all-check status cannot be changed after seeing output.

### Gate C: collision audit

Search exact and adjacent terms covering Top-K Concrete/truncated Concrete,
Plackett-Luce plus continuous weights, compound or hierarchical PPO clipping,
factorized trust regions, multi-discrete PPO, and KL decomposition for
structured policies. Record exact overlaps and non-overlaps. A direct collision
stops the method claim even if Gate B passes.

### Gate D: real-checkpoint logit audit

Freeze prompts, layers/latent steps, checkpoints, perturbations, metrics, and
thresholds before inspecting downstream rewards. Use at least the public SFT
and GRPO checkpoints. Confirm that factor shares and hidden violations persist
on real logits and are not artifacts of iid Gaussian vectors.

### Gate E: matched-compute optimization pilot

Compare four systems:

1. released latent surrogate;
2. exact joint ratio with one joint KL budget;
3. exact joint ratio with two fixed component budgets;
4. FTK-PO with preregistered dual-budget adaptation.

Match model initialization, generated tokens, prompts, reward calls, effective
batch size, wall-clock reporting, and total accelerator hours. The decisive
comparison is (4) versus (2), not (4) versus the released surrogate.

## Stop rules

- Stop FTK-PO if Gate B shows one negligible component or less than a material
  rate of hidden component drift across held-out regimes.
- Stop before checkpoint download if a direct prior-art collision is found.
- Stop before training if Gate D fails.
- Downgrade to a technical analysis if Gate E does not beat exact-joint
  single-budget control under matched conditions.

## Current state

Gate A passes the targeted density tests. Gate B was executed unchanged over
135 scenarios and **failed**. Only `8.89%` of scenarios had two material KL
components; ordered-support KL had a `99.84%` median share; hidden component
violations averaged `2.03%`. The stop rule is active: FTK-PO will not proceed to
checkpoint download or training. The frozen artifact is
`artifacts/topk_concrete/factorized_trust_gate_v1.json`.
