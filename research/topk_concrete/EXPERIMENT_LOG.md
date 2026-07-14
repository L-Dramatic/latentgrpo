# Top-K Concrete Experiment Log

## 2026-07-14: frozen CPU gate v1

### Question

Does completing the top-k selection likelihood expose a reproducible
normalization defect and an operationally material PPO-ratio difference before
any model training is attempted?

### Frozen protocol

The configuration in `configs/topk_concrete_toy_v1.json` was written before the
reported run. It uses five seeds, `V=32`, `K=5`, 131,072 ratio samples per seed,
32,768 score samples per seed, temperature `0.7`, and logit drift scale `0.35`.
All eight checks are conjunctive. No threshold may be relaxed after observing a
result.

### Mathematical validation

- `K=1` matches the categorical law.
- `K=V` matches PyTorch's `RelaxedOneHotCategorical` density.
- common-logit-shift error is at most `2.66e-13`.
- independent Gauss-Legendre quadrature for `V=3, K=2` gives total mass 1 to
  floating-point precision for a boundary-smooth parameter setting.
- six targeted density tests pass.

### Result

**Status: fail by the frozen all-checks rule.**

Seven of eight checks passed:

| Diagnostic | Observed | Frozen requirement | Result |
|---|---:|---:|---|
| Known boundary-law error | `1.71e-13` | at most `1e-8` | pass |
| Shift-invariance error | `2.66e-13` | at most `1e-10` | pass |
| Exact ratio mean error | `0.00328` max | at most `0.05` | pass |
| Exact score mean norm | `0.01275` max | at most `0.08` | pass |
| Naive ratio mean bias | `0.01104` min | at least `0.03` | **fail** |
| Naive score mean norm | `0.8165` min | at least `0.10` | pass |
| PPO clip disagreement | `38.72%` mean | at least `5%` | pass |
| Selection correction RMS | `0.1516` min | at least `0.10` | pass |

The five naive ratio-mean biases were `0.3419`, `0.0481`, `0.0110`, `0.2307`,
and `0.0953`. A misspecified density can have an importance-ratio mean close to
one for a particular drift through cancellation, so the failed minimum-over-
directions condition is not a mathematical necessity. This observation does
not retroactively change the gate outcome.

The selection-omitting score is a more direct local normalization diagnostic:
its mean norm was `0.82-0.89`, versus `0.011-0.013` for both exact densities.
The official-style mean reduction also shrank policy-gradient norms by roughly
five to eight times while retaining high gradient cosine in this synthetic
reward family.

### Decision

Do not call v1 a passed method gate and do not begin large training. Continue
only with:

1. an independent official-code replay that reproduces the stored action and
   separates top-k selection, mean reduction, top-p support, clipping atoms,
   and the straight-through gradient;
2. a sampler-validity decision between an unclipped, support-explicit TKC
   policy and a substantially harder exact clipped law;
3. a real-checkpoint ratio audit frozen before labels or downstream rewards are
   inspected.

Stop the direction if the official replay shows that the corrected likelihood
does not materially change ratios or gradients after accounting for the actual
candidate set and clipping, or if a close prior derivation/method collision is
found.

Artifact: `artifacts/topk_concrete/topk_concrete_toy_v1.json`.

## 2026-07-14: official-default source replay v1

### Question

Under the released Latent-GRPO defaults, which discrepancies are operationally
material: post-noise top-k selection, clipped Gumbel atoms, the uniform
one-sided shift, dynamic top-p support, mean reduction, or the conditional
straight-through gradient?

### Frozen protocol

`configs/official_replay_v1.json` was frozen before the reported run. It uses
the released `top_p=0.95`, `K=10`, Gumbel-Softmax temperature `1.0`, noise scale
`1.0`, clipping interval `[-1.5, 3]`, and one-sided shift. Five synthetic logit
profiles each use 32,768 paired draws. The same raw Gumbels drive the unchanged
official sampler and an unbounded clean sampler.

### Result

**Status: fail by the frozen all-checks rule.** Five of seven checks passed.

| Diagnostic | Observed | Frozen requirement | Result |
|---|---:|---:|---|
| Clean exact ratio mean error | `0.00410` max | at most `0.05` | pass |
| Clean selection correction RMS | `0.08127` min | at least `0.10` | **fail** |
| Clean exact/official clip disagreement | `37.59%` mean | at least `5%` | pass |
| Selected upper clipping atom | `27.89%` min | at least `10%` | pass |
| Clipped/unclipped support change | `98.65%` min | at least `10%` | pass |
| One-sided shift ratio RMS | `0.00518` min | at least `0.10` | **fail** |
| Mean/sum clip disagreement | `36.53%` mean | at least `5%` | pass |

Additional diagnostics:

- dynamic current top-p support rejected `7.69%` of old clean actions on
  average;
- candidate-set Jaccard similarity was `0.917-0.946`;
- clipped/unclipped support overlap was `0.818-0.835`, despite almost every full
  ordered support changing;
- the straight-through branch triggered on exactly zero default one-sided
  components, so its gradient matched the standard surrogate in this replay;
- the official mean-ratio clip rate was almost zero, while the corresponding
  joint-sum clip rate was `33.4%-39.6%`.

The one-sided shift is behaviorally null because it adds the same scalar to all
selected scores. In this local-drift replay, most of its effect also cancels in
old/current likelihood ratios. It should not be a headline defect. Likewise,
the selection correction is real but not uniformly above the preregistered
materiality threshold.

### SofT-GRPO versus Latent-GRPO boundary

The inspected SofT-GRPO implementation at commit
`8d3c61380b15c3400818da5ce41c62c293a1bfb4` chooses top-k components from the
renormalized model probabilities before applying Gumbel noise. Its support is
therefore fixed conditional on the logits, and no post-noise top-k selection
event is missing from the auxiliary product density. It still clips rollout
Gumbels, clamps reconstructed margins, masks components, and averages their log
scores.

The inspected Latent-GRPO implementation at commit
`c0994fb781a2d180662bb522d8ff3e8638dcf56d` instead perturbs every top-p
candidate and selects top-k after noise. The selection-completion term and TKC
joint law apply specifically to this changed sampler. The Latent-GRPO paper
explicitly calls the one-sided objective a surrogate rather than an exact
density, so the appropriate scientific question is whether a consistent action
law performs better, not whether the authors concealed a likelihood caveat.

### Decision

Stop the selection-only and one-sided-error framing. Preserve the exact TKC law
as a component of a narrower **sampler-likelihood-consistent latent policy**.
Before downloading checkpoints or training, complete the prior-art audit and
freeze a real-logit study that focuses on the three surviving effects:

1. clipped boundary atoms and their action distortion;
2. dynamic candidate-support mismatch;
3. joint-density versus `sum/K` PPO scaling under matched trust-region tuning.

Artifact: `artifacts/topk_concrete/official_replay_v1.json`.

## 2026-07-14: FTK-PO frozen factor-separability gate v1

### Question

At matched joint KL, are ordered-support drift and conditional-mixture drift
both broadly material, sufficiently distinct, and sufficiently hidden by a
single joint ratio to justify a two-budget trust-region method?

### Frozen protocol

The configuration in `configs/factorized_trust_gate_v1.json` was written before
the reported run. Its SHA-256 is
`50bb47b36249dacb04e877f1a0326a33ee868c49a8647e82dd1ec916bf1fcdb5`.
It crosses three values of `K`, three logit concentrations, three drift modes,
and five independent seeds, for 135 scenarios. Every drift is calibrated on a
separate sample to target joint KL `0.03`; 8,192 fresh actions measure each
scenario. All nine checks are conjunctive.

### Result

**Status: fail.** Five of nine checks passed.

| Diagnostic | Observed | Frozen requirement | Result |
|---|---:|---:|---|
| Calibration measurement error | `0.7358` max | at most `0.30` | **fail** |
| Exact factorization error | `1.42e-14` max | at most `1e-10` | pass |
| KL chain residual | `4.64e-17` max | at most `1e-10` | pass |
| Exact ratio mean error | `0.00943` max | at most `0.06` | pass |
| Minimum component KL | `-0.000190` | at least `-0.005` | pass |
| Both components material | `8.89%` | at least `75%` | **fail** |
| Hidden component violations | `2.03%` mean; `22.96%` of scenarios above `3%` | at least `5%` mean and `60%` of scenarios | **fail** |
| Opposite-sign components | `49.72%` mean | at least `25%` | pass |
| Support-share interdecile range | `0.0925` | at least `0.15` | **fail** |

The decisive result is component imbalance, not the noisy worst-case
calibration estimate. Ordered-support KL has median share `99.84%`. Mean support
shares by `K` are `98.89%`, `97.61%`, and `94.80%` for `K=2`, `K=5`, and
`K=10`. Mean shares are `99.99%`, `99.57%`, and `91.75%` from diffuse to highly
concentrated logits. Only a post hoc narrow, highly concentrated subset supports
the two-budget premise.

### Decision

Activate the preregistered stop rule. Do not run the collision audit as a method
priority, download checkpoints, or begin training for FTK-PO. Preserve the
exact support/conditional decomposition and the empirical support-dominance
finding as analysis assets. Do not reinterpret opposite-sign sample ratios as
method evidence: they rarely create a material hidden trust-region violation.

Artifact: `artifacts/topk_concrete/factorized_trust_gate_v1.json`.
