# SWITCH C2 Scientific Gate

Frozen on 2026-07-15 before downloading the Qwen3-8B base weights or inspecting
any paper-final checkpoint output. This document governs the only experiment
that can revive FCTR as a method direction.

## 1. Question and evidence boundary

C2 asks two conjunctive questions:

1. Does a moderate exact reparameterization change a coordinate-Euclidean
   latent update enough to alter the deployed visible continuation after
   scalar retuning?
2. Does a 32-token continuation pullback metric predict and control strictly
   later behavior better than latent H1/H3/H4, visible next-token FishBack,
   V3/V8, a semantic-prefix metric, activation whitening, and exact scalar KL
   retuning?

Coordinate invariance by itself is not a result. Every pullback Fisher baseline
is invariant under an exact chart. The candidate earns a method claim only if
its longer horizon produces an operational gain unavailable to simpler
invariant baselines.

C2 is a falsification and mechanism gate, not training. A pass authorizes an
efficient estimator. Only a later matched-compute estimator gate can authorize
a training pilot. A failure is retained and stops the FCTR method claim.

## 2. Pinned execution object

- Official source: `LARK-AI-Lab/SWITCH` at
  `d8d97cdc6276fcfa6e48f6a6b19ce472c7b87fcd`.
- Base: `Qwen/Qwen3-8B` at
  `b968826d9c46dd6066d109eabc6255188de91218`.
- Adapter: `LARK-Lab/SWITCH-Phase3-GRPO-LoRA-Qwen3-8B` at
  `246fee75d774c02a110ea8608ac841a916dd5d35`.
- Dataset: `HuggingFaceH4/MATH-500` at
  `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`.
- Scientific config:
  `configs/switch_c2_scientific_gate_v1.json`.
- Identity config canonical SHA-256:
  `e2df1ad9f577c8d44c97e8738575ea383ae1e75287206792aca42c56e8da09b5`.
- Scientific config canonical SHA-256:
  `603629a1eab5bc0c68af70fd160904e1dda30c388a5ec4fae312585e46202c9d`.

The released latent driver, hidden recurrence, minimum dwell of four, tokenizer
IDs, forced `</swi>` insertion, and all weight hashes are source-pinned. The
latent-step logits are treated only as mechanistic metric baselines. They are
not mislabeled as sampled GRPO actions because the released source force-appends
`</swi>` with `sampled_mask=0`.

Before C2 can run, the source preflight, Coconut C1c integration gate, and the
eight-prompt paper-final checkpoint identity smoke must all pass. The identity
smoke requires exact generated IDs, latent metadata, forward counts, and
identity-hook hidden states, with at least two of eight prompts entering a
latent block.

## 3. Prompt assignment

All 500 MATH-500 rows have a fixed order given by

```text
sha256(71531 NUL unique_id NUL problem).
```

The model scans this order once. A prompt is eligible only when its first
latent block:

- uses the paper-final minimum dwell of four;
- exits naturally;
- leaves at least 64 ordinary visible tokens;
- contains no `<swi>`, `</swi>`, or EOS within that 64-token horizon; and
- is not accepted from a max-token-truncated run.

The first 16 eligible prompts are calibration; the next 32 are test. Accuracy,
chart effects, and metric outcomes cannot influence eligibility. The test set
must contain at least five MATH subjects and four difficulty levels.

## 4. Intervention and non-leakage split

Only the first hidden-state input consumed after the first natural `<swi>` is
changed. All later latent steps and visible histories follow the factual
recurrent computation. The zero intervention must reproduce the audited run
exactly.

Visible tokens 1-8 after `</swi>` define the local update objective. Tokens
9-64 are a secondary held-out continuation. Tokens 33-64 form the strict
primary extrapolation interval, so the V32 candidate never evaluates itself on
the tokens used to construct its metric. V64 sees the primary interval and is
therefore an oracle only.

All KL quantities use the unperturbed greedy token history for teacher forcing.
Separate free rollouts measure token and termination changes; they do not
replace the exact same-history KL diagnostic. Answer correctness is reserved
for a later training/evaluation gate and is not fabricated from this local
intervention study.

## 5. Four-dimensional consequential subspace

For each prompt, compute the full 4096-dimensional gradient of mean factual
log likelihood on visible tokens 1-8. Its normalized direction is axis one.
Three seeded Gaussian vectors are projected off that direction and
QR-orthonormalized. This avoids a random subspace that could miss the actual
update while keeping exact Jacobians tractable.

The intervention is

```text
h'(z) = h + B z,
```

where `B` has four orthonormal columns. Charts use `u = A z`. Identity and a
seeded orthogonal chart are controls; seeded affine charts have condition
numbers 4 and 12. Mapped-back coordinate-Euclidean gradients are expected to
differ under anisotropic charts. Tensor metrics and their trust-region updates
must transport back to the same native vector within frozen numerical
tolerances.

## 6. Metrics and simple baselines

For a factual shared history, each prefix metric is the accumulated categorical
pullback Fisher

```text
G_H = sum_{t=1}^H J_t^T (diag(p_t) - p_t p_t^T) J_t.
```

The frozen metric family is:

- latent exit-logit prefixes L1, L3, and L4;
- visible V1, the FishBack-style next-token baseline;
- visible V3 and V8;
- the first semantic boundary between 3 and 32 visible tokens;
- V32, the sole candidate method;
- V64, an ineligible oracle;
- calibration-only diagonal activation whitening.

Singular metrics receive a tensorial ridge based on the whitening metric. Its
scale is `0.01 * trace(W^-1 G) / 4`, which is invariant under congruent chart
transport. No coordinate identity ridge is allowed.

## 7. Probe and update comparisons

Twelve seeded paired native probe directions are evaluated at one global
relative-hidden-L2 scale selected on calibration from the frozen grid. Each
metric predicts risk quadratically. Actual primary risk is exact summed KL on
tokens 33-64. Within-prompt Spearman and top-20-percent risk recall are then
bootstrapped by prompt.

For the update comparison, each direction is normalized to the same native
first-order gain on tokens 1-8. Calibration chooses one global gain from the
frozen grid and never revisits it on test. The ordinary-sequence-KL control may
retune only the scalar of a coordinate-Euclidean direction, using bisection to
match exact V8 KL; it cannot change direction.

The simple metric maximum is selected once on calibration from whitening,
L1/L3/L4, V1/V3/V8, and semantic prefix. V32 is compared against that selected
maximum on the untouched 32-prompt test set. Exact V8 scalar retuning is a
separate chart-consequence control because it changes only update magnitude and
does not define a probe-risk ranking metric.

## 8. Statistical decision

All intervals use 10,000 prompt-level bootstrap resamples. C2 passes only if
all identity, derivative, transport, precision, coverage, and diversity checks
pass and all of the following are true on test:

1. The condition-12 coordinate-Euclidean mapped direction discrepancy has
   median at least `0.25`.
2. At least `25%` of prompts show a free-rollout token change or strict-holdout
   symmetric KL of at least `1e-4` between identity-chart and condition-12
   coordinate-Euclidean updates.
3. V32 exceeds the best simple baseline in probe-risk Spearman by at least
   `0.10`, with 95% interval lower bound at least `0.03`.
4. V32 improves top-risk recall by at least `0.10`.
5. At matched predicted objective gain, the V32-to-best-simple strict-holdout
   KL ratio is at most `0.90`, with interval upper bound below `1.0`.
6. V32 does not reduce strict-holdout factual utility relative to the best
   simple baseline; the difference interval lower bound is nonnegative.
7. V32's Spearman correlation is within `0.05` of the V64 oracle.

If V3, V8, semantic prefix, whitening, or FishBack is within `0.03` of V32 on
the two primary advantages, the simpler metric wins and FCTR is not claimed.
If exact V8 scalar retuning removes the chart consequence, item 2 fails and the
method claim is likewise stopped.

## 9. Stop and continuation rules

- Identity-smoke failure stops before scientific measurement and triggers only
  source-equivalence debugging.
- Insufficient eligible prompts or diversity stops C2 without replacement
  sampling, threshold changes, or a stronger chart.
- Failure of chart consequence rules stops the reparameterization method paper.
- Failure of V32 superiority converts the result into negative analysis and
  forbids FCTR training.
- A clean pass authorizes implementation of a matched-compute V32 estimator.
  That estimator must preserve the C2 ordering advantage before any GRPO
  training run is authorized.

Every run records model calls, backward calls, visible tokens, wall time, peak
GPU memory, package versions, config hash, checkpoint revisions, and artifacts.
The long stages use a read-only detached KV-prefix snapshot. DynamicCache
restoration and repeated-logit equivalence were tested before execution; each
journal is bound to both config and implementation hashes.
