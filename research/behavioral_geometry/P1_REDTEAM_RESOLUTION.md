# P1 Red-Team Resolution

**Date:** 2026-07-16  
**Reviews:** independent source/estimand, mathematics/statistics, and
collision/AAAI-positioning audits  
**Reviewed artifacts:** blocked first draft and revised P1 v2 design snapshot  
**Verdict on reviewed draft:** `BLOCK`  
**Verdict on revised v2:** `PASS-DESIGN / BLOCK-EXECUTION`

This is a post-P0 resolution record. It does not alter the immutable P0 hashes
or authorize checkpoint measurements.

## 1. Unanimous blockers in the reviewed draft

All three reviews independently identified the following fatal defects:

1. **The reference pair was inconsistent.** Family-A and Family-B deltas were
   not always applied at the action from which they were constructed, while
   every Fisher and finite label was written around `z_0`.
2. **The last-soft-action interface did not guarantee a visible-only suffix.**
   Perturbing that action can change the next latent-end proxy, so forced common
   exit would estimate a clamped counterfactual rather than deployment behavior.
3. **The last-action framing removed the mechanism of interest.** Once later
   latent recursion is removed, the method is largely FishBack-style pullback
   geometry accumulated over an ordinary autoregressive suffix.
4. **Hard top-k/top-p forward KL was silently treated as regular and finite.** A
   candidate that loses any reference-supported token has infinite forward KL;
   the local Fisher is valid only inside a stable support cell.
5. **The draft implied a full/projected Fisher without defining a projection.**
   P1 only needs scalar directional risk on the frozen natural update bank.
6. **The problem split made test updates impossible.** Construction, Fisher,
   and labels must use disjoint RNG pools inside each calibration/test problem,
   not disjoint problem sets.
7. **False-safe labels used the wrong confidence direction.** `UCB(d)>tau`
   means not certified safe; confirmed unsafe requires `LCB(d)>tau`.
8. **Power, ratio, oracle stability, and compute caps were incomplete.** The
   old sample formula omitted target power, ratio-of-means and denominator
   stability were undefined, and no numeric compute ceiling existed.

## 2. Binding design decisions

The revised draft makes these decisions before outcome inspection:

- use the official no-noise deterministic evaluation policy as the latent
  reference process;
- intervene at the first structurally eligible nonterminal soft action;
- recursively recompute every later latent mixture, proxy, exit, and cache;
- execute a proposed joint action only after the source's append-then-stop
  check: generic token/string/length stops are absorbing structural endpoints,
  a surviving proxy 524 executes hard one-hot `E_524`, and only a surviving
  nonterminal proxy consumes `z`;
- use one common deterministic action `z_0` for every candidate pair;
- make source exploration actions Family A;
- make a source-surrogate gradient pulled through the frozen LM head to a
  hidden-state counterfactual Family B, explicitly not an optimizer update;
- use temperature-only full-softmax visible decoding as the primary smooth
  analysis law;
- retain exact top-30/top-p decoding as a support-aware extended-real
  sensitivity;
- estimate directional `q_H(v)` only;
- make late-latent, ordinary-activation, support-churn, exit-margin, stability,
  and matched-time direct-MC controls mandatory; and
- require the effect to survive in a trace-stable population and not be
  explained by discrete branch changes.

The mathematics review noted that last-action intervention is locally cleaner.
The positioning and source reviews showed that it either changes exit anyway or
removes the recursive mechanism. The project therefore accepts the harder
early-recursive design and pays for an explicit trace-stability gate. If this
gate leaves inadequate natural-update coverage, the strong idea is killed
rather than reverting to the weaker last-action headline.

## 3. Platform decision

| Platform | Decision | Role |
|---|---|---|
| Coconut GPT-2 | `PASS-ENGINEERING` | identity, same-history, estimator, and accounting tests only |
| Latent-GRPO 1B | `CONDITIONAL` | preferred P1 candidate after recursive/source/trace/compute gates |
| SofT-GRPO 1.5B | `CONDITIONAL-ROBUSTNESS` | same-mechanism robustness, not distinct replication |
| CODI GPT-2 | `CONDITIONAL-PRIORITY` | preferred low-compute mechanism-distinct candidate if tail and adapter pass |
| SWITCH Qwen3-8B | `DEFER` | stronger hidden-recurrence replication only after small-model survival |

## 4. Statistics decisions

- top-level inference unit: near-duplicate-safe problem group;
- primary label: post-H8 full-support expected chain-rule KL;
- primary predictor: fixed `H_cert` directional expected trajectory Fisher;
- confidence: nested problem/history bootstrap;
- unsafe truth: family-wide simultaneous max-T `LCB(d)>tau_d`;
- predictor comparison: exact common acceptance coverage;
- indeterminate labels: the long screen must still win under the globally
  consistent assignment that maximizes its false-safe disadvantage;
- semantic Fisher: an online reference-filtration stopping time only, never a
  hindsight suffix parser;
- binding cheap comparator: uniquely defined `best-simple_aug`, with direct-MC
  and adapted-metric predictors retained as separate IUT competitors;
- consequence inference: paired contrasts `mu_long-r*mu_simple` with a positive
  simple-risk gate, with screen histories resampled before candidate selection;
  ratio of means is descriptive;
- utility: paired non-inferiority confidence bound, not point estimate;
- family/endpoint claim: intersection-union test; and
- history count: smallest nested-prefix `R<=32` passing two-independent-
  replicate stability in at least 90% of calibration contexts.

## 5. Unresolved blockers

The revised protocol remains non-executable until all are closed:

1. exact source-equivalent recursive adapter;
2. tokenizer and probability-row assertions;
3. exact full-support and hard-truncated sample/force implementations;
4. recursive JVP and trace/exit tests;
5. semantic-tail and trace-stable coverage;
6. independent review, fake-model validation, and freeze of the drafted exact
   adapted-metric baseline contract;
7. local 8-GB microbenchmark and numeric compute cap;
8. endpoint-wide power and utility sample cap; and
9. independent PASS reviews of the final frozen candidate.

## 6. Collision addendum

The binding additive record is
[`P1_COLLISION_ADDENDUM.md`](P1_COLLISION_ADDENDUM.md). It formally subtracts
[**Weight Updates as Activation Shifts: A Principled Framework for
Steering**](https://arxiv.org/abs/2603.00425): its first-order
weight/activation equivalence removes the novelty available to a generic later
parameter-bridge theorem. The paper may use such a bridge only as empirical
optimizer/training validation. The addendum also freezes the residual boundary
against FishBack, latent causal dynamics, recurrent stability, and Certified
World Models without modifying any P0 file.

## 7. Current authorization boundary

Authorized now: document closure, exact baseline review, CPU/fake-model test
design, and source-only adapter work that does not expose P1 risk outcomes.

Not authorized: real-checkpoint measurements or calibration, held-out test, GPU
rental, adaptive-certificate implementation, or training. Those require the
separate staged gates `GO-CHECKPOINT-PREFLIGHT`, `GO-CAL`,
`GO-P1a-HELDOUT`, and `GO-P1b-HELDOUT` in that order.
