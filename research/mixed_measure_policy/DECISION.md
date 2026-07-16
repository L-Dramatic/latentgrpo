# MMLPO Gate Decision

Decision date: 2026-07-16

## Verdict: KILL as the primary method direction

| Gate | Result | Evidence |
|---|---|---|
| C1: mixed source-faithful law | PASS | Pinned sampler has dynamic support, ordered Top-K, continuous interiors, clipping atoms, mixture execution, and a proxy. |
| C2: non-CAPG ratio failure | PASS | The exact two-token counterexample gives policy-dependent action atoms and generic failure of mutual absolute continuity. |
| C3: practical black-box optimizer | FAIL | Valid alternatives require differentiable simulation, explicit boundary/weak-derivative constructions, or costly counterfactual/finite-difference rollouts. No sampler-preserving near-GRPO estimator survived. |
| C4: method novelty | FAIL | CAPG, HPO, LEPO, boundary-corrected reparameterization, and weak-derivative policy gradients cover the generic method families. |

## What remains publishable

The moving-atom theorem is a useful result for a policy-contract audit:

- it explains why selected interior Gumbel densities cannot be called the full
  executed-action likelihood;
- it separates fixed clipping atoms from policy-dependent action atoms;
- it supplies an exact adversarial test for future latent-policy objectives.

This material can strengthen LPCA or a broader benchmark/analysis paper. It is
not strong enough to justify a standalone AAAI method paper without C3.

## Forbidden continuation

- No GPU training under the MMLPO name.
- No implementation of CAPG with latent terminology.
- No claim that the released surrogate is invalid merely because it is not an
  exact PPO likelihood; empirical utility remains a separate question.
- No return to exact-density substitution without new evidence, because the
  frozen LPCA B2 gate already found worse local gain than the released
  surrogate.

## Portfolio action

Archive MMLPO as a negative, theory-backed audit. The next primary search should
avoid repairing the same likelihood contract and instead target a mechanism
that preserves every rollout marginal while improving group-level exploration
or credit assignment. Such a candidate must undergo a fresh collision audit
before any model training.
