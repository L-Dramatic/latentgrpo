# AAAI Direction Decision After OMPI and PCMC Gates

Decision date: 2026-07-18

## Verdict: HOLD, no training-ready method idea

The external report proposed a two-candidate sequence. Both candidates have now
reached their preregistered terminal conditions:

| Candidate | Terminal result | Resource decision |
|---|---|---|
| OMPI-R | KILL: exact identity with TPO on the same empirical marginal logits, plus an incomplete policy-gradient contract | No checkpoint inference or training |
| PCMC | KILL at checkpoint A0: Latent-GRPO passes, but source-faithful SofT-GRPO is almost one-hot and misses the absolute JS threshold by about five orders of magnitude | No A1, Gate B, or training |

This is not a failed implementation phase. It is the intended outcome of doing
method-identity and phenomenon-existence gates before expensive optimization.
No H20 rental or RL run is justified by the current portfolio.

## What is preserved

- All OMPI exact derivations, counterexamples, and negative decisions.
- The complete PCMC protocol, source adapters, official sampler replay, 1,000
  A0 records, positive Latent result, negative SofT result, and diagnostics.
- Managed execution and shutdown-safe infrastructure for future checkpoint
  gates.

The strong Latent non-closure phenomenon remains scientifically interesting,
but it is not promoted after observing the cross-checkpoint failure. A future
project may reuse it only under a new claim and new preregistration.

## Rejected reactions

- Lowering the PCMC 0.005 JS threshold after seeing SofT.
- Removing SofT and presenting a Latent-only universal claim.
- Running A1 only on Latent to rescue the original method narrative.
- Rebranding ordinary distribution distillation or top-1 hardening as PCMC.
- Starting LoRA/RL because implementation infrastructure already exists.

## Next research phase

Start a fresh collision-first idea search rather than extending either failed
candidate. A new main-track candidate must satisfy all of the following before
GPU training:

1. an independently stated contribution that is not a composition of a current
   optimizer and a standard latent-variable estimator;
2. direct-neighbor and strongest-composed-baseline checks;
3. an oracle ceiling showing at least a material, actionable gap;
4. a natural-checkpoint existence test across at least two policy families;
5. an explicit leakage/identifiability audit and immutable KILL thresholds;
6. a low-compute causal gate whose failure ends the direction.

The cross-checkpoint action-geometry contrast is a useful observation for idea
generation, not yet a candidate: Latent actions have median effective support
4.43 and strong non-closure, while SofT actions have median effective support
approximately 1 and close trivially. Any future claim built from this contrast
must first beat simple entropy/temperature/top-1 explanations and establish an
outcome-level need, not merely report a Jensen gap.
