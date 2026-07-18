# Deterministic Intervention Confirmation Preregistration

Frozen: 2026-07-18, after A0 calibration passed and before any confirmation
checkpoint intervention was run.

## Locked Reuse

This protocol uses the 32 rows already marked `confirmation` in the original
label-blind GSM8K prompt manifest. Those rows were frozen before A0 and no
confirmation output has been inspected.

Relative to A0, the following are unchanged:

- Coconut and CODI checkpoints, source revisions, six-step recurrence, and
  native target formats;
- native replay, repeat-first, reverse-step, zero-state, norm-matched-random,
  and no-latent conditions;
- random and bootstrap seeds;
- target NLL, teacher-forced KL, source controls, memory ceiling, effect
  thresholds, and aggregation;
- the requirement that both methods independently show a qualifying
  equal-depth effect.

The only experiment change is `split: confirmation`. Calibration summary and
integrity-audit hashes are pinned as prerequisites.

## Decision

Any source, reconstruction, finite-value, depth, target-identity, or memory
failure is `HOLD_CONFIRMATION`. With controls passing, the unchanged A0 signal
rule passing in both methods is `PASS_CONFIRMATION`; otherwise the
deterministic effect branch is killed.

A pass confirms reproducible within-method checkpoint sensitivity. It still
does not establish novelty over existing latent-state causal audits. The paper
thesis additionally requires non-redundant stochastic policy-contract evidence
from Latent-GRPO and SofT-GRPO and held-out value beyond simple baselines.

No training or cloud GPU is authorized.
