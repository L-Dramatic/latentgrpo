# Four-Method Checkpoint Smoke Decision

Decision: `PASS_CHECKPOINT_SMOKE`

Date: 2026-07-18

## Result

All four methods have hash-pinned checkpoint evidence under their native
execution contracts. Coconut, Latent-GRPO, and SofT-GRPO reuse immutable
engineering artifacts. The official CODI checkpoint loaded with zero missing
and zero unexpected keys, executed exactly six latent steps twice, produced
finite and bitwise-repeatable outputs, and peaked at 331.37 MiB of allocated
CUDA memory on a local RTX 4060 Laptop GPU.

No training, answer decoding, task scoring, cloud GPU, or intervention-effect
inspection occurred. The artifact explicitly records `scientific_evidence` as
false.

## Frozen Evidence

- Config SHA-256: `1ba5be44bff4c929112edd88f59c4cab643b1a9db2a0a4ec306684ebebcb0977`.
- Runner SHA-256: `dd6dfe48cf7150f5ae51723631ad3ecb5dde3480b1c5c6c4b2c047f654a6aea2`.
- Result SHA-256: `e0c1a0baf20f7c91017f4b33d2b2e23680d489d3bd9431a8af15dc6ea4358198`.

## Invalidated Attempt

The first CODI invocation stopped before producing an artifact because the
runner moved the wrapper to CUDA but omitted the full-wrapper BF16 cast used by
the official `test.py` lines 99-100. This left the projection in FP32 while its
input was BF16. The attempt is invalid engineering evidence: it neither passed
nor failed a scientific gate.

The runner was corrected to reproduce the pinned official cast without
changing the frozen checkpoint, prompt, architecture, threshold, dependency,
or acceptance criteria. The successful rerun is the only checkpoint-smoke
result used downstream.

## Authorized Next Step

Freeze a tiny checkpoint-state intervention preflight before examining any
effect. It must specify the intervention, native tasks, strongest simple
baselines, outcome metrics, and GO/HOLD/KILL thresholds in advance. Full
training remains unauthorized.
