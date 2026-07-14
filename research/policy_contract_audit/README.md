# Latent Policy Contract Audit (LPCA)

LPCA is the active primary research direction. It treats latent-reasoning RL as
a policy-definition problem before treating it as an optimizer-design problem.

## Thesis

For a latent policy update to have a clear interpretation, the rollout sampler,
the object executed by the model, the stored action, and the likelihood ratio
used by optimization must form one coherent contract. Current methods make
different choices at each boundary. LPCA formalizes those choices, tests exact
probability identities where they apply, and measures whether violations alter
clipping, gradients, or downstream behavior at real checkpoints.

## Evidence order

1. Pin official papers and source commits.
2. Specify action, execution, support, likelihood, and aggregation contracts.
3. Validate the audit with exact Concrete and exact Top-K Concrete controls.
4. Run source-faithful synthetic replays without tuning thresholds post hoc.
5. Preregister and run public-checkpoint logit/rollout audits.
6. Start matched training only if operational effect-size gates pass.

## Current status

Public-checkpoint Stage A passed its frozen gate on all 500 MATH-500 prompts:
all five controls and all five effect tests passed. Stage B1's public-base
preflight exposed severe later-step singleton-support collapse and is retained
only as a stress test. The frozen 48-state Stage B2 audit found a real
exact-score/surrogate gradient mismatch, but the exact-density candidate had
lower held-out local utility at every temperature. Matched replacement
training is rejected and no cloud GPU is authorized for that branch.

Pinned SofT-GRPO 1.5B and Latent-GRPO 1B checkpoints now pass weight,
native-load, prompt-range, finite-logit, and repeatability controls on the local
RTX 4060 Laptop GPU. The next low-compute gate is source-sampler equivalence at
one deterministic checkpoint state; 7B replication remains deferred.

## Current files

- `SOURCE_MANIFEST.json`: immutable source locations and commit hashes.
- `METHOD_MATRIX.md`: current source-grounded contract classification.
- `DERIVATION.md`: exact and surrogate score derivations for the LEPO audit.
- `contracts.py`: generic normalization, score, ratio, and clipping checks.
- `lepo.py`: source-faithful replay of the released LEPO latent step.
- `EXPERIMENT_LOG.md`: append-only gate outcomes, including invalidations.
- `PUBLIC_CHECKPOINT_PREREGISTRATION.md`: frozen Stage-A protocol and outcome.
- `STAGE_B_PREREGISTRATION.md`: frozen sequential and reward-conditioned
  score-direction gates.
- `stage_b_robustness.py`: non-gating prompt-clustered bootstrap for completed
  Stage B2 records.
- `trained_checkpoint_smoke.py`: pinned native-checkpoint integrity and
  no-effect forward-pass controls.
- `LITERATURE_BOUNDARY.md`: primary-source collision and novelty boundaries.
- `TRAINED_CHECKPOINT_MATRIX.md`: immutable public-weight inventory, scope, and
  cross-method execution order.

The historical ideas and all stopped method gates remain archived in the root
`RESEARCH_IDEA_ARCHIVE.md`.
