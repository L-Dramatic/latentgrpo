# Gate -1 Decision

Date: 2026-07-18

Decision: `PASS_GATE_MINUS_ONE / NO GPU / NO SCIENTIFIC RESULT YET`

## What passed

- Four source-pinned methods in four mechanism families.
- Two deterministic methods and two stochastic latent-RL methods.
- Four immutable public checkpoint revisions; three are official author
  releases and Coconut is an explicitly labeled independent reproduction.
- All four checkpoints are downloaded locally and SHA-256 verified.
- Every required source file matches its pinned commit and file hash.
- The schema forces stochastic policy-measure auditing only where applicable.
- The collided broad shortcut/mechanism claim is machine-locked as excluded.

The machine report is
`artifacts/latent_reasoning_contract_benchmark/gate_minus_one_v1.json` with
SHA-256
`d25913124ff535f56509c0137366d6be2f3b16fae1ae43c880cd2fbc49724f5a`.
The source manifest SHA-256 is
`61e630ad769f7b8cfa548b418ed00e5ffec5d205e7420431ac3dc8a889eb7fe2`.

## Interpretation

This pass establishes feasibility and a nontrivial residual novelty boundary.
It does not establish that contract distinctions predict behavior, that any
release is defective, or that LRC-Bench is already a competitive AAAI paper.
The direct-neighbor collision risk remains high, especially against Four
Axioms and causal audits of Coconut/CODI.

## Authorized next work

Implement Gate 0 source-native fixtures for Coconut, CODI, Latent-GRPO, and
SofT-GRPO. Reuse existing checkpoint and sampler replays where they match the
new typed interface, but do not import their old thesis conclusions.

No checkpoint intervention GPU run is authorized until all four fixtures pass
source-equivalence, reconstruction, deterministic replay, and no-effect
controls. No training is authorized.
