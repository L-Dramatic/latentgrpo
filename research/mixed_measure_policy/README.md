# Mixed-Measure Latent Policy Audit

This directory records the zero-GPU decision gate for the provisional
**Mixed-Measure Latent Policy Optimization (MMLPO)** idea.

The audit separates three statements that must not be conflated:

1. the official Latent-GRPO sampler induces a mixed action law;
2. moving clipping atoms can invalidate a finite-step PPO likelihood ratio;
3. a practical sampler-preserving optimizer follows from those facts.

The first two statements survive. The third does not. MMLPO is therefore
stopped as a method-paper direction. The theorem and source audit remain useful
as a policy-contract diagnostic and as a narrow extension to LPCA.

## Contents

- `M0_CLAIM_CONTRACT.md`: frozen claims and kill conditions;
- `SOURCE_CONTRACT.md`: source-pinned sampler and objective semantics;
- `LITERATURE_COLLISION_AUDIT.md`: closest-work boundary;
- `MATHEMATICAL_RED_TEAM.md`: exact counterexample and overclaim checks;
- `DECISION.md`: final gate decision;
- `counterexample.py`: dependency-free exact calculations used by tests;
- `AUDIT_RESULTS.json`: machine-readable values for the frozen example;
- `SOURCE_MANIFEST.json`: source and literature identities.

No GPU experiment or model training is authorized by this audit.
