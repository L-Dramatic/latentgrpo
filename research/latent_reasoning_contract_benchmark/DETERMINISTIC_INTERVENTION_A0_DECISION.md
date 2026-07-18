# Deterministic Intervention A0 Decision

Decision: `PASS_A0_SIGNAL`

Integrity audit: `PASS_A0_AUDIT` with all seven controls passing.

## What Passed

All 64 method-prompt records completed on the 32 frozen calibration questions.
Both methods reconstructed their native recurrence with exactly zero NLL and
scored-logit error. Every output was finite and every matched condition
executed six latent positions.

Both methods cleared the frozen calibration signal rule under all four
equal-depth interventions. The strongest paired mean NLL increases were:

| Method | Repeat first | Reverse steps | Zero state | Norm-matched random |
|---|---:|---:|---:|---:|
| Coconut | 0.5921 | 0.2155 | 1.1026 | 0.4344 |
| CODI | 0.1924 | 0.0630 | 2.5834 | 1.9092 |

The repeat-first and zero-state mean effects had bootstrap intervals entirely
above zero in both methods. The reverse-step intervals crossed zero in both
methods, so A0 does not establish robust ordering sensitivity.

## Interpretation

This is a valid calibration result: the instrumentation can reconstruct native
execution and detect outcome sensitivity in two deterministic latent-reasoning
families. It is not a novel causal-use claim and does not establish the
LRC-Bench paper thesis. The effects may still be explained by simple state
content, depth, or perturbation magnitude.

The untouched 32-question confirmation set remains unread. The next protocol
must preserve the same questions, target formats, conditions, thresholds, and
aggregation, and the stochastic policy-contract branch must still contribute
non-redundant evidence before the benchmark thesis can pass.

No training or cloud GPU was used.

## Frozen Evidence

- Config SHA-256: `3ba1656e68ca18202bef015a36164c0c70018a8e51a5f8d504c73bf33793760b`.
- Runner SHA-256: `f3ecb84a0d15c99ec67ae76c93b93f447217ac5e0f7428098eb8ebee8e6fa49c`.
- Records SHA-256: `6fbf9c0fcf9ea6a29132137e414553de76a984d9a9ebd4d003dfb0930f03ee49`.
- Summary SHA-256: `4f7bafe38bd0a41fe701c80a8dbb2a86336fbc0b7fbdd3cc7335a987dbbc4858`.
- Integrity audit SHA-256: `6ccdf7fe41b6595671cfbf9a154fb4e6bedaf6693f25c93fd5cdad7a7230c205`.
