# Deterministic Intervention Confirmation Decision

Decision: `PASS_CONFIRMATION`

## Result

The untouched 32-question confirmation split passed the unchanged A0 rule for
both Coconut and CODI. All 64 method-prompt records completed, every value was
finite, each equal-depth condition executed six latent positions, and native
replay again had exactly zero NLL and scored-logit error.

| Method | Repeat first | Reverse steps | Zero state | Norm-matched random |
|---|---:|---:|---:|---:|
| Coconut | 0.6529 | 0.2526 | 0.7303 | 0.6569 |
| CODI | 0.3390 | 0.0563 | 2.6791 | 2.1241 |

Values are within-method mean target-NLL changes from native execution. The
repeat-first 95% bootstrap intervals were `[0.1572, 1.1530]` for Coconut and
`[0.1307, 0.5653]` for CODI, entirely above zero.

## Decision Boundary

The deterministic intervention phenomenon is real and reproducible: both
released mechanisms depend on the content of their recurrent states at fixed
latent depth. This is necessary for LRC-Bench instrumentation, but not
sufficient for an AAAI contribution because generic causal intervention on
latent reasoning has close prior art. Reverse-step uncertainty also prevents a
strong ordering-specific claim.

The benchmark thesis remains conditional on the distinctive stochastic branch:
Latent-GRPO and SofT-GRPO must expose source-faithful execution/storage/scoring
distinctions with operational effects that simple entropy, support, depth, and
perturbation baselines do not explain.

No training or cloud GPU was used.

## Frozen Evidence

- Config SHA-256: `b34b613681bdd3629f6fcefdac53f874be508fd28fe227fbe442cb720520e3b4`.
- Confirmation wrapper SHA-256: `3452ba832490676253ee0f6bfe3860854b136393ab996ed92bee42723ac1a610`.
- Shared runner SHA-256: `f3ecb84a0d15c99ec67ae76c93b93f447217ac5e0f7428098eb8ebee8e6fa49c`.
- Records SHA-256: `1d23ce1717d1c5b39331f61ebfa7e3d87bd2da5613e5e2e1cc1680e14f18c6e5`.
- Summary SHA-256: `fce0e9b759b1b4918b4e7ae435d6c6c85e2d87b138ba60ca5d700008cfa9489d`.
- Integrity audit SHA-256: `3585a3524d32a3cbbd17b2d0532acdb1780a6179a62db0181a1fd9fb3e4045dc`.
