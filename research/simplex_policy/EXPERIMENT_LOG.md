# Simplex-GRPO Experiment Log

This file is append-oriented. Thresholds and failed gates are retained.

## 2026-07-14: Frozen CPU gate v1

- Config: `research/simplex_policy/configs/simplex_policy_toy_v1.json`
- Artifact: `artifacts/simplex_policy/simplex_policy_toy_v1.json`
- Status: **Fail under the frozen gate**
- Interpretation boundary: estimator-level evidence only; no real-model claim

### Passed checks

- The Concrete log density matched PyTorch's independent implementation with maximum absolute error `2.84e-14`.
- The simplex score matched the pathwise-gradient reference; the worst relative error across five scenarios was `0.0328`.
- The auxiliary-to-simplex likelihood-ratio variance reduction had geometric mean `1.264`.
- The simplex effective-sample-size ratio had geometric mean `1.415` and minimum `1.136`.
- Auxiliary log ratios were strongly sensitive to behaviorally null common score offsets.
- The inspected Latent-GRPO surrogate diagnostic reproduced misaligned objective gradients after a crossed margin for both positive and negative advantages.

### Failed frozen checks

- Gradient trace-variance reduction had geometric mean `1.2196`, below the frozen minimum `1.25`.
- PPO clipping-decision disagreement had mean `0.0210`, below the frozen minimum `0.05`.
- Thresholds were not relaxed after observing the result.

### Decision

Simplex marginalization is a valid and measurable variance reduction, but the operational difference was too small under the preregistered moderate-drift scenarios to justify a standalone headline method. Stop the Simplex-GRPO main line. Preserve the exact density implementation and surrogate diagnostic as reusable baselines or a possible implementation note.
