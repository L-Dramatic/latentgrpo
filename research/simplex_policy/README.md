# Simplex-GRPO research track

This package tests whether policy optimization should operate on the executed
soft-thought mixture rather than on its redundant auxiliary Gumbel scores.

The first gate is deliberately synthetic. It verifies the exact Concrete
density, compares score-function variance and importance-ratio quality, records
PPO clipping disagreement, and reproduces the inspected Latent-GRPO surrogate
gradient. A pass permits a released-checkpoint ratio audit; it is not evidence
of a real-model training improvement.

Run the frozen gate with the isolated research environment:

```powershell
& 'E:\LantentGRPO\_research_env\Scripts\python.exe' `
  -m research.simplex_policy.toy_gate `
  --output artifacts\simplex_policy\simplex_policy_toy_v1.json
```
