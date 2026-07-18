# LRC-Bench Current Thesis Decision

Decision date: 2026-07-18

Decision: `KILL_LRC_CURRENT_THESIS`

## Why the decision is final

The frozen claim contract permanently kills the candidate if policy-contract
findings occur only in Latent-GRPO and disappear in SofT-GRPO. That condition
is now met under source-native execution and an independently recomputed
1,000-record experiment.

| Evidence branch | Result | Interpretation |
|---|---|---|
| Coconut and CODI | Untouched confirmation passes | The latent states are causally load-bearing, but this is not a novel paper thesis. |
| Latent-GRPO | A0 passes strongly | Its ten-token actions are genuinely distributed and arithmetic execution differs from the hard-branch teacher. |
| SofT-GRPO | A0 fails by about five orders of magnitude | Its source-native `tau=0.1` action is effectively a sampled hard token. |

SofT-GRPO's calibration Q75 JS is `5.37e-8` against the frozen `0.005`
minimum. Across all 500 prompts, median maximum weight is `0.9999989`, median
effective support is `1.000002`, and arithmetic/teacher top-token disagreement
is zero. Latent-GRPO has median effective support `4.4262` and 42.4% top-token
disagreement. Raw accuracy across the checkpoints is not used.

## Integrity

`cross_family_thesis_audit.py` independently:

1. verifies every frozen evidence hash;
2. reruns the deterministic confirmation integrity analysis;
3. parses and validates all 1,000 stochastic records;
4. recomputes the frozen A0 decisions and all descriptive distributions;
5. checks both source-native sampler preflights; and
6. checks that the permanent KILL clause existed before the result.

Every control passes. The machine-readable result is
`artifacts/latent_reasoning_contract_benchmark/cross_family_thesis_audit_v1.json`.

## What is preserved

The instrumentation, four source-pinned adapters, deterministic intervention
records, stochastic source-equivalence tests, and positive/negative findings
remain valid research assets. They must not be presented as a surviving
LRC-Bench AAAI thesis.

No additional training, threshold change, prompt change, or selective method
removal is authorized for this thesis. Any successor must have a new claim,
new collision audit, and new frozen gate.
