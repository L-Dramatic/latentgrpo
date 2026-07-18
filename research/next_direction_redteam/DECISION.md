# Next-Direction Red-Team Decision

Decision date: 2026-07-18

## Binding verdict

| Candidate or portfolio | Decision | Reason |
|---|---|---|
| 26-candidate portfolio | `NO-GO` | No candidate clears independent estimand, non-composition, identification, simple-baseline, and oracle gates. |
| PTPU | `KILL-COLLISION` | Its policy-indexed prefix utility is already explicit in PUM; target-policy transport is standard conditional OPE; no independent estimator is defined. |
| LRPE | `NO-GO` | Standard OPE/replay applied to test-time reasoning controllers; online rerun remains stronger and simpler. |
| APVC | `NO-GO` | Direct collision with setwise verification, adaptive calibration, reward-hacking control, and anytime certification. |

PTPU does not advance to the proposed synthetic CPU Gate 0. That gate can
validate standard estimators on a designed DAG but cannot reverse a method
identity failure.

## Resource authorization

- Do not implement or run `ptpu-cpu-identification-v1-20260718`.
- Do not run PTPU checkpoint inference or natural-policy Gate 1.
- Do not open or rent a GPU for any candidate in this report.
- Do not train a critic, verifier, LoRA adapter, RL policy, or 7B model.
- Preserve the current repository, reports, and negative artifacts unchanged.

## Research boundary for the next search

The next idea search must start from a fresh deployment failure or
identification problem, not from another recombination of the following closed
families:

1. latent/soft action likelihood repair;
2. generic prefix value plus policy conditioning;
3. standard OPE applied to reasoning logs or controllers;
4. verifier plus adaptive stopping/calibration;
5. retrospective rescue of OMPI, PCMC, LPCA, SGGE, or FCTR.

A new candidate is allowed into implementation only after all of these are
written and pass independently:

1. a one-sentence estimand unavailable in the nearest primary paper;
2. an exact subtraction against the strongest composed baseline;
3. a natural-data oracle ceiling before method training;
4. a leakage and identifiability contract;
5. a low-compute KILL gate whose positive result cannot be forced by synthetic
   construction.

Until such a candidate exists, the project status is `HOLD / NO TRAINING-READY
METHOD`, not an invitation to spend more compute.
