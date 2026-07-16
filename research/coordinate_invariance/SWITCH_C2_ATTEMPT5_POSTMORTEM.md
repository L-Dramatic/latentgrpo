# SWITCH C2 Attempt 5 Postmortem

## Decision

**No-go for the frozen SWITCH C2 v1 operationalization.**

The run produced valid checkpoint-identity and eligibility evidence and a
complete 16-prompt calibration journal. It failed the frozen calibration
selection, so no calibration artifact exists and the held-out 32-prompt test
was correctly not run. This result does not logically disprove every possible
coordinate-invariant latent-geometry method, but it rejects the current V32
metric, gain rule, and numerical contract as a package.

## Stage outcomes

| Stage | Outcome | Evidence |
| --- | --- | --- |
| Source and release identity | Pass | Frozen v5 commit and source pins matched |
| Checkpoint execution identity | Pass | 8 prompts, zero mismatches in all four identity categories |
| Frozen natural-block eligibility | Pass | 500 rows scanned, 76 eligible, 16 calibration and 32 test prompts fixed |
| Calibration measurement | Complete | 16/16 append-only prompt records |
| Calibration selection | Fail | No global gain satisfied all-prompt radius and objective controls |
| Held-out C2 test | Not authorized | No calibration artifact was created |

The run used the pinned SWITCH Qwen3-8B adapter on one NVIDIA H20. Identity,
eligibility, and calibration consumed approximately 5,161, 25,586, and 11,979
seconds respectively. Peak allocated GPU memory reported by the checkpoint
stages was about 25,764 MiB.

## Formal blocker

The calibration selector requires both V32 and the calibration-selected simple
baseline to remain within relative hidden L2 0.05 on every calibration prompt.
At the smallest predicted objective gain (0.0001), prompt 8
(`test/intermediate_algebra/2046.json`) required:

- V32 relative hidden L2: 0.060155
- selected V3 baseline relative hidden L2: 0.060163

The prompt's full gradient norm was only 2.30e-5, roughly two orders of
magnitude below the next-smallest calibration gradients. Larger gains therefore
could not repair the violation: one prompt was invalid for the first three gain
levels, two for gain 0.003, and three for gain 0.01. Under the frozen all-prompt
rule, the valid gain count is zero.

## Independent negative signals

The formal radius failure is not the only reason to stop.

### Finite-difference control

The selected central-difference step was relative hidden L2 0.01. Its median
relative error was 0.261846 and p90 error was 0.574329, versus allowed maxima
0.10 and 0.20. The local derivative measurement therefore did not meet the
frozen numerical reliability contract.

### Candidate-versus-baseline calibration signal

V32's mean within-prompt Spearman margin over the strongest selectable simple
baseline was negative at every probe scale:

| Relative hidden L2 | V32 margin |
| ---: | ---: |
| 0.0025 | -0.046862 |
| 0.0050 | -0.067459 |
| 0.0100 | -0.020706 |
| 0.0200 | -0.007302 |

Even if the near-zero-gradient prompt were handled differently in a future
protocol, these calibration observations give no positive evidence that V32 is
better than a simple prefix baseline on this checkpoint.

## Controls that passed

The following controls argue against attributing the outcome to a basic chart
or transport implementation error:

| Control | Observed | Limit | Outcome |
| --- | ---: | ---: | --- |
| Zero-point logit absolute error | 0 | 0 | Pass |
| Basis orthogonality absolute error | 4.22e-15 | 1e-5 | Pass |
| Projected gradient relative error | 0.007897 | 0.01 | Pass |
| Metric update transport relative error | 5.80e-13 | 0.005 | Pass |
| Orthogonal Euclidean discrepancy | 2.19e-16 | 0.005 | Pass |
| Median condition-12 Euclidean discrepancy | 1.561674 | minimum 0.25 | Pass |

## Interpretation boundary

The defensible claim is narrow but decisive: the frozen SWITCH C2 v1 method is
not viable and should not advance to held-out testing or training. It would be
invalid to relax the radius, drop prompt 8, enlarge the finite-difference
tolerances, or choose another scale after seeing this result and then present
the same test split as confirmatory evidence.

Any continuation must be labeled a new protocol. It should first solve
derivative conditioning and near-zero-gradient eligibility on development data,
then demonstrate a positive candidate-versus-simple calibration margin before
renting another GPU for a held-out test. Given that all four observed V32
margins are negative, the recommended choice is to pivot away from this exact
V32/FCTR estimator rather than tune it further.

## Evidence map

- Machine-readable replay:
  `artifacts/coordinate_invariance/switch_c2_calibration_postmortem_v1.json`
- Complete calibration journal:
  `artifacts/coordinate_invariance/journals/switch_c2_calibration_4b0c45b26fbd8138.jsonl`
- Passing checkpoint identity:
  `artifacts/coordinate_invariance/switch_checkpoint_identity_smoke_v1.json`
- Passing eligibility scan:
  `artifacts/coordinate_invariance/switch_c2_eligibility_v1.json`
- Full local return folder (ignored by Git):
  `artifacts/coordinate_invariance/autodl_return_2026-07-16_attempt5_final/`

The final return bundle SHA-256 is
`da0cc1c3090ea27c63b6c086d7903229cb65ef052a2149f30b3d05143e441dae`.
Its tar stream contains 37 members, and all 36 files listed by the internal
manifest independently matched their recorded hashes.
