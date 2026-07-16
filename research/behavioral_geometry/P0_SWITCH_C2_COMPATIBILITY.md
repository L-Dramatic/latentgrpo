# P0 Compatibility Audit: Frozen SWITCH C2 vs Revised Horizon Sufficiency

**Audit date:** 2026-07-16  
**Status:** frozen P0 compatibility decision  
**Frozen C2 scientific state:** `Not run`  
**Revised project state:** `HOLD`  
**GPU decision:** `NO-GO`; an additive overlay is necessary but not sufficient

This audit does not modify `../coordinate_invariance/SWITCH_C2_PREREGISTRATION.md`.
It determines what that experiment could and could not establish for the revised
Horizon-Sufficiency project.

## 1. Binding conclusion

Frozen SWITCH C2 is a valid, source-pinned experiment for its original FCTR
question. For the revised project, it is at most a **path-conditioned preliminary
screen**. It cannot establish the expected stochastic continuation Fisher or a
natural-update Horizon Gap, and it contains no adaptive horizon certificate.

Therefore the existing rationale is not sufficient to rent a 96 GB GPU. The
checkpoint may be run only after an additive overlay states exactly which new
claim the expensive measurement can decide without changing the old C2 verdict.
Even then, GPU calibration additionally requires frozen P0 documents, a passed
CPU/source feasibility audit, a frozen overlay or stochastic P1 preregistration,
and an explicit `GO-P1` decision.

## 2. Object-level mismatch

| Component | Frozen SWITCH C2 | Revised Horizon-Sufficiency target | Compatibility |
|---|---|---|---|
| Future history | Unperturbed greedy visible history | Histories sampled from the stochastic reference continuation law | **Mismatch** |
| Geometry | Cumulative pullback matrices along one factual path | Expected trajectory Fisher over reference histories | **C2 is a path-conditioned empirical GGN only** |
| Update subspace | Objective gradient plus three seeded Euclidean random axes | Preregistered, in-support natural update distribution, including an objective/optimizer family | **Partial**; the gradient requires a support check, and random axes are stress probes |
| Short baselines | L1/L3/L4, V1/V3/V8, semantic prefix, whitening, scalar V8 retuning | H1/H2/H3/H8, semantic prefix, FishBack, whitening, scalar retuning, direct MC, stability diagnostic | **Strong but incomplete** |
| Long candidate | Fixed V32; V64 oracle | Adaptive selection of the shortest sufficient horizon | **Mismatch** |
| Consequence | Same-history strict-holdout KL, local factual utility, and separate free-generation changes | Population false-safe decisions, expected continuation risk, utility/reward, and uncertainty over prompts and histories | **Partial**; C2 utility is not task accuracy or training reward |
| Coordinate claim | Required conjunct of original C2 | Metamorphic correctness control only | **Must be reported separately** |
| Training | None | Needed only at later strong-method stage | C2 cannot support a training claim |

## 3. Why the two Fisher objects cannot share one label

Frozen C2 explicitly states that all KL quantities use the unperturbed greedy
history for teacher forcing. Its `prefix_geometry` implementation accumulates
per-token pullback categorical Fisher matrices along that one visible prefix.

Archive Section 16 defines CFG as

\[
F_H(z)=
\sum_{t=1}^{H}
\mathbb E_{Y_{<t}\sim P_z}
\left[J_t(Y_{<t})^\top C_t(Y_{<t})J_t(Y_{<t})\right].
\]

The C2 matrix is one path-conditioned sample at a greedy, probability-dependent
history; it is not the reference-history expectation above. A deterministic
greedy path is not an unbiased Monte Carlo sample from the stochastic policy
unless the policy itself is declared deterministic, in which case the required
softmax-distribution KL is no longer the same target.

All future code, artifacts, and tables must therefore use distinct names:

- `factual_path_prefix_ggn` for the C2-type object;
- `single_reference_rollout_fisher_sample` for a stochastic path sample; and
- `expected_trajectory_fisher` for the reference-history expectation.

## 4. Evidence ladder for C2

| Outcome | Maximum revised-project interpretation |
|---|---|
| C2 controls fail | Engineering `FAIL`; do not interpret geometry |
| Coordinate effect fails, V32 sub-comparison fails | `FCTR: FAIL`; Horizon screen `FAIL` |
| Coordinate effect passes, V32 sub-comparison fails | `FCTR: FAIL`; Horizon screen `FAIL` |
| Coordinate effect fails, predeclared V32 Horizon overlay passes | `FCTR: FAIL`; factual greedy-prefix Horizon screen `PASS` |
| Both old C2 conjuncts pass | `FCTR C2: PASS`; factual greedy-prefix Horizon screen may pass if separately frozen |

Even the last row reaches only evidence level L1 in `P0_CLAIM_CONTRACT.md`. It
can authorize a stochastic natural-update P1 experiment, not SFTR implementation
or training.

## 5. Archive status conflict and resolution

Archive Section 16.6 says a disjoint BCG evaluation set must not be touched until
a substantive-continuation architecture and a second hidden-feedback
architecture are screened. Section 21 later authorizes SWITCH C2 operationally
for the revived FCTR line, but its supersession sentence names Sections 14 and
20.10, not Section 16.6.

For the revised Horizon-Sufficiency project, the conservative binding
interpretation is:

1. Section 21 authorizes C2 only under its original FCTR purpose.
2. It does not retroactively turn a greedy single-history C2 result into the BCG
   population gate required by Section 16.
3. SWITCH is a promising substantive-continuation architecture candidate, but
   the revised stochastic P1 population must be frozen before its held-out data
   are consumed for Horizon claims.
4. A second mechanism-distinct architecture remains mandatory for the full
   strong-method claim.

This interpretation preserves every earlier negative result and avoids using a
later operational note to weaken a scientific gate.

## 6. Required additive overlay before any checkpoint output

The overlay must be a new file. It must not edit the frozen C2 protocol. Before
checkpoint identity or eligibility output is inspected, it must freeze:

1. **Two separate verdicts:** the original FCTR C2 decision and a revised
   factual greedy-prefix Horizon-screen decision.
2. **Natural versus stress updates:** which gradient/algorithmic updates define
   the population and which seeded axes are only diagnostic.
3. **Object labels:** factual path GGN, stochastic single-rollout sample, and
   expected trajectory Fisher cannot be pooled.
4. **Primary delayed-risk target:** strict token interval,
   reference-sampled free-generation endpoint, termination, and local utility,
   with no test-time choice.
5. **Short baselines:** H1/H2/H3/H8, semantic boundary, FishBack, whitening,
   exact scalar retuning, reduced step, direct MC, spectral stability, and a
   Certified-World-Model-style predictability-horizon diagnostic.
6. **Horizon roles:** cheap `h`, finite oracle `H_cert`, and seed-disjoint
   behavioral endpoint `H_eval`, including an oracle-adequacy check.
7. **Calibration and test split:** prompts, construction/evaluation rollout
   seeds, eligibility, update families, thresholds, and all exclusions.
8. **Dependence and uncertainty:** hierarchical inference over prompts,
   templates, checkpoints, optimizer trajectories, and shared histories.
9. **Support validity:** representation and behavior-level OOD checks for every
   update family labeled natural.
10. **Interpretation ceiling:** a C2 overlay pass is L1 preliminary evidence only.
11. **No rescue rule:** if the original coordinate conjunct fails, report FCTR as
   failed even if the Horizon sub-analysis is positive.
12. **Compute accounting:** checkpoint calls, backward passes, rollout count,
    wall time, peak memory, and a stop rule after calibration.

## 7. What can be reused safely

Because no checkpoint result has been produced, the following assets can be
reused after the overlay is frozen:

- pinned source, checkpoint, adapter, tokenizer, and MATH-500 order;
- identity, eligibility, cache, precision, and source-equivalence controls;
- the first 16 eligible / next 32 eligible calibration-test partition;
- V1/V3/V8/semantic/V32/V64 factual-prefix code under corrected labels;
- consequence and utility measurement code;
- resumable journals, release bundle, and compute accounting; and
- all original C2 outputs for their original verdict.

The following cannot be inferred or reused without new measurement:

- expectation over stochastic reference histories;
- prevalence over a natural optimizer-update population;
- optional-stopping-valid horizon sufficiency;
- matched-compute comparison with direct Monte Carlo continuation KL;
- parameter-space trust-region behavior; or
- a second-architecture replication.

## 8. Next permitted work

The next step is a **CPU/source-only overlay feasibility audit**, not the GPU run:

1. determine whether SWITCH can replay multiple stochastic visible histories
   while differentiating candidate conditionals on exactly those histories;
2. identify one genuine objective/algorithmic update family beyond random axes;
3. estimate the number of forward/backward calls for multiple reference
   histories and each required baseline;
4. determine whether 16 calibration and 32 test prompts can support the desired
   prompt-and-rollout uncertainty, or whether a new frozen sample is required;
5. design an early calibration stop so a negative signal does not consume the
   full 96 GB budget; and
6. freeze the overlay only if its result can distinguish stopping the direction
   from authorizing a separate stochastic natural-update P1 gate.

Until that audit passes, do not download the checkpoint, rent a GPU, inspect
held-out outputs, implement the certificate, or train.
