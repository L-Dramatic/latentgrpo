# P0 Claim Contract: When Is a Short Horizon Enough?

**Project identifier:** Horizon Sufficiency for Recursive Latent Updates  
**Status:** frozen P0 contract  
**Drafted:** 2026-07-16  
**Frozen:** 2026-07-16 after independent claim, mathematics, and collision red-teams  
**Current decision:** `HOLD`  
**Paper type if all gates pass:** method + mechanism diagnosis + supporting theory

This contract narrows the project after collision subtraction. It is not a new
name for the stopped coordinate-invariance headline. The prior BCG, CFG/SFTR,
FCTR, and SWITCH assets are inputs and controls; none is automatically evidence
for the revised claim.

## 1. Working title and one-sentence thesis

### Preferred working title

**When Is a Short Horizon Enough? Auditing Continuation Risk in Recursive
Latent Reasoning**

### Forbidden title family

Do not use “No Privileged Coordinates,” “Coordinate-Equivariant Trust Region,”
or “CETR” as the paper headline. Those phrases center claims already covered by
gauge-freedom, information-geometry, FishBack, and natural-gradient work.

### One-sentence thesis

> Natural latent-state updates in recursive reasoning can be mispriced as safe
> by cheap short-horizon behavioral metrics even when their longer observable
> continuations drift materially; a budget-adaptive estimator can certify when a
> short horizon is sufficient for local trajectory-Fisher risk, with controlled
> finite-sample error and a separately validated bridge to finite-step
> continuation KL, without first computing the complete long-horizon Fisher
> oracle.

This sentence is aspirational. At present, neither the natural-update phenomenon
nor the certificate has been established.

### Full paper claim that must be earned

> In a preregistered recursive latent-reasoning regime, some natural updates
> satisfy next-token or short-prefix behavioral budgets yet cause substantially
> larger delayed observable-continuation drift. We introduce a compute-adaptive
> horizon-sufficiency certificate that selects the shortest adequate horizon
> without first evaluating the full oracle and provides a finite-sample local
> trajectory-risk guarantee. A uniform remainder bound or an expected-KL upper
> confidence line search bridges the certificate to finite-step behavior. Used
> to allocate or constrain Successor-Fisher updates, the method controls delayed
> drift better than FishBack-style one-step
> geometry, short-prefix Fisher, fixed long-horizon Fisher, direct Monte Carlo
> continuation KL, and recurrent-stability baselines at matched compute and local
> objective gain, while preserving task utility; the result replicates in two
> mechanism-distinct recursive latent-reasoning architectures.

Every clause requires separate evidence. Partial success cannot borrow wording
from the unearned remainder.

The cost claim is deliberately narrow. The local certificate aims to avoid the
full long-horizon **matrix** oracle. If a finite-step bridge samples suffixes to
`H_cert`, every such call counts in matched compute. Only a structural/remainder
envelope can claim a finite-step guarantee without paying that suffix cost. If
the bridge routinely costs as much as direct long-horizon Monte Carlo, the
method has no compute claim.

## 2. Exact problem definition

At a replayable latent interface, an algorithm proposes an update `v` drawn from
a preregistered natural update distribution `Q(v | x)`. Keep three objects
separate:

- `q_h(v) = 0.5 v^T F_h v`, the local unregularized Fisher risk;
- `a_h(v) = 0.5 v^T (F_h + lambda M) v`, the regularized local update budget; and
- `d_K(v) = D_K(z, R_z(v))`, the finite-step reference-rollout expected
  chain-rule KL at evaluation horizon `K`.

The empirical problem is not whether `q_H >= q_h` in a quadratic form. That is a
standard consequence of path-Fisher decomposition. The empirical problem is:

> Does a deployable short-horizon decision rule make materially wrong safety or
> update-ranking decisions on natural updates, after calibration against all
> cheap baselines and after accounting for prompt and rollout uncertainty?

The method problem is:

> Can an anytime procedure inspect horizons and reference histories
> sequentially, stop before the full oracle cost when possible, and certify that
> the unobserved tail cannot change the operational decision beyond a declared
> tolerance?

The certificate controls `a_H` unless an explicit Taylor-remainder bound or
expected-KL upper-confidence line search connects it to `d_H`. The project is
specifically about observable behavioral consequences of
recursive or soft latent reasoning. It is not a generic representation metric,
generic activation-steering method, or generic recurrent-stability paper.

## 3. Primary claim ladder

Claims are cumulative. A higher claim is unavailable unless all lower claims
pass their frozen gates.

### C0 — measurement correctness

The implementation correctly distinguishes factual-history diagnostics,
single-reference-rollout chain-rule estimates, and the reference-rollout
expected chain-rule trajectory target. EOS, temperature, support, shared
histories, Jacobians,
precision, and compute accounting pass metamorphic and identity controls.

**Role in paper:** necessary correctness only; no novelty claim.

### C1 — natural Horizon-Gap phenomenon

On held-out prompts and natural updates, the best cheap short-horizon method
misranks or falsely accepts a practically meaningful subset of updates relative
to a stochastic reference-history expectation and seed-disjoint finite-step
behavioral labels. The gap survives `H2/H3/H8`, a
semantic prefix, FishBack, whitening, step-size/scalar-KL retuning, direct
matched-budget Monte Carlo, and recurrent-stability diagnostics.

**Role in paper:** mechanism diagnosis and motivation.

P1 may use a fixed long-horizon oracle to prove operational materiality. That is
not the adaptive method contribution. C3a is earned only when the adaptive
certificate recovers the seed-disjoint finite-step advantage under matched
compute without routinely paying the oracle cost.

### C2 — horizon-sufficiency certificate

An anytime, finite-sample procedure decides when the current horizon is
sufficient for local trajectory-Fisher risk in the preregistered update
subspace, with simultaneous coverage over both adaptive horizons and adaptive
rollout counts. It does not first compute the full oracle horizon and remains
valid under optional stopping and shared rollout histories. Behavioral-risk
wording additionally requires the finite-step bridge above.

**Role in paper:** primary method and theory contribution.

### C3a — activation-level operational consequence

Using the adaptive certificate to allocate measurement or constrain local
activation updates reduces seed-disjoint finite-step expected long-horizon KL or
changes a preregistered free-generation consequence at matched utility and
compute relative to the strongest simple method, without routinely paying the
fixed oracle cost. False-safe reduction is supporting evidence, not a sufficient
operational pass by itself.

**Role in paper:** local method value, not merely estimator fidelity.

### C3b — parameter-space bridge

For actual parameter deltas or replayed optimizer updates, the complete parameter
trajectory score is controlled and the activation-level certificate predicts
the seed-disjoint finite-step parameter-update consequence.

**Role in paper:** bridge evidence; insufficient for a training-method headline.

### C3c — actual multi-step training consequence

Certificate-guided multi-step training improves or preserves reward, utility,
and stability under matched tokens, calls, time, memory, and seeds relative to
all simple controls.

**Role in paper:** mandatory for the full strong-method headline.

### C4 — mechanism-distinct replication

C1–C3c replicate in a second latent-reasoning architecture with a materially
different recurrence or soft-thought mechanism, on more than one task/template
family and with a frozen distribution-shift test.

**Role in paper:** scope and robustness required for a strong main-track claim.

## 4. Evidence and wording ladder

| Level | Minimum evidence | Maximum allowed wording |
|---|---|---|
| L0 | Estimator, EOS, teacher-forcing, precision, and transport controls | Engineering contract only |
| L1 | Frozen fixed-history, fixed-horizon SWITCH screen | Candidate `factual_path_prefix_ggn` horizon effect |
| L2 | Natural updates, reference-history expectation, seed-disjoint finite-step consequence, and strict confound controls | One architecture exhibits a Horizon Gap |
| L3 | L2 replicates in a mechanism-distinct architecture | Cross-architecture recursive-latent mechanism claim |
| L4 | Finite-sample adaptive certificate beats direct estimation | New horizon-sufficiency estimator/certificate |
| L5 | Certificate-guided local updates beat all simple controls | SFTR may be used as a method name |
| L6 | Parameter-training benefit, task utility, and cross-architecture replication | Full strong-method paper claim |

No artifact or abstract may use wording from a higher level.

## 5. Headline claim authorization

The preferred title and thesis may be used as the paper headline only if all of
the following are true:

1. C0 passes on every architecture.
2. C1 passes on a disjoint held-out set with the effect sizes in
   `P0_KILL_CRITERIA.md`.
3. C1 is present for objective- or optimizer-derived updates, not only searched
   eigenvectors or artificial chart directions.
4. C2 provides a genuinely new compute-aware certificate rather than fixed
   long-horizon Fisher, low-rank natural gradient, or generic damping.
5. C2 beats direct continuation-KL Monte Carlo under matched wall time, rollout
   calls, and memory.
6. C3a demonstrates a local operational consequence, C3b validates the
   parameter-space bridge, and C3c demonstrates actual multi-step training
   benefit.
7. C4 passes without test-dependent threshold changes.

If only C0–C1 pass, the maximum defensible product is a diagnostic/benchmark
paper. If C0–C2 pass but C3a fails, the contribution must be re-evaluated as a
narrower estimation paper. If C3a passes but C3b fails, the maximum is a
local-update certificate paper. If C3b passes but C3c or C4 fails, the maximum is
a parameter-update certificate paper; it cannot retain the full training-method
headline by default.

## 6. Required contribution package

A competitive method paper must contain exactly one dominant story supported by
five connected contributions:

1. **Problem definition:** horizon sufficiency for natural recursive-latent
   updates, including an operational false-safe decision.
2. **Measurement contract:** unbiased or explicitly bounded reference-history
   estimation of observable continuation risk.
3. **Mechanism result:** a clean delayed-risk regime not explained by formatting,
   generic instability, or an adversarial coordinate chart.
4. **Adaptive method:** a sequential estimator/certificate with finite-sample
   uncertainty and a matched-compute stopping rule.
5. **Operational validation:** safer or more stable local and parameter updates
   at matched utility, replicated across mechanism-distinct architectures.

Coordinate equivariance, path-Fisher algebra, low-rank solvers, and exact
function-preserving charts may appear only as correctness controls or supporting
analysis.

## 7. Natural update population

The primary empirical distribution must be generated before seeing test risk and
must reflect real algorithmic actions. It must include at least:

- an objective- or optimizer-derived update family; and
- one additional family such as released exploration perturbations, latent-head
  or projection updates, or replayed adapter/LoRA steps.

Worst-case generalized eigenvectors may diagnose existence but cannot establish
prevalence or practical importance. Random directions are acceptable only when
they match an actual exploration mechanism or are clearly labeled stress tests.
The strong phenomenon claim must survive in at least two natural update families.

“Generated from a task gradient” is not by itself an in-support guarantee. Each
family must pass calibration-frozen support checks, including a representation
density or Mahalanobis diagnostic and a behavior-level indicator such as decoder
entropy. Checkpoint deltas, real optimizer steps, or parameter updates mapped
through the complete interface are preferred over arbitrary activation edits.

## 8. Relationship to existing project assets

### 8.1 BCG and CFG

BCG supplied the correct same-history finite-horizon KL estimator contract. CFG
in archive Section 16 names the expected path-Fisher object. These are conceptual
and engineering foundations, not standalone novelty.

### 8.2 SFTR

The old SFTR concept — optimize under a fixed finite-horizon Fisher — is not the
new primary method. A fixed `F_H^{-1}g` method is too close to FishBack, natural
gradients, TRPO, and path Fisher. It may be a baseline or downstream consumer of
a successful horizon-sufficiency certificate.

### 8.3 FCTR and coordinate invariance

The old coordinate-first FCTR headline is permanently stopped. Its exact
transport tests remain useful metamorphic checks. A coordinate consequence is
neither necessary nor sufficient for the revised Horizon-Gap claim.

### 8.4 Frozen SWITCH C2

SWITCH C2 remains immutable under its original preregistration. It computes
`V1/V3/V8/semantic/V32/V64` geometry along the unperturbed greedy visible
history, then measures separate consequences. Therefore it can screen whether a
longer path-conditioned prefix matters in a substantive hidden-recurrence model,
but it does not estimate the expected stochastic trajectory Fisher required by
this contract.

Its strict utility outcome is a factual/local utility diagnostic, not task
accuracy, reward improvement, or training stability. Its objective-gradient axis
may support a natural-update screen only after a representation-support check;
the three seeded random axes remain stress probes.

A C2 result must retain its original coordinate-conjunctive verdict. It cannot
be relabeled post hoc as a complete C1 result. Before any 96 GB checkpoint run,
an additive compatibility decision must state whether C2 is worth its cost as a
factual greedy-prefix screen and whether a separate stochastic P1 protocol is
still needed.

## 9. Evidence status at contract time

| Evidence | What it establishes | What it does not establish |
|---|---|---|
| BCG estimator and identity tests | Same-history chain-rule KL can be implemented with EOS and temperature controls | A natural Horizon Gap |
| GPT-2 Coconut v2 pilot | Initial H1 weakness on one checkpoint | Clean delayed risk; all continuations began with `###`, and H2/H3 explained most ranking variation |
| FCTR C0 toy gate | Tensor transport and solver controls | Natural phenomenon or novelty |
| Coconut C1c | Real-checkpoint differentiation and transport engineering | Substantive continuation geometry; output is delimiter plus short answer |
| SWITCH C2 release/tests | Source-pinned, reproducible factual greedy-prefix experiment is engineered | Any scientific checkpoint result; the checkpoint experiment remains `Not run` |

There is currently no positive held-out evidence for C1, no implemented C2
certificate in the sense of this document, and no training authorization.

## 10. Explicit non-claims

The paper must not claim:

- that hidden representations have no privileged coordinates as a new result;
- that Euclidean metrics are coordinate-dependent as a new result;
- the first behavior-induced activation geometry;
- the first pullback-Fisher activation steering method;
- the first natural-gradient or KL trust region for latent actions;
- the first path/trajectory Fisher for a stochastic dynamical system;
- the first observation of non-local latent reasoning or recurrent instability;
- the first certified predictability horizon, Lyapunov horizon, or generic
  finite-sample trajectory certificate;
- that a single greedy or teacher-forced prefix equals an expected continuation
  distribution;
- that longer horizons are universally better;
- that `H32` or `H64` is a universal ground truth;
- that artificial coordinate charts demonstrate real-world prevalence;
- that a task-gradient activation edit is automatically in-distribution or
  representative of an optimizer update;
- that activation-space evidence automatically transfers to parameter training;
- that a local quadratic Fisher budget automatically guarantees finite-step or
  global behavioral safety;
- that SWITCH's forced `</swi>` marker is a GRPO-sampled continuous action;
- that SWITCH C2 has already run or produced a scientific result; or
- that generic contraction algebra is the main theorem.

## 11. Required baselines

Every final empirical claim must compare against the strongest applicable member
of each family:

1. `H1/H2/H3/H8` and semantic-prefix conditional KL/Fisher;
2. FishBack-style one-output pullback Fisher;
3. Euclidean, cosine, calibration-only whitening, and matched-step baselines;
4. exact conditional-KL sums on a fixed path where applicable, plus a
   high-precision Monte Carlo/UCB expected-KL line search for the stochastic
   target;
5. direct reference-rollout Monte Carlo long-continuation KL under matched
   compute;
6. fixed longer-horizon Fisher as an oracle/upper-cost baseline;
7. recurrent-Jacobian spectral or STARS-style stability diagnostics;
8. a Certified-World-Model-style adapted-metric/predictability-horizon baseline;
9. representation-support/OOD diagnostics; and
10. the full certificate without each uncertainty, tail, and adaptive component.

A simple baseline within the frozen practical-equivalence margin wins.

## 12. AAAI positioning

If C0, C1, C2, C3a, C3b, C3c, and C4 pass, the paper is a strong method paper
with a mechanism-first opening and a supporting theoretical certificate. The
AAAI story is not “better latent geometry.” It is:

1. an overlooked reliability failure in recursive latent updates;
2. a falsifiable operational definition of short-horizon adequacy;
3. a new budget-aware method that knows when more future evidence is necessary;
4. rigorous uncertainty and stop conditions; and
5. cross-architecture consequences for safe, efficient optimization.

Without the certificate, the ceiling is a diagnosis paper. Without the natural
phenomenon, the entire direction is killed regardless of mathematical elegance.

## 13. Current authorization boundary

The project remains `HOLD`. The next permitted work is:

1. finish independent P0 red-team review;
2. resolve every mathematical and collision objection;
3. audit frozen SWITCH C2 against this contract;
4. decide whether C2 is worth running as a factual greedy-prefix screen; and
5. freeze a separate stochastic, natural-update P1 preregistration before any
   held-out result.

GPU calibration still requires a passed CPU/source feasibility audit and an
explicit `GO-P1` decision. A frozen overlay is necessary but not sufficient.

Not authorized yet:

- implementing SFTR or a horizon certificate;
- downloading or running the SWITCH checkpoint solely on the old rationale;
- using held-out prompts;
- GPU training; or
- drafting a positive paper abstract.

The permanent stop and downgrade rules are in `P0_KILL_CRITERIA.md`.
