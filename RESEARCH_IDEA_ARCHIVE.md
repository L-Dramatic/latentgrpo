# Latent Reasoning Research Idea Archive

> Version: 2026-07-15  
> Scope: BCG-PO, SVCCO, CSTR-PO, FCTR, Simplex-GRPO, Top-K Concrete, FTK-PO, score-squashed Gumbel policy optimization, and LPCA  
> Purpose: preserve each direction as an independently recoverable research program while recording the current quality-first decision.

## 1. Shared problem setting

The project studies models that reason through continuous hidden vectors rather than fully verbalized chain-of-thought tokens.

- `h_t`: the complete prefix state at step `t`, including the prompt, visible tokens, latent history, and any cache needed to continue computation.
- `z_t`: a continuous latent action or latent thought produced at step `t`.
- `O_{t:H}`: an observable future outcome over horizon `H`, such as future tokens, answer distribution, verifier result, reward, tool action, or task success.
- `P_theta(O_{t:H} | h_t, z_t)`: the continuation distribution induced by a model after taking latent action `z_t` in prefix state `h_t`.

The central scientific difficulty is that a hidden vector has no intrinsic human-readable semantics. Euclidean distance, cosine similarity, isotropic noise, interpolation, density, and local principal directions are properties of a chosen coordinate representation. They are not automatically properties of the computation performed by the model.

All four ideas below rely on a common experimental substrate:

1. capture a complete prefix state and latent action;
2. replay from an arbitrary latent step;
3. compare continuations under matched random numbers;
4. distinguish raw-coordinate changes from observable functional changes;
5. account for every additional rollout, probe, and backward pass;
6. preregister success and stop conditions before large training runs.

## 2. Idea A: BCG-PO

### 2.1 Name

**Behavioral Continuation Geometry for Policy Optimization (BCG-PO)**

### 2.2 One-sentence thesis

Two latent thoughts should be considered close when they induce similar future behavior, not merely when their hidden vectors are close in the current coordinate system.

### 2.3 Intuition

Suppose two hidden vectors look very different numerically, but both lead the model to perform the same remaining reasoning and produce the same answer distribution. BCG treats them as behaviorally close. Conversely, two vectors can have high cosine similarity while causing different later reasoning, so BCG treats them as behaviorally distant.

BCG is therefore a map of hidden reasoning states organized by what they cause the model to do next.

### 2.4 Formal object

For a continuation policy `pi` and horizon `H`, define a continuation signature

```text
S_pi,H(h, z) = P_pi(O_{t:H} | h, z).
```

A functional distance can then be defined as

```text
d_BCG((h, z), (h, z')) = D(S_pi,H(h, z), S_pi,H(h, z')),
```

where `D` may be KL divergence, Jensen-Shannon divergence, Wasserstein distance, a reward-aware discrepancy, or a calibrated sketch of the full continuation distribution.

A stronger version should integrate over a family of continuation policies instead of using one frozen policy:

```text
d_family(z, z' | h) = E_{pi ~ Q}[D(S_pi,H(h, z), S_pi,H(h, z'))].
```

This reduces false equivalence caused by a weak or collapsed reference policy.

### 2.5 Candidate method pipeline

1. Collect latent actions together with replayable prefix states.
2. Produce matched-seed suffix rollouts from selected latent pairs.
3. Construct multi-horizon continuation signatures.
4. Train or calibrate an amortized estimator of behavioral distance.
5. Use the distance to regularize policy updates, identify aliases, or merge redundant branches.
6. Re-estimate or stress-test the geometry after policy updates to detect metric drift.

### 2.6 Potential contributions

BCG is publishable as a main contribution only if it establishes a latent-reasoning-specific result beyond classical behavioral equivalence:

- a reproducible failure law showing that coordinate proximity systematically mispredicts continuation behavior;
- a continuation estimator that is materially better or cheaper than simple next-token KL, Fisher distance, and verifier scores;
- an optimization or branch-management algorithm that improves real training under matched compute;
- a finite-horizon value-loss bound or a rigorous paired empirical bound;
- evidence across at least two distinct latent reasoning representations.

### 2.7 Experiments required

- **Architectures:** at least two latent reasoning mechanisms, not merely two model sizes of the same mechanism.
- **Tasks:** arithmetic reasoning, symbolic or logical reasoning, and at least one distribution-shift or long-horizon setting.
- **Coordinate baselines:** Euclidean, cosine, whitened Mahalanobis, local PCA, next-token KL, and pullback-Fisher approximations.
- **Continuation horizons:** next-token, short suffix, medium suffix, and answer-level outcome.
- **Primary tests:** pair-ranking agreement, alias detection, metric drift after training, reward prediction, merge safety, and matched-compute task performance.
- **Statistics:** paired effects, multiple seeds, bootstrap confidence intervals, sensitivity to horizon and rollout count.

### 2.8 Main literature collisions

- Causal states and bisimulation already organize states by future behavior.
- DeepMDP and deep bisimulation already connect representation distance to behavioral distance.
- BiPACE applies bisimulation-guided grouping and counterfactual estimation to LLM policy optimization.
- FishBack derives output-sensitive pullback geometry for transformer activations.
- Policy-dependent bisimulation work already studies geometry drift after policy changes.

Therefore, the claim must not be "the first behavioral geometry for hidden states." The defensible claim must concern a new latent-reasoning failure, estimator, or training consequence.

### 2.9 Compute profile

BCG requires repeated suffix rollouts for selected latent pairs. A naive all-pairs implementation is infeasible. Practical versions need active pair selection, shared continuations, cached prefix states, amortized signatures, and strict rollout accounting. Expected overhead is medium to high.

### 2.10 Decision and recovery condition

**Current status: Conditional Go as a foundation, not the headline direction.**

Revisit BCG as an independent paper only if experiments reveal continuation aliasing or policy-induced metric drift that is not explained by existing Fisher, KL, or bisimulation baselines and the resulting metric directly improves training or safe branch merging.

## 3. Idea B: SVCCO

### 3.1 Name

**Support-Valid Counterfactual Credit Optimization (SVCCO)**

### 3.2 One-sentence thesis

Credit for a final answer should be assigned to individual latent reasoning steps by comparing realistic counterfactual continuations, while excluding interventions that move hidden states outside the model's valid computational support.

### 3.3 Intuition

Outcome-only reinforcement learning gives every hidden step credit when the answer is correct and blame when it is wrong. This is too coarse. A correct trajectory may contain useless or harmful hidden steps, while an incorrect trajectory may contain a valuable prefix.

SVCCO asks what would happen if one latent action were replaced by another plausible action while the earlier context remained fixed. The replacement must be something the model could realistically produce, not an arbitrary zero vector or random perturbation.

### 3.4 Formal object

Let `I(z' | h_t, z_t)` be a valid intervention distribution over replacement latent actions. A step-level counterfactual effect can be written as

```text
A_t = E[R | h_t, do(z_t = z_t)]
      - E_{z' ~ I(. | h_t, z_t)}[R | h_t, do(z_t = z')].
```

The scientific burden lies in defining `I` so that it is:

- inside a defensible conditional support;
- semantically localized rather than changing many hidden factors at once;
- invariant or equivariant under an invertible reparameterization of latent coordinates;
- sufficiently overlapping with observed trajectories for stable estimation;
- compatible with the changed downstream cache and recurrent latent history.

### 3.5 Candidate method pipeline

1. Capture complete replayable states at each latent step.
2. Generate candidate replacement actions from conditional peers, a learned conditional generator, or behaviorally matched neighborhoods.
3. Reject candidates that fail functional support or continuation-validity checks.
4. Run matched-seed suffix rollouts for factual and counterfactual actions.
5. Estimate a step-level advantage, with variance reduction and selection-bias correction where justified.
6. Train the latent policy using step-specific credit rather than broadcasting terminal reward uniformly.

### 3.6 Potential contributions

The strongest possible contribution would be the first useful intervention semantics specifically for continuous latent actions: one that is support-valid, coordinate-independent, computationally tractable, and demonstrably improves step-level learning.

Secondary contributions could include:

- a positivity or overlap diagnostic for latent interventions;
- uncertainty intervals for counterfactual credit;
- an active intervention policy that spends rollouts only on ambiguous steps;
- a theorem bounding bias from imperfect support validity.

### 3.7 Experiments required

- Synthetic systems with known causal latent factors to test effect recovery.
- Real latent reasoning tasks with controlled harmful or redundant steps.
- Intervention baselines: zeroing, Gaussian noise, linear interpolation, nearest neighbors, conditional generators, peer trajectories, and functional neighborhoods.
- Credit baselines: terminal reward broadcast, leave-one-step-out, value functions, process reward models, counterfactual path methods, and action-conditioned peer baselines.
- Metrics: effect sign accuracy on controlled data, bias and variance, support violations, downstream task gain, rollout cost, and robustness under coordinate changes.

### 3.8 Main literature collisions

- IBPO already constructs implicit process advantages from multiple counterfactual reasoning paths.
- CVT-RL already combines controlled interventions, validity gates, frozen continuation policies, and adjusted estimators.
- BiPACE already uses action-conditioned peer baselines.
- Latent chain-of-thought studies already perform step-wise hidden-state interventions.
- Causal representation studies show that common interventions can be off-manifold and activate unnatural pathways.

Consequently, "counterfactual credit for reasoning" is not sufficient novelty. The remaining research space is the semantics and estimation of continuous latent interventions.

### 3.9 Compute profile

If `T` latent steps each require `M` counterfactual suffixes of horizon `H`, naive cost is proportional to `T * M * H`. This is the most expensive direction among the four. Active step selection and amortized outcome models are mandatory for scaling.

### 3.10 Decision and recovery condition

**Current status: No-Go as the first standalone direction; preserve as a high-upside extension.**

Revisit after the reparameterization work yields a coordinate-independent intervention family or proves that existing counterfactual credit changes arbitrarily under equivalent charts. Without that result, the method is too close to existing counterfactual process-credit work.

## 4. Idea C: CSTR-PO

### 4.1 Name

**Conformal Support-Trust Region Policy Optimization (CSTR-PO)**

### 4.2 One-sentence thesis

Latent policy updates should be restricted to regions supported by reliable model experience, with a calibrated uncertainty threshold controlling how far an update may move.

### 4.3 Intuition

A latent policy can produce a mathematically valid vector that lies in a region where the model has no reliable reasoning behavior. CSTR-PO treats policy optimization like driving on a partially mapped road network: well-supported regions permit larger updates, while uncertain regions require smaller or rejected updates.

### 4.4 Formal object

Let `s(h, z)` be a nonconformity or unsupportedness score. A calibration set defines a threshold `q_alpha` intended to control a target error level. A candidate update is accepted or penalized according to

```text
s(h_t, z_t_new) <= q_alpha
```

and a trust-region constraint such as

```text
E[d(z_t_old, z_t_new | h_t)] <= epsilon(h_t),
```

where the radius becomes smaller when support uncertainty increases.

For this to be scientifically defensible, `d` and `s` should have functional meaning. Raw latent density or local PCA distance is coordinate-dependent and policy density is not equivalent to reasoning validity.

### 4.5 Candidate method pipeline

1. Fit a conditional support or validity score on replayed latent states and outcomes.
2. Calibrate rejection thresholds on held-out states.
3. Estimate calibration drift as the policy changes.
4. Reject, project, or downweight unsupported latent updates.
5. Combine support gating with a functionally defined trust region.
6. Recalibrate online or use time-uniform methods when exchangeability is not credible.

### 4.6 Potential contributions

- a functionally meaningful definition of latent support;
- policy-drift-aware calibration rather than static split conformal prediction;
- adaptive trust radii tied to calibrated risk;
- empirical prevention of latent policy collapse or catastrophic drift.

### 4.7 Experiments required

- In-distribution and shifted tasks with controlled levels of support mismatch.
- Calibration curves before and after policy updates.
- Coverage, selective risk, reward, exploration loss, and update rejection rate.
- Baselines from supported policy optimization, KL trust regions, density constraints, uncertainty penalties, conformal RL, and function-space constraints.
- Ablations separating the support model, calibration method, and trust-region metric.

### 4.8 Main literature collisions

- SPOT and STR already optimize within estimated behavior-policy support.
- APO already replaces global KL reasoning with high-confidence support coverage in LLM reinforcement learning.
- CCPO already combines conformal prediction and constrained policy optimization for LLM agents.
- Conformal off-policy evaluation already discusses validity under policy shift.

The likely reviewer interpretation is "supported trust-region optimization plus conformal calibration applied to latent actions." This makes it difficult to sustain as a headline contribution.

### 4.9 Compute profile

This direction is cheaper than BCG and SVCCO because a learned support model can screen updates without full suffix rollouts. However, high-dimensional conditional support estimation and reliable calibration require many held-out states. Cheap density estimation is not automatically scientifically valid.

### 4.10 Decision and recovery condition

**Current status: No-Go as a headline idea; retain as an engineering module.**

The best current role is a cheap first-stage gate inside a functional trust-region system. Revisit independently only if a new drift-valid conformal result or a genuinely functional notion of latent support emerges.

## 5. Idea D: Reparameterization-invariant latent policy optimization

### 5.1 Working title

**Latent Thoughts Have No Privileged Coordinates: Reparameterization Stress Tests and Functional Trust Regions for Latent Policy Optimization**

### 5.2 Core method name

**Functional Continuation Trust Region (FCTR)**

### 5.3 One-sentence thesis

Functionally equivalent coordinate descriptions of latent thoughts should induce the same behavioral optimization update; current coordinate-based latent operations can violate this principle, so policy constraints should instead be defined through observable continuation distributions.

### 5.4 Intuition

Take every latent vector `z`, transform it through an invertible map `u = phi(z)`, and apply the exact inverse before the original model consumes it. The transformed system computes the same function and produces the same outputs. Only the internal coordinate description changed.

An optimization algorithm should therefore behave equivalently in the `z` and `u` descriptions. However, Euclidean distance, cosine similarity, isotropic noise, linear interpolation, raw density, and PCA directions generally change under rotation, anisotropic scaling, or nonlinear reparameterization. An algorithm based on those objects may learn a different policy even though the underlying model is functionally identical.

### 5.5 Central invariance principle

For a smooth invertible chart `u = phi(z)`, the charted system applies `z = phi^{-1}(u)` before the original computation. Thus

```text
P(O | h, z) = P(O | h, phi^{-1}(u)), where u = phi(z).
```

The desired property is not that parameter vectors or gradients are numerically identical. It is that the optimization rule is equivariant and induces the same observable behavioral update after transporting between charts.

### 5.6 Failure hypotheses

The project should test the following preregistered hypotheses:

1. Function-preserving chart changes alter nearest neighbors selected by raw latent metrics.
2. Coordinate-isotropic perturbations induce different native perturbations and continuation distributions.
3. Interpolation, density thresholds, local PCA directions, and Euclidean trust regions produce chart-dependent decisions.
4. These differences persist under numerically moderate, well-conditioned transformations.
5. Chart dependence changes training stability, reward, branch selection, or method ranking, rather than only a visualization.
6. A functional continuation constraint substantially reduces those differences without merely tuning a chart-specific threshold.

### 5.7 Functional geometry

Define the functional continuation distance

```text
d_F(z, z' | h) = D(P(O_{t:H} | h, z), P(O_{t:H} | h, z')).
```

Because it is defined through observable continuations, `d_F` is invariant to an exact reparameterization of the hidden coordinate chart.

A local approximation may use a pullback Fisher metric

```text
G(z) = J_z^T F_O(z) J_z,
```

but neither Fisher geometry nor behavior equivalence should be claimed as new. The new claim must be the latent-reasoning audit, the demonstrated optimization consequence, and a sequential multi-horizon correction.

### 5.8 FCTR objective

A generic FCTR update is

```text
maximize_theta    E[R]
subject to        E_{h,z}[KL(P_old(O_{t:H} | h,z)
                              || P_new(O_{t:H} | h,z))] <= epsilon.
```

Practical estimators may form a ladder:

1. exact matched suffix rollouts as an oracle on a small audit set;
2. next-token KL as a cheap but myopic baseline;
3. local pullback-Fisher approximations;
4. amortized multi-horizon continuation sketches;
5. adaptive exact checks for samples near the trust-region boundary.

### 5.9 Planned contribution package

The paper should have one dominant story with five connected contributions:

1. Define reparameterization equivariance for latent policy optimization.
2. Release exact function-preserving chart stress tests.
3. Show that common latent operations and at least one training conclusion are chart-dependent.
4. Introduce FCTR with a sequential multi-horizon continuation estimator.
5. Provide a light theorem or bound connecting continuation divergence to finite-horizon value change, plus cross-architecture empirical evidence.

BCG supplies the behavioral metric foundation. CSTR-style support screening may reduce oracle calls. SVCCO remains outside the main paper unless it becomes necessary to demonstrate a specific chart-dependent failure.

### 5.10 Experiments required

#### Stage A: exactness and numerical controls

- Identity, orthogonal, anisotropic linear, affine, and later nonlinear invertible charts.
- Logit and reward identity before any coordinate-defined operation.
- Round-trip error, condition number, precision, and Jacobian checks.
- Float32, bfloat16 where appropriate, and float64 control experiments.

#### Stage B: operation audit

- Nearest-neighbor selection.
- Isotropic noise and perturbation norm.
- Linear interpolation and branch averaging.
- Raw density and support thresholds.
- Local PCA/tangent-normal operations.
- Euclidean or cosine trust regions.

#### Stage C: behavioral consequences

- Multi-horizon continuation divergence.
- Reward variance and branch-ranking flips.
- Training trajectory divergence under matched data and seeds.
- Final task quality, stability, calibration, and matched wall-clock compute.

#### Stage D: FCTR evaluation

- Exact suffix oracle versus cheap functional estimators.
- Raw-coordinate trust region, whitened metrics, next-token KL, Fisher approximations, and standard KL policy baselines.
- At least two latent reasoning architectures, three tasks, and three or more seeds for the main result.
- Sensitivity to horizon, trust radius, chart condition number, estimator budget, and policy drift.

### 5.11 Hard gates

The direction continues only if all scientific controls are satisfied:

1. The unmodified charted and native systems agree up to a preregistered numerical tolerance.
2. Meaningful chart dependence appears under moderate condition numbers, not only extreme ill-conditioning.
3. The effect survives precision controls and matched random numbers.
4. It changes an operational decision or training outcome, not only raw vector metrics.
5. Simple whitening, ordinary KL, or retuned thresholds do not fully remove the effect.
6. FCTR reduces chart sensitivity and improves or preserves task performance under matched compute.
7. The result reproduces in a second latent representation.

If gates 1-4 fail, stop the method paper. If gate 5 fails, report that a simpler correction is sufficient and do not oversell FCTR. If gates 6-7 fail, retain the work only as a critical analysis or workshop paper.

### 5.12 Main literature boundary

The direction inherits concepts from reparameterization invariance, natural gradients, information geometry, causal states, bisimulation, and pullback Fisher geometry. None of those concepts should be presented as original.

The defensible novelty target is their first systematic use as an exact function-preserving audit of latent reasoning optimization, together with a multi-horizon functional optimizer and evidence that the issue changes real learning conclusions.

### 5.13 Compute profile

- Toy and contract tests: CPU only.
- Initial 0.5B-1B phenomenon gate: approximately 20-80 A100-equivalent GPU-hours.
- Full inference-only chart audit: approximately 80-250 GPU-hours.
- 1B training pilot with multiple seeds: approximately 300-900 GPU-hours.
- Minimum main-paper evaluation: approximately 1,500-4,000 GPU-hours.
- Strong version including 7B and long-horizon tasks: approximately 4,000-10,000 GPU-hours.

These are planning ranges, not measured guarantees. Every experiment must report actual tokens, rollout calls, wall time, device type, and peak memory.

### 5.14 Decision

**Current status: highest-priority direction.**

It has the clearest falsifiable premise, best AAAI fit, strongest cross-field significance, and highest paper ceiling. Its novelty remains conditional: if the result reduces to "Euclidean distance is inferior to Fisher distance," the direction becomes incremental.

## 6. Comparative scorecard

Scores assume each idea is evaluated as the independent headline of an AAAI-style paper. Ten is best. Compute friendliness is also scored with ten meaning cheaper and easier.

| Direction | Novelty | Importance | AAAI fit | Theory ceiling | Experiment clarity | Collision safety | Compute friendliness | Idea ceiling | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BCG-PO | 5.0 | 8.5 | 8.0 | 8.0 | 7.5 | 4.0 | 3.5 | 8.0 | Conditional foundation |
| SVCCO | 4.5 | 8.5 | 7.5 | 7.0 | 6.5 | 3.0 | 1.5 | 8.0 | Preserve for later |
| CSTR-PO | 3.5 | 7.0 | 7.0 | 6.5 | 7.5 | 2.5 | 5.5 | Auxiliary module only |
| Reparameterization audit + FCTR | 7.5 | 9.0 | 9.0 | 9.0 | 9.5 | 7.5 | 3.5 | Primary direction |

## 7. Relationship among the four ideas

The ideas are not four unrelated projects. They form a research tree:

```text
Replayable latent-state harness
            |
            +-- Behavioral continuation geometry (BCG)
            |        |
            |        +-- Functional trust region (FCTR)
            |        +-- Safe branch merging as an application
            |
            +-- Support modeling
            |        +-- Cheap CSTR-style screening inside FCTR
            |
            +-- Valid interventions
                     +-- Future SVCCO credit assignment
```

The current paper should not claim all leaves as simultaneous contributions. The preferred single thesis is coordinate-invariant latent optimization. BCG is a required foundation, CSTR is an optional efficiency component, and SVCCO is a future direction.

## 8. Recovery guide for future work

### To restart BCG as the main idea

Begin from policy-family continuation signatures and metric-drift experiments. Do not begin by training another cosine replacement. The first gate is whether BCG predicts a real failure that next-token KL, Fisher distance, and whitening miss.

### To restart SVCCO as the main idea

Begin from a synthetic causal latent benchmark and define intervention validity before implementing a policy loss. The first gate is coordinate-invariant effect recovery under known ground truth.

### To restart CSTR-PO as the main idea

Begin from calibration under sequential policy drift, not static latent density. The first gate is a coverage or risk guarantee that existing supported policy optimization and conformal RL do not already provide.

### To continue the selected FCTR direction

Proceed in this order:

1. exact chart and round-trip contracts;
2. trace, replay, and matched-RNG continuation evaluation;
3. coordinate-operation baselines and functional metrics;
4. toy numerical controls;
5. 0.5B-1B inference-only phenomenon gate;
6. minimum functional trust-region estimator;
7. training pilot;
8. cross-architecture main evaluation;
9. theory, ablations, and paper packaging.

## 9. Key references and collision map

- Causal State Representations: <https://arxiv.org/abs/1906.10437>
- Deep Bisimulation for Control: <https://arxiv.org/abs/2006.10742>
- DeepMDP: <https://proceedings.mlr.press/v97/gelada19a.html>
- Approximate Policy Iteration with Bisimulation Metrics: <https://arxiv.org/abs/2202.02881>
- BiPACE: <https://arxiv.org/abs/2606.25556>
- FishBack: <https://arxiv.org/abs/2605.17231>
- IBPO: <https://arxiv.org/abs/2605.16302>
- CVT-RL: <https://arxiv.org/abs/2606.05263>
- Dynamics Within Latent Chain-of-Thought Reasoning: <https://arxiv.org/abs/2602.08783>
- Divergent Representations from Causal Interventions: <https://arxiv.org/abs/2511.04638>
- SPOT: <https://arxiv.org/abs/2202.06239>
- Supported Trust Region Optimization: <https://proceedings.mlr.press/v202/mao23c.html>
- APO: <https://arxiv.org/abs/2602.05717>
- CCPO: <https://arxiv.org/abs/2511.11828>
- Conformal Off-Policy Evaluation: <https://arxiv.org/abs/2304.02574>
- Latent-GRPO: <https://arxiv.org/html/2604.27998>

## 10. Frozen quality-first decision

The project will optimize for the strongest defensible scientific contribution rather than the fastest paper. The active direction is reparameterization audit plus FCTR. This decision should change only when evidence triggers one of the hard gates above or when a newly published direct neighbor removes the remaining novelty.

All negative results, failed gates, numerical artifacts, and compute costs must be retained. A direction will not be kept alive by changing thresholds after seeing results.

---

## 11. Dated evidence update: 2026-07-14

This section preserves the original four-idea archive while recording evidence that changed the active ranking. It supersedes Section 10 only for current execution; it does not erase the earlier rationale or make the archived ideas unavailable for later work.

### 11.1 Coordinate-audit result

The exact chart, trace, replay, matched-RNG, and compute-accounting infrastructure passed its engineering contracts on a public GPT-2 Coconut checkpoint. The first preregistered affine pilot failed its frozen frequency thresholds. A moderate nonlinear chart was then selected transparently from the pilot and tested on 100 disjoint examples.

The disjoint holdout also failed its frozen gate:

- `sinh scale=0.04` changed the Euclidean nearest neighbor for `16/100` examples;
- the 95% bootstrap interval was `[0.09, 0.23]`, missing the frozen lower-bound requirement of `0.10` by `0.01`;
- all no-op, orthogonal, numerical-condition, audit-size, continuation-divergence, and token-mismatch checks passed;
- changed decisions had large consequences, but their held-out frequency was not robust enough to carry the paper.

Per the preregistered rule, the nearest-neighbor evidence line is stopped. No stronger chart and no looser threshold will be selected.

### 11.2 Collision and target-validity update

Two additional findings weaken the original FCTR headline:

1. FishBack already derives pullback Fisher geometry, natural-gradient equivalence, and minimum-output-KL steering for Transformer activations. A method that merely extends this machinery to a latent-thought position risks looking like a direct sequential application rather than a new optimization principle.
2. The official Latent-GRPO action is a top-k vocabulary mixture produced from token probabilities and Gumbel-perturbed scores. Those token/simplex coordinates have canonical vocabulary semantics. Arbitrary hidden-state chart changes therefore do not directly invalidate the full official policy update in the way they can invalidate raw hidden-feedback methods such as Coconut.

The official implementation was inspected at commit `c0994fb781a2d180662bb522d8ff3e8638dcf56d`. It reconstructs top-k soft-token embeddings from saved perturbed scores and evaluates a product Gumbel surrogate in `verl/utils/torch_functional.py`. Its one-sided rollout noise is clipped and shifted, while the training objective intentionally remains a surrogate rather than an exact density. The advantage-dependent straight-through branch in the inspected code is applied only when the advantage is non-positive and the reconstructed margin is negative.

### 11.3 Consequence for Idea D

**Revised status: preserve as a critical-analysis branch, not the active method headline.**

The chart harness remains valuable and the broader invariance principle remains correct for hidden-feedback latent states. FCTR can be revived if a future training-level gate shows a stable cross-architecture failure that is not explained by FishBack, natural gradient, ordinary sequence KL, or a canonical simplex action model. The failed nearest-neighbor line alone is insufficient.

## 12. Idea E: Simplex-GRPO

### 12.1 Working title

**Soft Thoughts Live on a Simplex: Rao-Blackwellized Policy Optimization for Continuous Latent Reasoning**

### 12.2 Core method name

**Simplex-GRPO**

Alternative precise descriptor: **Quotient-Action Policy Optimization for Soft Thoughts**.

### 12.3 One-sentence thesis

Soft-thinking models execute a probability mixture on a vocabulary simplex, but current methods optimize a higher-dimensional auxiliary Gumbel perturbation whose common-shift degree of freedom has no behavioral effect; optimizing the induced simplex action distribution gives an exact, lower-variance, and better-conditioned policy objective without an advantage-dependent straight-through surrogate.

### 12.4 Intuition

For a fixed top-k vocabulary support, let the rollout construct perturbed scores

```text
a_i = log p_i + g_i,
q = softmax(a / tau),
z = sum_i q_i e_i.
```

The model consumes `z`, and both `q` and `z` are unchanged if the same scalar is added to every `a_i`. The common offset in `a` is therefore a nuisance variable: it changes the stored auxiliary action but not the soft thought or any downstream behavior.

SofT-GRPO explicitly treats `a` as the RL action and the softmax-to-embedding map as part of the transition. This makes the product Gumbel likelihood mathematically usable, but it spends policy-gradient variance and PPO clipping budget on a behaviorally invisible degree of freedom. Latent-GRPO further replaces the underlying noise with a clipped, shifted, one-sided construction and uses a custom surrogate backward rule.

Simplex-GRPO instead treats the executed mixture `q` as the continuous action. Under ordinary Gumbel-Softmax sampling, `q` has the known Concrete distribution, whose density is exact on the `(k-1)`-dimensional simplex and is invariant to common logit shifts.

### 12.5 Formal object

For positive parameters `alpha_i` and temperature `tau`, the Concrete density is

```text
p(q | alpha, tau)
  = Gamma(k) * tau^(k-1)
    * product_i [alpha_i * q_i^(-tau-1)]
    / (sum_j alpha_j * q_j^(-tau))^k.
```

It depends on the equivalence class of logits because multiplying every `alpha_i` by the same positive constant leaves the density unchanged.

Let `A` denote the auxiliary perturbed-score action and `Q = softmax(A/tau)` the executed action. For any reward that depends on `A` only through `Q`, the pushforward score satisfies the Fisher identity

```text
grad log p_Q(Q) = E[grad log p_A(A) | Q].
```

Consequently, the simplex score is a Rao-Blackwellization of the auxiliary score and cannot have larger conditional variance under the standard regularity assumptions. This does not claim that Gumbel reparameterization is invalid; it identifies an avoidable nuisance-action variance and a clipping mismatch.

### 12.6 Candidate method pipeline

1. During rollout, retain the selected vocabulary support and the executed mixture weights `q`, not only perturbed scores.
2. Evaluate the exact Concrete log density of `q` under the old and current policies on the same support.
3. Form the GRPO/PPO ratio from these simplex densities.
4. Remove the advantage-conditioned straight-through likelihood surrogate.
5. Control exploration with the Concrete temperature and a calibrated entropy or KL target.
6. For a complete sparse policy, extend the action to `(S, q)`: sample an ordered support `S` with a tractable without-replacement policy and sample `q` conditionally on `S`.
7. Compare the fixed-support conditional version and the full support-and-mixture policy before claiming exact end-to-end policy likelihood.

### 12.7 Potential theoretical contributions

- A precise distinction between auxiliary perturbation actions and executed simplex actions in soft-thinking RL.
- A pushforward-score/Rao-Blackwell variance result specialized to Gumbel-Softmax latent actions.
- A proof that unclipped importance sampling remains valid in either action space, while nonlinear PPO clipping generally does not commute with marginalizing the nuisance offset.
- Shift-invariance and normalization guarantees for the Concrete ratio.
- A support-change analysis showing when fixed-top-k conditional optimization is biased and when a joint support-and-mixture policy is required.
- A variance or effective-sample-size bound connected to temperature, support size, and policy drift.

### 12.8 Required experiments

#### Contract and synthetic tests

- Match the implemented Concrete log density against an independent distribution implementation.
- Verify logit-shift invariance, normalization, finite gradients, and low-temperature stability.
- Compare auxiliary and simplex score variance against a pathwise-gradient reference on controlled simplex rewards.
- Measure likelihood-ratio variance, effective sample size, clipping rate, and clipping-decision disagreement.
- Reproduce the inspected Latent-GRPO surrogate gradient on crossed and uncrossed margins for both advantage signs.

#### Real-model tests

- Start from an official released Latent-SFT or equivalent soft-thinking checkpoint, not this repository's hidden-projection approximation.
- Recompute rollout log ratios under SofT-GRPO, Latent-GRPO, exact Concrete, and ordinary discrete-token GRPO.
- Audit gradient variance, ratio tails, clip fraction, entropy, valid termination, and reward per rollout token.
- Run matched-compute training on at least two reasoning tasks and three seeds.
- Include a second model family and a support-change ablation before the main-paper claim.

### 12.9 Main literature collisions

- Gumbel-Softmax/Concrete already provides the density; that mathematics is not new.
- Rao-Blackwellized Gumbel estimators and clipped-action policy gradients already establish related variance-reduction principles.
- SofT-GRPO already defines the auxiliary Gumbel action and its off-policy ratio.
- Latent-GRPO already diagnoses exploration-optimization misalignment and introduces one-sided surrogate updates.
- NF-CoT already provides exact latent likelihoods through a much heavier normalizing-flow architecture.
- Dirichlet resampling has already been used for soft-thinking exploration.

The defensible novelty is the quotient-action diagnosis for soft-thinking PPO, its exact low-variance policy ratio, the support-and-mixture completion, and evidence that these differences change real latent-reasoning training rather than only a toy density calculation.

### 12.10 Compute profile

- Formula, variance, clipping, and synthetic gates: CPU only.
- Released 1B checkpoint ratio audit: one 24 GB GPU is sufficient in principle.
- 1B matched-compute training pilot: approximately 200-700 A100-equivalent GPU-hours depending on rollout length and engine reuse.
- Main evaluation with two model families, two to three tasks, and three seeds: approximately 1,500-5,000 GPU-hours.
- A 7B confirmation substantially raises the budget but is not needed before the 1B gate passes.

### 12.11 Hard gates

1. The exact Concrete implementation must match an independent reference and remain numerically stable.
2. The quotient score must show reproducible variance or effective-sample-size improvement over the auxiliary score; a purely aesthetic invariance result is insufficient.
3. PPO clipping decisions must differ often enough to plausibly affect optimization, not only on extreme synthetic logits.
4. A released soft-thinking checkpoint must show heavy-tailed or unstable auxiliary ratios that the simplex ratio materially improves.
5. The advantage-conditioned surrogate must not already dominate after fair temperature, KL, and clipping retuning.
6. At least one matched-compute 1B training task must improve stability or reward without reducing exploration and pass@k.
7. The result must survive support changes and a second model/task setting.

Failure of gates 1-3 stops the idea immediately. Failure of gate 4 reduces it to a theoretical note. Failure of gates 5-7 prevents a main-track method claim.

### 12.12 Decision

**Current status: active first-priority direction, conditional on the frozen CPU variance/clipping gate.**

This direction is more tightly grounded in the official algorithm than the hidden-chart audit, has a clearer method contribution than standalone BCG, and avoids presenting known pullback Fisher geometry as new. Its ceiling depends on whether quotient-space optimization produces a material training benefit; replacing one known density formula with another is not enough.

## 13. Revised comparative scorecard

These scores incorporate the failed coordinate holdout, the FishBack/TILR collision update, and the official action-space inspection.

| Direction | Novelty | Importance | AAAI fit | Theory ceiling | Experiment clarity | Collision safety | Compute friendliness | Idea ceiling | Current decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Simplex-GRPO** | **7.5** | **9.0** | **9.0** | **8.5** | **9.0** | **7.0** | **4.5** | **9.0** | **Primary, gated** |
| BCG-PO | 5.5 | 8.5 | 8.0 | 8.0 | 7.0 | 4.0 | 3.5 | 8.0 | Best fallback foundation |
| Reparameterization audit + FCTR | 5.5 | 8.5 | 8.0 | 8.5 | 7.0 | 4.5 | 3.5 | 8.0 | Preserve as critical-analysis branch |
| SVCCO | 4.0 | 8.5 | 7.5 | 7.0 | 6.5 | 2.5 | 1.5 | 7.5 | Preserve for later |
| CSTR-PO | 3.5 | 7.0 | 7.0 | 6.5 | 7.5 | 2.5 | 5.5 | 7.0 | Auxiliary module only |

## 14. Revised quality-first decision

The active work now follows this order:

1. freeze and run the Simplex-GRPO CPU density/variance/clipping gate;
2. stop immediately if the quotient score has no material statistical advantage;
3. if it passes, integrate an official released soft-thinking checkpoint and audit real rollout ratios;
4. implement the minimum exact simplex policy before adding support models, behavior geometry, or causal credit;
5. begin large training only after the real-checkpoint ratio gate passes;
6. retain every failed coordinate result as negative evidence and keep the chart harness available for hidden-feedback architectures.

BCG is the first fallback if Simplex-GRPO fails, but it must begin with a multi-horizon-versus-next-token phenomenon gate. FCTR is no longer the default fallback because its nearest-neighbor gate failed and its local Fisher method space is crowded.

## 15. Simplex-GRPO gate result: 2026-07-14

The frozen CPU gate in Section 12.11 failed and the headline direction is stopped.

- Exact-density reference error: `2.84e-14` (pass).
- Worst simplex-score versus pathwise-gradient relative error: `0.0328` (pass).
- Likelihood-ratio variance reduction: `1.264x` geometric mean (pass).
- Effective-sample-size improvement: `1.415x` geometric mean (pass).
- Gradient trace-variance reduction: `1.2196x`, below the frozen `1.25x` requirement (fail).
- PPO clipping-decision disagreement: `2.10%`, below the frozen `5%` requirement (fail).
- The inspected advantage-dependent surrogate remained directionally misaligned after a crossed margin for both advantage signs (diagnostic reproduced).

**Revised status: useful correction and baseline, not an AAAI headline.** The thresholds are not changed. The next active line is the predeclared BCG fallback: test whether next-token output geometry systematically misses multi-horizon continuation risk in recurrent latent reasoning. That gate must use a valid same-history estimator of joint continuation KL and must beat next-token KL, Fisher-style local metrics, and simple coordinate baselines before any learned BCG method is implemented.

## 16. BCG restart: estimator contract and refined method candidate

### 16.1 Correct finite-horizon target

For two latent interventions `z` and `z'` in the same replayable prefix `h`,
the directional finite-horizon target is

```text
KL(P_z(Y_1:H | h) || P_z'(Y_1:H | h))
  = E_{Y ~ P_z} [sum_t KL(P_z(. | h, Y_<t) || P_z'(. | h, Y_<t))].
```

The candidate must be teacher-forced on histories sampled from the reference.
Comparing logits along two independently sampled continuations is a coupled-path
diagnostic, not joint continuation KL. Temperature-scaled sampling also requires
temperature-scaled logits in the density calculation.

The implementation in `research/behavioral_geometry/joint_kl.py` enforces this
contract, supports shared reference rollouts across many candidates, records
per-step KL, and rejects deterministic decoding for this estimator.

### 16.2 Real-checkpoint integration result and protocol correction

The original v1 smoke and pilot used a fixed horizon after EOS. The resulting
post-EOS repeated answer tokens are outside the terminated continuation policy,
so both v1 artifacts are invalid for behavioral conclusions and are retained
only as debugging records.

The corrected public GPT-2 Coconut v2 smoke treats EOS as absorbing and passed
all measurement controls:

- identical latent: exact logits, exact forced tokens, and zero joint KL;
- norm-`0.5` latent perturbation: effective three-step joint KL `4.2813e-9`;
- numerically stable float64 first-step KL `7.1907e-20`;
- artifact: `artifacts/behavioral_geometry/public_gpt2_coconut_joint_kl_smoke_v2.json`.

This is explicitly **not scientific evidence**. It is one prompt and one
perturbation, retained only as an integration contract and a reason to run the
predeclared population gate.

### 16.3 Refined method candidate: finite-horizon continuation Fisher

If the population gate passes, the strongest BCG method form is not a generic
learned pairwise distance. It is a finite-horizon pullback metric for the joint
continuation distribution. For an infinitesimal latent displacement `delta`,

```text
KL(P_z(Y_1:H) || P_z+delta(Y_1:H))
  = 0.5 * delta^T G_H(h, z) delta + o(||delta||^2),
```

where the path Fisher decomposes over sampled shared histories:

```text
G_H(h, z)
  = E_{Y ~ P_z} [sum_t J_t(Y_<t)^T F_t(Y_<t) J_t(Y_<t)].
```

`F_t` is the categorical Fisher of the next-token distribution and `J_t` maps
the intervened latent to logits at the future prefix. The working method names
are **Continuation Fisher Geometry (CFG)** for the metric and **Successor-Fisher
Trust Region (SFTR)** for its policy-optimization use.

The high-ceiling contribution would require all of the following:

1. a reproducible latent-reasoning regime where exact next-token KL or its local
   FishBack metric ranks consequential updates as safe while finite-horizon KL
   ranks them as risky;
2. an efficient low-rank, sketched, or amortized estimator of `G_H` that retains
   this ranking advantage under matched compute;
3. a trust-region or latent-update rule that improves training stability or task
   reward over next-token KL, Euclidean, cosine, whitening, and FishBack-style
   one-step baselines;
4. replication in a second recurrent or soft latent-reasoning architecture.

### 16.4 Collision boundary after renewed search

- Predictive-state representations and Predictive-State Decoders already define
  recurrent state through distributions of future observations.
- Future Lens already tests how a hidden state predicts multiple future tokens.
- Sequence-level KL chain rules and path-space Fisher information are standard;
  neither can be claimed as new mathematics.
- Bisimulation and causal-state work already define behavioral equivalence by
  future outcomes.
- FishBack derives the pullback Fisher for a single softmax output and provides
  the strongest immediate local-output baseline.
- TILR studies stable directions in latent reasoning trajectories but does not
  define same-history finite-horizon continuation KL.

Therefore the defensible novelty is narrow: a demonstrated one-step blind spot
specific to recurrent latent reasoning, a practical finite-horizon estimator,
and a policy-optimization consequence. Without all three, this direction is an
analysis note rather than a competitive AAAI method paper.

### 16.5 Exploratory population result and stronger baseline

The corrected v2 training-split pilot evaluated 72 latent candidates on 12
GSM8K examples. H1 next-token KL looked weak against full terminated
continuation KL (Spearman `0.0887`, top-20% risk recall `0.20`). However, all 24
sampled continuations started with the deterministic formatting delimiter
`###`. This is an adverse confound, not a clean delayed-reasoning result.

A post-hoc audit found:

- cumulative H2 KL: Spearman `0.8830`, top-risk recall `0.80`;
- cumulative H3 KL: Spearman `0.9306`, top-risk recall `0.80`;
- every reference continuation began with `###`;
- artifact: `artifacts/behavioral_geometry/public_gpt2_coconut_bcg_pilot_v2_posthoc_prefix_audit.json`.

Therefore every future BCG gate must compare against H2/H3 and a
semantic-boundary prefix, not only H1. A fixed template token cannot establish
the claimed multi-horizon blind spot.

### 16.6 Current decision

**Status: conditional architecture screen; held-out gate not yet authorized.**
The single public GPT-2 GSM8K checkpoint does not yet justify BCG as an AAAI
method direction because its apparent H1 failure is mostly explained by a
two-token prefix. Before touching a disjoint evaluation set, screen a model or
mode with a substantive continuation and a second hidden-feedback architecture.
Freeze a held-out gate only if full continuation geometry retains an operational
advantage over H2/H3, semantic-prefix KL, coordinate distances, and one-step
FishBack-style geometry. Do not implement SFTR or an amortized metric before
that result exists.

Additional direct references:

- Predictive-State Decoders: <https://arxiv.org/abs/1709.08520>
- Future Lens: <https://arxiv.org/abs/2311.04897>
- Path-space information bounds: <https://arxiv.org/abs/1503.05136>
- Recurrent neural LMs as probabilistic finite-state automata: <https://arxiv.org/abs/2310.05161>

## 17. Idea F: Selection-Complete Latent Policy Optimization

### 17.1 Working title and method name

**Top-K Concrete Policies for Likelihood-Complete Latent Reasoning**

Working method name: **Selection-Complete Latent Policy Optimization
(SC-LPO)**. Distribution name: **Top-K Concrete (TKC)**.

### 17.2 One-sentence thesis

Sparse soft-thought policies execute the ordered top-k support and its mixture
weights, but the inspected Latent-GRPO objective evaluates selected order
statistics as independent Gumbels and omits the selection event; deriving the
exact induced action law permits normalized PPO ratios without a new latent
architecture.

### 17.3 Concrete failure in the inspected implementation

At official repository commit `c0994fb781a2d180662bb522d8ff3e8638dcf56d`,
the sampler:

1. constructs a policy-dependent top-p candidate set at
   `sglang/.../sampler.py:78-88`;
2. samples Gumbels and clips them to `[-1.5, 3]` at lines `91-92`;
3. selects top-k entries after perturbation at lines `105-110`;
4. executes their softmax mixture at line `122`.

The actor likelihood reconstructs each selected margin and evaluates a
standard marginal Gumbel log density at
`verl/utils/torch_functional.py:150-177`, then averages over `K` at line `179`.
It does not multiply by the event that every unselected score lies below the
K-th selected score. It also uses an advantage-dependent straight-through
gradient for a subset of negative-advantage margins.

Consequently, the forward value is not the normalized density of either the
stored selected order statistics or the executed top-k simplex action. The
clipped rollout noise further is not a continuous standard Gumbel law: it has
atoms at both clipping boundaries. The changing top-p set creates an additional
support-accounting requirement.

This is not an accusation that the Latent-GRPO paper presents the final
one-sided objective as exact. Section 3.2 explicitly describes it as a surrogate
latent likelihood rather than a mathematically exact probability density. The
research gap is whether a sampler-objective-consistent alternative yields
better calibrated and more effective optimization.

### 17.4 Exact law for the clean sparse policy

For independent unbounded Gumbels,

```text
A_i = l_i + sigma G_i,
S = (s_1, ..., s_K),       A_s1 > ... > A_sK,
q_r = exp(A_sr / tau) / sum_m exp(A_sm / tau).
```

Let `eta = tau / sigma` and `alpha_i = exp(l_i / sigma)`. The exact density over
the ordered support and the first `K-1` simplex coordinates is

```text
p(S, q)
  = Gamma(K) eta^(K-1) product_r[alpha_sr q_r^(-eta-1)]
    / [sum_r alpha_sr q_r^(-eta)
       + q_K^(-eta) sum_(j not in S) alpha_j]^K.
```

The final denominator term is the integrated effect of every unselected item.
The exact auxiliary density for selected scores is the product of selected
Gumbel densities times

```text
product_(j not in S) F_j(A_sK),
```

the missing top-k selection event.

Boundary checks are exact: `K=1` becomes a categorical distribution and `K=V`
becomes the ordinary Concrete distribution. The density is invariant to a
common logit shift and can be evaluated using selected logits plus a total
log-sum-exp, without a full-vocabulary action tensor.

### 17.5 Candidate method

The cleanest defensible method is not a drop-in likelihood change while leaving
the sampler untouched. It is a matched sampler-objective correction:

1. use unbounded standard Gumbel perturbations, with numerical stabilization
   that does not clip random variables into point masses;
2. define and record the candidate set, ordered support `S`, and executed
   weights `q` as the policy action;
3. compute old and current candidate-aware TKC densities, assigning zero ratio
   when a recorded support is impossible under the current candidate policy;
4. form token-level PPO or GRPO ratios from the exact joint law, without the
   advantage-dependent gradient flip;
5. compare the executed-action density with the complete selected-score
   auxiliary density as a Rao-Blackwellization and variance ablation;
6. use full vocabulary or a policy-independent candidate set in the first
   training study, then study dynamic top-p support separately.

An alternative exact clipped-noise method is possible in principle but must
handle mixed continuous and atomic components under top-k order statistics. It
is substantially more complex and currently has a worse quality-to-risk ratio.

### 17.6 Potential contributions

- the exact joint density of ordered top-k Gumbel support and normalized
  selected weights, specialized into a practical latent-policy objective;
- a proof that the inspected selected-Gumbel surrogate is not normalized after
  selection and that mean reduction is not a likelihood;
- a sampler-objective contract that handles candidate support explicitly;
- normalized off-policy ratios and unbiased score identities;
- a matched-compute demonstration that likelihood completion improves ratio
  calibration, clipping behavior, stability, or reasoning reward;
- a lighter analytic alternative to architecture-level exact-likelihood methods
  such as NF-CoT.

The density derivation alone is not enough for a main-track paper. The central
empirical burden is to connect the defect to training behavior and final
reasoning quality.

### 17.7 Collision boundary

- Gumbel-Top-k and Plackett-Luce work already gives exact ordered sampling
  without replacement.
- Concrete/Gumbel-Softmax already gives the full-vocabulary simplex density.
- SofT-GRPO and Latent-GRPO already use Gumbel perturbations for continuous
  latent reasoning.
- LEPO uses full-vocabulary Gumbel-Softmax with a soft-label objective rather
  than this sparse executed-action density.
- NF-CoT obtains exact likelihood through a new normalizing-flow architecture.

The currently defensible novelty is the normalized sparse support-and-mixture
law, the diagnosis of selection-incomplete latent PPO, and a drop-in-scale
sampler/objective correction. A dedicated prior-art search must continue before
claiming the Top-K Concrete density itself as new.

### 17.8 Frozen CPU gate result

The v1 gate is **fail/mixed**, not pass. Seven of eight checks passed, but the
minimum selection-omitting ratio-mean bias across five random policy drifts was
`0.0110`, below the frozen `0.03` threshold. Thresholds remain unchanged.

Positive diagnostics retained for deciding whether an official replay is worth
running:

- boundary-law error `1.71e-13` and shift error `2.66e-13`;
- exact ratio mean error at most `0.00328`;
- exact score mean norm at most `0.01275`;
- selection-omitting score mean norm at least `0.8165`;
- exact/official-style clip-decision disagreement `38.72%` on average;
- omitted-selection log-ratio correction RMS at least `0.1516`.

Independent `V=3, K=2` quadrature also verifies unit total mass for a
boundary-smooth setting. The failed all-check gate prevents a premature method
claim but does not erase the analytic normalization defect, which is tested
more directly by the nonzero expected surrogate score.

### 17.9 Compute profile

- density, quadrature, and synthetic replay: CPU only;
- official rollout replay on stored logits: CPU or one modest GPU;
- 1B checkpoint ratio audit: one 24 GB GPU in principle;
- 1B matched-compute pilot: roughly 200-700 A100-equivalent GPU-hours;
- main evidence across tasks, model families, and seeds: roughly 1,500-5,000
  A100-equivalent GPU-hours.

Large training is prohibited until the unchanged official replay and a clean
unclipped-sampler replay both show a material effect under frozen metrics.

### 17.10 Hard gates and stop conditions

1. Independent normalization, score-identity, and boundary checks must pass.
2. Official replay must separate selection omission, mean reduction, top-p
   support changes, clipping atoms, and the straight-through gradient.
3. A close prior exact-density or latent-RL method collision must not subsume
   the claimed contribution.
4. On real checkpoint logits, corrected and current objectives must differ in
   ratios, clipping, or gradients at operational scale.
5. A clean sampler/objective pilot must improve stability or reward under
   matched exploration, tokens, wall-clock, and compute.
6. Results must replicate across at least two tasks, three seeds, and a second
   model or latent-reasoning implementation.

Failure of gates 1-3 stops the direction. Failure of gate 4 reduces it to a
technical critique. Failure of gates 5-6 prevents an AAAI method-paper claim.

### 17.11 Current decision

**Status: primary audit candidate, not yet an authorized training method.**

SC-LPO currently has the best combination of concrete mathematical defect,
latent-reasoning specificity, tractable exact correction, and method-paper
ceiling among the archived ideas. It also carries a serious implementation
risk because the official sampler uses clipped noise and dynamic support. The
next justified step is an exact official-code replay, not large-scale training.

### 17.12 Updated comparative scorecard

| Direction | Novelty | Importance | AAAI fit | Theory ceiling | Experiment clarity | Collision safety | Compute friendliness | Idea ceiling | Current decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FTK-PO / support-mixture trust regions | 7.8 | 9.0 | 9.2 | 9.2 | 8.5 | 7.0 | 5.0 | 9.3 | **Stopped after frozen Gate B** |
| SC-LPO / Top-K Concrete | 8.0 | 9.0 | 9.0 | 9.0 | 9.0 | 7.5 | 5.0 | 9.2 | **Density result retained; method framing stopped** |
| BCG-PO / CFG-SFTR | 5.5 | 8.5 | 8.0 | 8.0 | 7.0 | 4.0 | 3.5 | 8.0 | Conditional architecture screen |
| Simplex-GRPO, fixed support | 6.0 | 8.0 | 7.5 | 7.5 | 9.0 | 7.0 | 5.0 | 7.8 | Correction/baseline only |
| Reparameterization audit + FCTR | 5.5 | 8.5 | 8.0 | 8.5 | 7.0 | 4.5 | 3.5 | 8.0 | Critical-analysis branch |
| SVCCO | 4.0 | 8.5 | 7.5 | 7.0 | 6.5 | 2.5 | 1.5 | 7.5 | Preserve for later |
| CSTR-PO | 3.5 | 7.0 | 7.0 | 6.5 | 7.5 | 2.5 | 5.5 | 7.0 | Auxiliary module only |

### 17.13 Official-default replay and thesis narrowing

The frozen official-default replay failed two of seven conjunctive checks:

- the weakest seed's clean selection-correction RMS was `0.0813`, below `0.1`;
- the one-sided shift changed old/current ratios by only `0.0052-0.0114` RMS,
  far below `0.1`;
- the conditional straight-through branch triggered on zero components under
  the default one-sided rollout and local policy drift.

These are adverse results. Selection completion and one-sided correction cannot
individually carry the paper.

Three effects remained large:

- `27.9%-35.1%` of selected entries lay on the upper clipping atom;
- clipping changed the complete ordered top-k support in at least `98.65%` of
  paired clean/clipped draws, with `81.8%-83.5%` mean item overlap;
- replacing a joint sum by `sum/K` changed PPO clipping on `36.53%` of samples;
- dynamic top-p support made `7.69%` of old actions impossible under the
  current candidate set on average;
- on the clean unbounded sampler, exact versus official-style likelihoods
  changed clipping decisions on `37.59%` of samples.

The implementation lineage matters. SofT-GRPO commit
`8d3c61380b15c3400818da5ce41c62c293a1bfb4` fixes top-k support before adding
Gumbel noise. Latent-GRPO commit
`c0994fb781a2d180662bb522d8ff3e8638dcf56d` perturbs a top-p candidate set and
then selects top-k. The missing selection event is specific to the latter
post-noise support policy; it should not be generalized to all soft-thinking
methods.

**Revised status: stop the selection-only SC-LPO framing.** Preserve TKC inside
a broader but still narrow sampler-likelihood-consistency method. A new gate may
be preregistered only after the prior-art audit, and it must focus on clipping
atoms, support validity, and trust-region calibration under real checkpoint
logits. No training is authorized by either failed synthetic gate.

## 18. Idea G: Factorized Top-K Policy Optimization

### 18.1 Working title and method name

**Factorized Trust Regions for Sparse Latent Policies**

Working method name: **Factorized Top-K Policy Optimization (FTK-PO)**.
Interpretive name: **support-mixture trust-region optimization**.

### 18.2 Plain-language idea

A sparse latent action makes two decisions at once:

1. which `K` vocabulary items enter the latent thought, and in what order;
2. how much probability weight each selected item receives.

Existing code compresses all selected-component scores with `sum/K`. A direct
exact correction would instead use one joint likelihood. Both treatments still
give the optimizer only one control knob. FTK-PO keeps the exact joint policy
ratio but separately constrains changes to the selected support and changes to
the mixture weights. In plain terms, the model may reweight a familiar latent
vocabulary without being forced to change its membership at the same rate, and
vice versa.

### 18.3 Exact factorization

The Top-K Concrete action from Idea F has the exact decomposition

```text
p_l(S, q) = P_l^PL(S) p_l(q | S),
```

where the ordered-support probability is the Plackett-Luce law

```text
P_l^PL(S)
  = product_(r=1)^K
      exp(l_sr / sigma)
      / sum_(j not in {s_1,...,s_(r-1)}) exp(l_j / sigma).
```

The conditional mixture density is obtained without approximation as

```text
log p_l(q | S) = log p_l(S, q) - log P_l^PL(S).
```

Therefore the old-policy KL obeys the chain rule

```text
KL(p_old(S,q) || p_new(S,q))
  = KL(P_old(S) || P_new(S))
    + E_(S~P_old) KL(p_old(q|S) || p_new(q|S)).
```

The implementation validates this factorization per sample, verifies that all
ordered supports normalize, and independently integrates each fixed-support
density back to its Plackett-Luce mass.

### 18.4 Candidate method

The conservative method version is a constrained policy update, not a product
of independently clipped pseudo-ratios:

1. sample an unbounded-Gumbel Top-K Concrete action over the full vocabulary or
   a policy-independent candidate set;
2. optimize the usual reward surrogate with the **exact joint importance
   ratio** `exp(Delta log p(S,q))`;
3. estimate the ordered-support KL and conditional-mixture KL separately from
   old-policy actions;
4. maintain separate target budgets and dual penalties for the two KL terms;
5. adapt budgets only by a preregistered controller, while reporting their sum
   as the exact joint KL;
6. compare against the released `sum/K` surrogate, an exact-joint single-budget
   baseline, and matched-strength generic KL/PPO baselines.

Separate clipping of the two ratios is an optional ablation, not the primary
method, because component-wise clipping would itself create a new biased
surrogate. Dynamic top-p and clipped Gumbel atoms are excluded from the first
method study; they require explicit support bookkeeping or mixed-measure
likelihoods and would obscure the central test.

### 18.5 Why this is not merely Idea F renamed

Idea F asked whether adding the omitted selection event was itself a material
method. Two frozen gates rejected that framing. FTK-PO instead uses the exact
law to expose and control the hierarchy of the executed action. Its hypothesis
is not “the missing term is large”; it is “support drift and conditional-weight
drift are distinct optimization resources that a single scalar calibration can
hide or misallocate.”

This is a genuinely new hypothesis and must pass a genuinely new gate. The
earlier `37.6%` exact/surrogate and `36.5%` mean/sum clip disagreements motivate
the question but do not count as evidence for factorized control.

### 18.6 Potential paper contributions

- the exact Top-K Concrete joint policy and its support/mixture factorization;
- a KL chain-rule view of sparse latent-policy updates;
- a two-budget trust-region optimizer that keeps the exact joint ratio;
- diagnostics for hidden component drift and cancellation inside a seemingly
  small joint update;
- a matched-compute test of whether controlling latent support separately from
  latent mixture improves stability, exploration, and reasoning reward;
- an analytic, architecture-preserving alternative to exact-likelihood latent
  policies that require a new flow model.

The first two bullets alone are not enough for a high-quality method paper.
The optimizer must outperform exact-joint single-budget control, not only the
released surrogate.

### 18.7 Collision boundary and novelty risk

The ingredients individually have substantial prior art: Gumbel Top-k induces
Plackett-Luce sampling, Concrete distributions model continuous simplex
actions, KL chain rules are classical, and hierarchical/factorized policies
have used component-wise constraints. The claim must therefore stay specific:

> an exact sparse vocabulary-superposition policy, decomposed into ordered
> support and conditional mixture, with separately controlled trust regions for
> latent-reasoning policy optimization.

A paper that already combines this exact action law and this optimization
decomposition would stop or materially narrow the direction. NF-CoT is the
closest high-level collision on exact latent likelihood, but it changes the
latent architecture with normalizing flows rather than deriving the existing
sparse vocabulary-mixture policy.

### 18.8 Compute profile

- exact identities and factor-drift diagnostics: CPU only;
- real-logit factor audit: checkpoint download plus CPU inference or one modest
  GPU;
- controlled 1B pilot: approximately 300-900 A100-equivalent GPU-hours;
- high-quality multi-task evidence: approximately 2,000-6,000
  A100-equivalent GPU-hours, depending on sequence length and baselines.

These are planning ranges, not measured costs. No training allocation is
authorized before the synthetic and real-logit gates pass.

### 18.9 Hard gates and stop conditions

1. Exact factorization, ordered-support normalization, and per-support
   integration must pass independent tests.
2. Across frozen synthetic regimes, both KL components must be non-negligible
   and a material fraction of updates must hide a component-level trust-region
   violation through cancellation in the joint log ratio.
3. A focused prior-art audit must find no method that already combines the same
   sparse action law and factorized trust-region optimization.
4. On preregistered real-checkpoint prompts, support and mixture drift must show
   stable, operationally distinct behavior across layers, token positions, and
   at least two checkpoints or model conditions.
5. Under matched joint KL, tokens, wall-clock, and exploration, FTK-PO must beat
   both the released surrogate and exact-joint single-budget optimization on
   stability or reward without sacrificing final task quality.
6. The gain must replicate on at least two reasoning tasks, three seeds, and a
   second model or latent-reasoning implementation.

Failure of gate 2 stops the factorized optimizer while preserving TKC as an
analysis tool. Failure of gate 4 stops training. Failure of gate 5 reduces the
work to theory/diagnostics rather than an AAAI-level method paper.

### 18.10 Current decision

**Status: primary method candidate, synthetic gate pending.**

This is currently the highest-ceiling direction because it converts the exact
distribution result into a falsifiable optimization method and directly tests
against the strongest corrected baseline. It is also high risk: generic
factorized-policy and constrained-RL literature may narrow novelty, and the two
components may prove too coupled for separate budgets to help. Those risks are
the next work items, not details to defer until after training.

### 18.11 Frozen Gate B result and final decision

The preregistered 135-scenario separability gate is **fail**. The configuration
hash is
`50bb47b36249dacb04e877f1a0326a33ee868c49a8647e82dd1ec916bf1fcdb5`.
Five of nine checks passed; four method-necessity checks failed:

| Diagnostic | Observed | Frozen requirement | Result |
|---|---:|---:|---|
| Both components material | `8.89%` of scenarios | at least `75%` | **fail** |
| Hidden component violations | `2.03%` mean; `22.96%` of scenarios above `3%` | at least `5%` mean and `60%` of scenarios | **fail** |
| Support-share interdecile range | `0.0925` | at least `0.15` | **fail** |
| Matched-KL measurement error | `0.736` worst independent estimate | at most `0.30` | **fail** |
| Opposite-sign component ratios | `49.72%` mean | at least `25%` | pass |
| Exact factorization error | `1.42e-14` max | at most `1e-10` | pass |
| KL chain residual | `4.64e-17` max | at most `1e-10` | pass |
| Exact ratio mean error | `0.00943` max | at most `0.06` | pass |
| Minimum component KL estimate | `-1.90e-4` | at least `-0.005` | pass |

The substantive failure is not calibration noise. The ordered support accounts
for a median `99.84%` of joint KL. Mean support shares are `98.89%`, `97.61%`,
and `94.80%` for `K=2`, `K=5`, and `K=10`. Conditional-mixture drift becomes
occasionally material only for the most concentrated logit profile; selecting
that subset after seeing the result would violate the gate.

**Final status: stop FTK-PO as a primary method.** Do not download checkpoints
or run training for this direction. Retain the exact decomposition as a theory
and diagnostic result: locally, the Top-K Concrete policy's information change
is overwhelmingly carried by its ordered Plackett-Luce support in broad
synthetic regimes. This result may inform a later support-centric baseline but
does not independently justify an AAAI method paper.

## 19. Idea H: Score-Squashed Gumbel Policy Optimization

### 19.1 Working title and method name

**Bounded Exploration without Broken Likelihoods: Score-Squashed Policies for
Latent Reasoning**

Working method name: **Score-Squashed Gumbel Policy Optimization (SSG-PO)**.

### 19.2 One-sentence thesis

Hard-clipping additive Gumbel noise creates policy-dependent atoms and changes
the selected vocabulary support, whereas a smooth monotone squash of the full
Gumbel-perturbed score preserves ordinary Gumbel Top-K exploration exactly,
bounds the executed mixture, and admits a common-support exact likelihood for
policy optimization.

### 19.3 Plain-language idea

The released Latent-GRPO sampler first clips each random Gumbel perturbation and
then adds it to a token log-probability. This both caps extreme exploration and
changes which tokens win. It also puts positive probability on exact clipping
boundaries. When model logits change, those boundary points move, so an old
boundary action generally has zero probability under the new policy. An
ordinary PPO importance ratio is then not mathematically available for that
positive-mass part of the rollout distribution.

SSG-PO reverses the design order:

1. form the standard score `x_i = log p_i + sigma G_i`;
2. apply the same strictly increasing smooth squash `y_i = h(x_i)` to every
   complete score;
3. select Top-K and execute `softmax(y_S / tau)`.

Because `h` is increasing, sorting `y` gives exactly the same ordered support as
sorting the original unbounded Gumbel scores. Because `h` maps the real line to
one fixed open interval, all finite-logit policies have positive density on the
same action support. The mixture weights are nevertheless bounded away from
arbitrarily extreme score gaps.

### 19.4 Canonical policy

Normalize model logits first:

```text
l_i = log_softmax(raw_logits)_i,
x_i = l_i + sigma G_i,        G_i ~ Gumbel(0,1),
y_i = m + s tanh(x_i / s),
S = ordered TopK(y) = ordered TopK(x),
q = softmax(y_S / tau).
```

The default bounds inherited for a fair first comparison are
`m-s=-1.5` and `m+s=3.0`, so `m=0.75`, `s=2.25`. This choice gives unit slope at
zero and is fixed before empirical evaluation. The common shift `m` cancels in
the executed softmax, while log-softmax supplies a canonical score origin.

For an observed squashed selected score `y`, define

```text
x = h^{-1}(y) = s atanh((y-m)/s).
```

For ordered support `S=(s_1,...,s_K)` and `y_s1>...>y_sK`, the exact augmented
action density is

```text
p(S,y_S)
  = product_r [f_G((x_r-l_sr)/sigma) / sigma * |dx_r/dy_r|]
    * product_(j not in S) F_G((x_K-l_j)/sigma).
```

The second product is the complete Top-K selection event. The Jacobian is
`|dx/dy| = 1 / (1-((y-m)/s)^2)` for the default squash.

### 19.5 Absolute-continuity claim

The key theorem target is not that clipping is numerically inelegant.

- A clipped additive location variable has atoms at `l_i + sigma a` and
  `l_i + sigma b`. For a nonzero logit update, the atom locations move. Except
  for exact coincidences, an old atom is a positive-mass set with zero mass
  under the new policy, so old and new selected-score measures are not mutually
  absolutely continuous.
- More generally, bounded additive noise has a support interval translated by
  the policy location, producing support mismatch under updates.
- SSG-PO applies an invertible map to the **complete score**, so every policy
  has the same open score interval. Standard Gumbel's full support gives a
  strictly positive density throughout it. Old/current importance ratios are
  therefore well defined almost everywhere.

The paper must state the action representation precisely. These claims concern
the stored ordered selected scores as an augmented policy action whose executed
effect is the deterministic mixture `q`.

### 19.6 Candidate method

1. Use the full vocabulary or a policy-independent candidate mask; dynamic
   top-p is excluded from the first clean method.
2. Sample ordinary unbounded Gumbels and canonical normalized scores.
3. Smoothly squash complete scores before mixture construction.
4. Store the ordered support and squashed selected scores.
5. Evaluate old/current exact order-statistic densities, including the
   unselected CDF event and squash Jacobian.
6. Optimize an exact joint PPO/GRPO ratio, with measured KL or early stopping
   as the primary trust control.
7. Compare against released hard-clipped Latent-GRPO, unbounded exact TKC,
   unclipped SofT-GRPO, and a generic squashed-policy baseline.

The primary method will not use a one-sided pseudo-score or independently clip
component ratios. Invalid-sample masking and correct-path selection can be
included only as matched modules shared by all relevant baselines.

### 19.7 Potential contributions

- a proof that hard-clipped latent score policies lack ordinary cross-policy
  importance ratios on their moving atomic mass;
- an exact selected-order-statistic likelihood after a fixed-support monotone
  score transform;
- a sampler that separates support exploration from mixture-range control:
  support remains exact Gumbel Top-K while weights are bounded;
- a probability-valid alternative to one-sided target-margin surrogates;
- a matched-compute demonstration of stable latent RL without sacrificing
  support diversity or pass@k.

The theory alone is not enough. The method must beat an exact unbounded policy
and carefully retuned released baselines on real training.

### 19.8 Collision boundary

- Tanh-squashed Gaussian policies and Jacobian-corrected bounded actions are
  standard in continuous-control RL.
- Gumbel Top-K and Plackett-Luce ordered support sampling are established.
- General order-statistic densities and transformed distributions are
  classical.
- An ICLR 2025 analysis already shows that truncated finite-precision Gumbels
  alter categorical sampling and diversity.
- SofT-GRPO, LEPO, Latent-GRPO, and NF-CoT already provide competing latent RL
  parameterizations.

The defensible novelty is their specific combination: monotone transformation
of complete vocabulary scores to preserve sparse Gumbel support exactly,
common-support order-statistic likelihoods for latent PPO, and evidence that
this resolves the exploration/stability trade-off. A direct prior method with
that combination stops or narrows the claim.

### 19.9 Compute profile

- derivation, normalization checks, and synthetic sampler gate: CPU only;
- released-checkpoint rollout audit: CPU inference or one modest GPU;
- matched 1B pilot: approximately 300-900 A100-equivalent GPU-hours;
- main evidence with two architectures, multiple tasks, and seeds:
  approximately 2,000-6,000 A100-equivalent GPU-hours.

### 19.10 Hard gates and stop conditions

1. Exact density, Jacobian, support identity, score identity, and importance
   ratio normalization must pass independent tests.
2. With fixed preregistered squash bounds, SSG must preserve unbounded Top-K
   support exactly while matching the hard-clipped policy's mixture-concentration
   regime without widespread squash saturation.
3. A focused prior-art audit must find no direct score-squashed Gumbel latent
   policy method.
4. On public checkpoint logits, hard clipping must create material atomic mass
   or support distortion, and SSG must retain meaningful rollout validity and
   diversity rather than only pass algebraic checks.
5. Under matched tokens, compute, KL, and exploration, SSG-PO must outperform
   exact unbounded policy optimization and the released surrogate on stability
   or reward without losing final quality or pass@k.
6. Gains must replicate across at least two tasks, three seeds, and a second
   latent-reasoning implementation or model condition.

Failure of gate 1 stops the idea. Failure of gate 2 prevents checkpoint work.
Failure of gate 4 prevents training. Failure of gate 5 reduces the result to a
technical analysis rather than a competitive method paper.

### 19.11 Current decision and provisional score

**Status: primary mathematical method candidate; Gate 1 passed, Gate 2
pending.**

| Novelty | Importance | AAAI fit | Theory ceiling | Experiment clarity | Collision safety | Compute friendliness | Idea ceiling |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7.8 | 9.0 | 9.0 | 9.2 | 8.8 | 6.5 | 5.0 | 9.2 |

This direction is preferred over rescuing FTK-PO on a post hoc concentrated
subset. It targets the largest surviving source-replay effect and supplies a
clean probability contract. Its main risks are generic squashed-policy
precedent, saturation near fixed bounds, and the possibility that hard clipping
is empirically beneficial precisely because it changes support.

### 19.12 Gate 1 result

The exact selected-score implementation passes all targeted mathematical tests:

- the squash/inverse round trip is accurate to floating-point tolerance;
- score squashing preserves every ordered Top-K support under paired Gumbels;
- canonical log-softmax scores are invariant to a common raw-logit shift;
- `K=1` matches category probability times the transformed Gumbel-maximum law;
- `K=V` matches the ordered product of independently transformed Gumbels;
- the density agrees pointwise with an independent selected-Gumbel density plus
  the transform Jacobian;
- a 100,000-sample old/current importance ratio has unit expectation within the
  frozen numerical tolerance;
- the Monte Carlo mean policy score is near zero;
- fixed candidate masks retain finite cross-policy densities.

The complete relevant suite has 25 passing tests. Gate 1 establishes the
probability contract only. It does not show that the fixed squash controls
mixture exploration as effectively as hard clipping; that is the purpose of the
next frozen gate.

### 19.13 Frozen Gate 2 result and final decision

The preregistered support/concentration gate is **fail**. Its configuration
SHA-256 is
`c62139fb723c6263f13ad5d12455a9742e4bf251341967d44caa384cf6cf1e51`.
Six of seven checks passed:

| Diagnostic | Observed | Frozen requirement | Result |
|---|---:|---:|---|
| SSG/unbounded ordered-support mismatch | `0` | `0` | pass |
| Exact ratio mean error | `0.00918` max | at most `0.03` | pass |
| Exact score mean norm | `0.0589` max | at most `0.12` | pass |
| Hard-clip support change | `93.64%` min; `98.28%` mean | at least `50%` | pass |
| Selected upper atom | `18.58%` min | at least `10%` | pass |
| Squash saturation | `0.101%` max | at most `10%` | pass |
| Support-inclusion entropy gain | `0.540` mean; positive in `100%` | at least `0.01`; positive in `80%` | pass |
| Hard-clip/SSG mixture entropy gap | `0.315` max | at most `0.15` | **fail** |
| Hard-clip/SSG 95th-percentile max-weight gap | `0.414` max | at most `0.08` | **fail** |

The two failed values form one conjunctive check: the fixed squash does not
match the hard-clipped policy's mixture-concentration regime. The discrepancy
is largest for diffuse logits, where SSG produces much sharper mixture weights.
Although SSG gives a clean likelihood and materially more diverse support, it
does not isolate support preservation from mixture control as hypothesized.

**Final status: stop SSG-PO as a method.** Do not tune temperature or bounds
after this result and do not proceed to checkpoint download or training. Retain
the absolute-continuity theorem, exact transformed order-statistic density, and
the empirical exploration trade-off as components of a broader policy-contract
analysis.

## 20. Idea I: Latent Policy Contract Audit

### 20.1 Working title and research type

**What Is the Policy in Latent Reasoning? A Contract Audit of Samplers,
Actions, and Likelihood Ratios**

Working name: **Latent Policy Contract Audit (LPCA)**.

This is primarily a theory, analysis, and benchmark paper, not a new optimizer
paper. Contract-compliant reference policies are controls and repair examples,
not automatically the headline method.

### 20.2 One-sentence thesis

Continuous latent-reasoning RL methods often optimize a stored auxiliary
quantity, execute a coarsened mixture, and modify sampling through truncation,
clipping, or policy-dependent support; a policy-gradient claim is interpretable
only after the action, induced measure, likelihood, and old/current support are
shown to satisfy one explicit contract.

### 20.3 The contract

LPCA evaluates every method against six questions:

1. **Action semantics:** What random variable is the policy action: full noise,
   selected scores, ordered support, simplex weights, hidden state, or a mixed
   object?
2. **Execution map:** Is the downstream latent state a deterministic measurable
   function of that stored action and replayable prefix state?
3. **Normalization:** Does the claimed likelihood integrate or sum to one, have
   zero expected score, and produce unit-mean old-to-new importance ratios?
4. **Support:** Are old and current measures mutually absolutely continuous on
   reused rollouts, including atoms, truncation, top-p masks, and deterministic
   selection?
5. **Sampler-likelihood consistency:** Is the evaluated density induced by the
   actual released sampler, or is it a declared surrogate with separately
   measured behavior?
6. **Aggregation:** Are component, latent-step, and trajectory scores combined
   as a valid joint law, or are averaging and clipping choices changing the
   optimization object?

Passing these questions does not imply high reward. Failing them does not imply
that a heuristic cannot work. It determines which probability and PPO claims
are valid and which empirical behavior requires another explanation.

### 20.4 Existing assets that make the audit concrete

- an exact Top-K Concrete density over ordered support and executed weights;
- an exact selected-order-statistic density including the unselected CDF event;
- a Plackett-Luce support/conditional-mixture decomposition;
- a proof and implementation for fixed-support score squashing;
- tests for boundary laws, normalization, expected score, shift invariance,
  candidate masks, and importance ratios;
- source-faithful Latent-GRPO and SofT-GRPO sampler replays;
- measured hard-clipping atoms, support distortion, dynamic top-p mismatch,
  `sum/K` scaling effects, and straight-through trigger behavior;
- failed repair gates showing that mathematically clean replacements do not
  automatically preserve the released sampler's useful behavior.
- a frozen Stage A audit on all 500 MATH-500 prompts with Qwen2.5-3B: all five
  numerical controls and all five semantic-effect gates passed, establishing
  that the action/likelihood mismatch is common on public checkpoint logits;
- preregistered Stage B sequential-history and reward-conditioned
  score-direction audits, with
  thresholds frozen before full evaluation.

The negative method results strengthen the audit: they rule out the simplistic
story that one algebraic correction is an obvious drop-in improvement.

### 20.5 Planned coverage

At minimum, audit:

1. SofT-GRPO;
2. LEPO;
3. Latent-GRPO;
4. exact sparse Gumbel references from this repository;
5. NF-CoT at the formal level, and at source level if code/checkpoints become
   available;
6. Dropout-GRPO as a formal positive control for an augmented stochastic
   action with rollout/update replay, and at source level if code becomes
   available.

Routing methods such as TARPO and SWITCH form a useful contrast because their
stochastic policy decisions remain discrete and their latent recurrence can be
trained through ordinary output-token likelihoods.

### 20.6 Candidate contribution package

- a formal taxonomy connecting sampler, stored action, executed action, and
  likelihood measure in latent reasoning;
- necessary contract tests for normalization and off-policy reuse;
- exact sparse vocabulary-mixture laws and hard-clipping singularity results;
- a public source-pinned audit suite with minimal counterexamples;
- real-checkpoint measurements of how often each violation affects ratios,
  clipping, gradients, support, termination, and reasoning diversity;
- contract-compliant reference implementations that separate mathematical
  validity from empirical utility;
- practical reporting standards for future latent-policy papers.

The paper should not claim that generic score identities, action squashing,
absolute continuity, discrete Top-K support alignment, exact continuous-thought
likelihood, or generic latent-representation auditing are new. The defensible
novelty target is a source-pinned audit of the *policy semantics* of stochastic
latent reasoning: the joint contract among sampled action, executed latent,
stored rollout object, support, likelihood ratio, proxy history, and
reward-conditioned update. New exact sparse-action results are included only
where that contract audit requires them.

### 20.7 Collision boundary

- Continuous-control RL already studies clipped versus squashed actions,
  transformed densities, support mismatch, and high-dimensional PPO ratios.
- Gumbel and order-statistic theory already supplies many mathematical tools.
- Latent-GRPO explicitly labels its one-sided objective a surrogate and already
  foregrounds density/sampling mismatch.
- NF-CoT already motivates exact likelihood as an architecture design goal.
- Dropout-GRPO already defines a replayable augmented latent action and proves
  its old-policy group surrogate targets a mask-marginalized objective.
- RLPT already diagnoses and repairs rollout-versus-optimization support
  mismatch for discrete Top-K token policies.
- CTPO and NFPO already study prefix or multi-step likelihood ratios.
- *Formalizing Latent Thoughts: Four Axioms of Thought Representation in LLMs*
  already owns a broad formal and empirical audit of intrinsic latent
  representations. LPCA must not claim the first axiomatic, intrinsic, or
  cross-model latent-thought audit; its boundary is policy semantics and
  off-policy/reward-conditioned updates rather than representation quality.
- Training-inference and sampler-policy mismatch are active topics in LLM RL.

No directly matching cross-method audit of the complete stochastic
latent-policy contract was found in the targeted search as of 2026-07-14. This
is a narrower claim than "latent audit" and remains conditional on continued
collision search and source-level verification.

### 20.8 Required evidence and stop conditions

1. Reproduce each audited method from a pinned official source, not only paper
   equations.
2. Include at least three stochastic latent-policy families and one exact or
   discrete contrast.
3. Confirm every claimed violation analytically or with an independent
   normalization/support test; implementation disagreement alone is not enough.
4. Show operational effects on public checkpoint logits and rollouts, not only
   iid Gaussian logits.
5. Separate acknowledged surrogate choices from accidental inconsistencies and
   give authors' intended rationale fairly.
6. Demonstrate at least one downstream consequence or robust predictor of
   instability, exploration, or performance under matched settings.
7. Release all negative repairs and sensitivity analyses so the audit is not a
   one-sided critique.

Failure of gates 1-3 stops the paper. Failure of gate 4 reduces it to a theory
note. Failure of gate 6 prevents a strong AAAI main-track claim.

### 20.9 Compute profile and provisional score

- source and synthetic audits: CPU only;
- checkpoint logit and rollout matrix: approximately 50-250 GPU-hours;
- matched small training diagnostics, if needed: approximately 500-1,500
  A100-equivalent GPU-hours;
- a broad cross-method study: approximately 1,500-4,000 GPU-hours.

| Novelty | Importance | AAAI fit | Theory ceiling | Experiment clarity | Collision safety | Compute friendliness | Idea ceiling |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8.0 | 9.2 | 8.8 | 9.2 | 9.0 | 7.5 | 6.0 | 9.1 |

### 20.10 Current decision

**Status: active primary direction.**

The source-pinned method matrix and Stage A public-checkpoint audit are complete.
Stage A passed every frozen control and effect gate. Stage B is now testing the
two claims required for a strong paper: whether proxy history changes sequential
execution, and whether the released surrogate materially changes
reward-conditioned gradients or local improvement. The first B1 engineering
preflight also exposed a serious limitation of the available public-base stress
test: 60.16% of latent steps collapsed to singleton support, so it cannot stand
in for a trained LEPO checkpoint. B2 is therefore the current decision gate.
Matched training is authorized only if its preregistered semantic and utility
criteria pass. Otherwise LPCA remains a theory/analysis/benchmark paper and the
negative optimizer branch is archived rather than rescued post hoc.

## 21. Dated portfolio update: 2026-07-15

This section supersedes the operational status in Sections 14 and 20.10. It
does not rewrite any earlier threshold, result, or idea definition.

### 21.1 LPCA B2 completed negatively

The frozen Stage B2 audit found a real exact-versus-surrogate gradient mismatch
but rejected exact-density substitution as a performance method. Across 48
prompts, three temperatures, and 144 state-temperature records, all numerical
controls passed. Exact/surrogate gradient cosine had median `0.79481` and
relative error median `0.70711`, confirming a semantic difference.

Operationally, the exact update gained `0.0063270` while the released surrogate
gained `0.0079559`. The paired difference was `-0.0016289`; prompt-clustered
95% interval `[-0.001942, -0.001349]`; every frozen temperature was negative.
Matched exact-density-substitution training is stopped permanently. LPCA remains
a defensible policy-semantics audit/benchmark direction, but not the active
optimizer method.

### 21.2 FCTR revival evidence that is valid

FCTR was revived only at the optimizer/update-equivariance level, not by
relabeling the failed nearest-neighbor result.

- C0 deterministic solver gate passed all seven controls. Coordinate-Euclidean
  steps changed under condition-4/12 charts; FCTR transported to numerical
  precision. This is a mathematical/engineering control only.
- Coconut C1c passed 11 of 11 frozen real-checkpoint differentiation and
  transport controls. It used 911.5 MiB and 33.4 seconds on the local RTX 4060.
  This is integration evidence only because the visible continuation is a
  fixed delimiter plus a short answer.
- The failed affine pilot, failed nonlinear holdout, and H1 delimiter confound
  remain binding negative evidence. They cannot be cited as FCTR success.

### 21.3 Why SWITCH is the scientific gate

SWITCH supplies a released Qwen3-8B hidden-recurrence checkpoint with long
visible math continuations. Source, base, adapter, tokenizer, dataset, and
MATH-500 prompt order are pinned. The source preflight passed.

The source also narrows the policy claim: paper-final latent noise and replay
are disabled, and latent exit is an argmax decision followed by a forced
`</swi>` whose `sampled_mask` is zero. C2 therefore studies hidden-update
geometry and continuation behavior; it does not call the forced end marker a
sampled continuous GRPO action.

### 21.4 Frozen C2 comparison

The first 16 eligible prompts in the fixed order are calibration and the next
32 are test. Eligibility requires a natural first latent block, dwell at least
four, 64 ordinary post-block tokens, no boundary/EOS in that horizon, and no
max-token truncation. All 500 prompts are scanned; geometry cannot affect
selection.

Only the first hidden state consumed after the first `<swi>` is intervened on.
Tokens 1-8 define the update objective; tokens 33-64 are the strict primary
holdout. The sole candidate is visible-prefix V32. It must beat:

- latent exit-logit L1/L3/L4;
- visible V1 FishBack, V3, and V8;
- semantic-prefix Fisher;
- calibration-only activation whitening;
- coordinate-Euclidean updates after scalar gain matching;
- a separate exact V8 KL scalar-retuning control.

V64 is an oracle and cannot be selected. Calibration chooses one probe scale,
one update gain, and one best simple metric exactly once. Test cannot retune.
V32 must improve strict-holdout risk ranking by at least `0.10` Spearman with
95% lower bound at least `0.03`, improve top-risk recall by `0.10`, reduce
matched-gain strict KL by at least 10% with ratio interval upper bound below
one, preserve held-out utility, remain within `0.05` of V64, and survive exact
V8 scalar retuning of the coordinate baselines.

### 21.5 Current portfolio status

| Direction | Current status | Binding reason / next permitted action |
|---|---|---|
| BCG-PO | Preserved, not active | H1 failure was a delimiter confound; V32 C2 can revive only the finite-horizon mechanism, not the old result. |
| SVCCO | Archived independent idea | High causal/value ambition and high estimator confounding; recover from Section 3 only. |
| CSTR-PO | Preserved as a component | Useful cheap screening or staged estimator, insufficient standalone headline without a new phenomenon. |
| FCTR / reparameterization audit | Active conditional gate | Run frozen SWITCH C2; failure stops the method, pass authorizes an efficient estimator only. |
| Simplex-GRPO | Stopped as standalone method | Exact marginalization was valid but operational effect was too small. |
| Top-K Concrete / SCLPO | Stopped as current mainline | Mathematical policy repairs did not establish a training advantage. |
| FTK-PO | Stopped | Factorized gate did not justify checkpoint training. |
| SSG-PO | Stopped | Clean support law failed to match hard-clipped mixture concentration. |
| LPCA | Preserve as analysis/benchmark | B2 rejects exact-density replacement training; continue only source-faithful mechanism audit if revisited. |

### 21.6 Compute and action decision

The SWITCH checkpoint has not been downloaded and C2 has not been run. No GPU
training is in progress. Local implementation, synthetic end-to-end replay,
source-equivalence tests, calibration/test statistics, resumable journals, and
the AutoDL handoff pass 124 project tests.

A cloud rental is now justified for C2 measurement only. Use H20 96 GB if
available; A100/H100 80 GB are alternatives. Do not rent a small card and alter
the protocol to fit. The four cloud stages are identity, all-500 eligibility,
calibration, and held-out test. Only a clean held-out pass can authorize the
next estimator-development stage; training remains unauthorized.

## 22. Idea J: Mixed-Measure Latent Policy Optimization

### 22.1 Candidate definition

MMLPO asked whether the released Latent-GRPO sampler should be optimized as a
joint policy over dynamic candidate support, ordered Top-K IDs, continuous
mixture weights, clipping events, and the request-driving proxy token. Its
intended contribution was a source-faithful replacement for the released
selected-score Gumbel surrogate.

### 22.2 Zero-GPU audit result: 2026-07-16

The source and mathematical claims are real. Hard clipping before adding
policy-dependent scores gives the executed mixture policy-dependent point-mass
locations. A two-token exact counterexample proves that a generic finite policy
update creates new action atoms outside the old law and old atoms outside the
new law. Consequently, the ordinary finite-step PPO likelihood ratio over the
executed action does not generically exist.

This does not yield a competitive optimizer. CAPG already handles fixed
clipping atoms, HPO handles hybrid mixed gradients with a differentiable
exogenous simulator, LEPO already gives a unified latent/discrete surrogate,
and boundary-corrected reparameterization plus weak-derivative policy gradients
cover the generic alternatives. For black-box terminal LLM rewards, the
remaining repairs require costly boundary constructions, counterfactual
continuations, or finite differences. Storing base noise gives a
parameter-independent score and changes the executed action under replay.

### 22.3 Decision

**KILL as a primary method-paper direction.** Preserve the moving-atom theorem,
source contract, literature audit, and exact tests under
`research/mixed_measure_policy/` as an LPCA extension or future benchmark
diagnostic. Do not run GPU training for MMLPO. A new primary candidate must move
away from likelihood repair and pass an independent novelty gate.
