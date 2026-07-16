# P0 Collision Matrix: Horizon Sufficiency After Claim Subtraction

**Search cutoff:** 2026-07-16  
**Snapshot status:** frozen P0 collision snapshot  
**Status:** `HOLD`, not `GO`  
**Original No-Privileged-Coordinates / CETR headline:** `KILL`

This audit asks what remains after subtracting the strongest direct neighbors.
It is not a generic related-work list. The project survives only if its residual
claim is materially different from every row below.

## 1. Direct-neighbor matrix

| Work | What it already covers | Claims removed from this project | Residual space, if any | Collision severity |
|---|---|---|---|---|
| [Gauge Freedom and Metric Dependence in Neural Representation Spaces](https://arxiv.org/abs/2603.06774) | Hidden representations under invertible linear gauge changes; metric dependence of Euclidean, cosine, and neighborhood structure; function-preserving stress tests | “No privileged coordinates,” Euclidean hidden geometry is not intrinsic, and function-preserving charts alter latent geometry | It does not study natural latent updates, delayed continuation risk, or horizon adequacy | **Fatal to old title and phenomenon story** |
| [The Information Geometry of Softmax: Probing and Steering](https://arxiv.org/abs/2602.15293) | Behavior-induced softmax information geometry and minimum-distortion steering under its stated geometry/factorization | Behavior-induced geometry is preferable to Euclidean geometry; one-step softmax information geometry | No recursive latent state or multi-token continuation horizon | Very high |
| [FishBack: Pullback Fisher Geometry for Optimal Activation Steering in Transformers](https://arxiv.org/abs/2605.17231) | Pullback softmax Fisher for intermediate activations, locally constrained minimum-KL steering, low effective rank, spectral diagnostics, regularization, and natural-gradient structure | Nearly the entire `H=1` CETR method; “Euclidean bad, Fisher good”; `F^{-1}g` activation steering; ordinary low-rank Fisher | Autoregressive reference-history expectation, recursive latent horizon, and an adequacy certificate that avoids paying full long-horizon cost | **Most dangerous direct method neighbor** |
| [A Natural Policy Gradient](https://proceedings.neurips.cc/paper/2001/hash/4b86abe48d358ecf194c56c69108433e-Abstract.html) and [TRPO](https://proceedings.mlr.press/v37/schulman15.html) | Parameterization-aware Fisher steepest descent and KL trust-region policy updates | Coordinate-equivariant trust regions and the generic `F^{-1}g` optimization template | Whether cheap short-horizon internal-update risk is sufficient in recursive latent reasoning | Fatal to generic optimizer claims |
| [Path-space information bounds for stochastic dynamics](https://arxiv.org/abs/1503.05136) | Path KL, path Fisher, transient and long-time observables, Monte Carlo sensitivity bounds | “Trajectory Fisher” and local perturbation risk over a future stochastic path as new mathematics | LM-specific natural-update misranking and a practical adaptive horizon decision | Very high |
| [Certified World Models: Predictability Across Configuration, Horizon, and Resolution](https://arxiv.org/abs/2606.13092) | A computable predictability-horizon certificate for equivariant world models, Lyapunov/adapted-metric error growth, held-out divergence correction, and a budgeted re-observation decision | First certified horizon, first Lyapunov/contraction horizon certificate, or generic “know when a short horizon is enough” theory | Stochastic autoregressive sequence-KL, naturally generated latent updates, and a matched-compute risk-ranking certificate | **Very high for the certificate story** |
| [Trustworthy Koopman Operator Learning](https://arxiv.org/abs/2603.15091) | A posteriori invariance diagnostics, multi-step error bounds, and certified forecasting | Generic multi-step forecasting certificates or a posteriori horizon validation as new | No autoregressive continuation Fisher or latent-reasoning update decision | Medium-high |
| [Deterministic World Models for Verification of Closed-loop Vision-based Systems](https://arxiv.org/abs/2512.08991) | Conformal statistical bounds on trajectory deviation for verification | Generic finite-sample trajectory-deviation guarantees as new | A different deterministic vision-control setting and risk object | Medium-high |
| [Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning](https://arxiv.org/abs/2605.26733) | Recurrent latent instability, stability/effectiveness trade-off, Jacobian spectral-radius regularization, stable fixed-point training | First discovery that recurrence can amplify perturbations or that long-run stability matters | Distributional continuation risk and short-versus-long update-risk ranking can still be distinct from spectral stability | High |
| [Interpreting Latent CoT Reasoning as Dynamical Systems](https://arxiv.org/abs/2607.09698) (2026 workshop/preprint) | Latent trajectories, Lyapunov sensitivity, and stable versus unstable architecture classes | First trajectory-sensitivity or cross-architecture stability analysis of latent CoT | No Fisher trust region or horizon-sufficiency certificate; evidence status is weaker than an archival main-track paper but the claim collision remains | High |
| [Dynamics Within Latent Chain-of-Thought](https://arxiv.org/abs/2602.08783) | Step-wise causal interventions, non-local routing, and a gap between early output bias and late representational commitment | First observation that early readout can fail to capture later latent behavior | Quantitative mispricing of natural updates by short behavioral horizons and a compute-aware safety decision | **Very dangerous to the phenomenon claim** |
| [Do Latent-CoT Models Think Step-by-Step?](https://arxiv.org/abs/2602.00449) | Late fusion, partial latent rollout, and collapse under regime shifts in CODI | First late-fusion or incomplete-latent-rollout phenomenon | No short-versus-long sequence-risk certificate | Medium-high |
| [Observable Patterns Are Not Explanations: A Causal-Geometric Analysis of Latent Reasoning Models](https://arxiv.org/abs/2606.12689) | Matched controls, causal interventions, and low-rank directions whose geometry tracks behavioral influence in Coconut and CODI | First causal-geometric intervention study or first low-rank behaviorally influential latent subspace | It does not compare short versus expected long continuation risk or certify horizon adequacy | High for experimental framing |
| [Addressing divergent representations from causal interventions on neural networks](https://arxiv.org/abs/2511.04638) | Activation interventions can leave the natural representation distribution and activate dormant behavioral pathways | Treating a task-gradient or activation intervention as automatically “natural,” or attributing delayed drift to the native algorithm without support checks | In-support optimizer/checkpoint updates and explicit OOD controls can preserve the phenomenon claim | **High confound risk** |
| [Spherical Steering](https://arxiv.org/abs/2602.08169) | Geometry-aware, norm-preserving activation interventions with adaptive confidence gating | “Geometry-aware activation update” or “adaptive steering” alone | It does not use output Fisher or recursive-horizon adequacy | Medium-high |

## 2. Mathematical subtraction ledger

For

\[
p_z(y_{1:H})=\prod_{t=1}^{H}p_z(y_t\mid y_{<t}),
\]

the actual sequence Fisher is

\[
F_H(z)
=
\mathbb E_{y_{1:H}\sim p_z}
\left[
\nabla_z\log p_z(y_{1:H})
\nabla_z\log p_z(y_{1:H})^\top
\right]
=
\sum_{t=1}^{H}
\mathbb E_{y_{<t}\sim p_z}
\left[J_t^\top C(\pi_t)J_t\right].
\]

Hence

\[
F_H=F_h+
\sum_{t=h+1}^{H}\mathbb E\!\left[J_t^\top C(\pi_t)J_t\right],
\qquad
F_H-F_h\succeq0.
\]

This subtraction has four binding consequences:

1. A single teacher-forced path sum is not the complete continuation Fisher; it
   is a path-conditioned empirical generalized Gauss-Newton estimate.
2. Naming `F_H-F_h` a Horizon-Gap operator does not make the standard tail
   path-Fisher decomposition novel.
3. `F_H^{-1}g`, a pseudoinverse variant, or a damped low-rank variant is a
   sequence-distribution instance of known Fisher/natural-gradient machinery.
4. Affine equivariance, prefix non-identifiability, generalized eigenvalues,
   and generic contraction tail bounds are supporting facts, not headline
   theoretical contributions.

FishBack makes ordinary low-rank, spectral, null-space, and adaptive damping
arguments especially unsafe as novelty claims.

## 3. Claims that are permanently unavailable

The project must not claim any of the following:

- the first recognition that hidden states lack privileged coordinates;
- the first proof that Euclidean latent updates depend on parameterization;
- the first behavior-induced geometry for activations;
- the first pullback-Fisher steering method for hidden states;
- the first coordinate-equivariant latent trust region;
- the first path or trajectory Fisher for long-term behavior;
- the first evidence that latent recurrence amplifies perturbations;
- the first evidence that early readout and later latent behavior differ;
- the first certified predictability horizon or Lyapunov-derived horizon;
- the first finite-sample trajectory-deviation certificate;
- CETR as a new optimization principle; or
- “No Privileged Coordinates” as a defensible main title.

## 4. Residual claim window

The only presently defensible candidate is:

> For natural updates produced by recursive latent-reasoning algorithms, cheap
> short-horizon behavioral metrics can label some updates safe even though their
> longer observable continuations drift materially. A budget-adaptive estimator
> can decide when a short horizon is sufficient for local stochastic
> trajectory-Fisher risk or update ranking, with an explicit finite-sample error
> bound, without first computing the full long horizon and while outperforming
> direct Monte Carlo at matched compute. Finite-step sequence-KL wording
> additionally requires the frozen remainder or expected-KL UCB bridge.

This residual claim requires all four conditions below.

### 4.1 Natural update distribution

The primary population must contain updates an actual system generates:
objective gradients, exploration perturbations, projection or adapter updates,
and eventually optimizer steps. Artificial coordinate charts are correctness
tests only.

### 4.2 Horizon adequacy rather than fixed long horizon

The method must decide whether `H1`, `H3`, `H8`, or a semantic prefix is already
adequate. Always computing `F32` or `F64` and then applying a natural-gradient
step is not a new method.

### 4.3 New compute-aware estimator or certificate

Potential forms include an anytime tail-Fisher confidence sequence, sequential
reference-history allocation, a certificate that update ranking cannot change,
or another method that beats direct continuation-KL Monte Carlo under matched
compute. A fixed low-rank Fisher, Tikhonov damping, or “increase `H` until the
number looks stable” is insufficient.

### 4.4 Separation from generic recurrent instability

The effect must not collapse to a large recurrent-Jacobian spectral radius. A
strong result would demonstrate short-horizon behavioral mispricing even in a
spectrally stable regime, or show that STARS-style stability diagnostics and the
proposed certificate make materially different decisions.

## 5. Verdict

### 5.1 Original CETR / coordinate-first project

**Permanent `KILL`.** After subtracting gauge freedom, softmax information
geometry, FishBack, natural gradients/TRPO, and path Fisher, the method reduces
to an application of known machinery. Existing latent-dynamics work also
preempts the broad phenomenon framing.

### 5.2 Revised horizon-sufficiency project

**`HOLD`.** No exact full collision was found as of the search cutoff, but almost
every primitive is known. The remaining novelty is conditional on both a clean
held-out natural-update phenomenon and a genuinely new adaptive certificate.

Qualitative P0 scores after claim subtraction:

| Dimension | Score / 10 | Reason |
|---|---:|---|
| Problem definition | 5.5 | Specific and falsifiable, but adjacent to causal-dynamics and intervention-validity results |
| Mathematical object | 3.0 | Path Fisher and tail decomposition are standard |
| Fixed-horizon method | 2.0 | Closely reduces to FishBack plus sequence Fisher |
| Adaptive certificate ceiling | 5.5 | Certified horizons already exist elsewhere; defense requires stochastic sequence-KL specificity and matched-compute value |
| Current overall P0 | 4.5 | Worth one cheap phenomenon gate, not yet worth method implementation |

The next permitted scientific action is a frozen, held-out natural-update
phenomenon gate. If the method later reduces to “compute a longer Fisher and use
a natural gradient,” `P0_KILL_CRITERIA.md` requires permanent termination.

## 6. Re-audit triggers

Repeat this collision audit before method implementation, before paper writing,
and whenever a new work appears on any of these phrases:

- adaptive or anytime trajectory Fisher;
- horizon sufficiency or continuation-risk certificates;
- certified world-model or predictability horizons;
- sequential Fisher estimation for autoregressive models;
- latent-reasoning trust regions;
- recurrent activation steering; or
- long-horizon behavioral geometry.
