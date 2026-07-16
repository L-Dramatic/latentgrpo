# P0 Mathematical Specification: Horizon Sufficiency in Recursive Latent Reasoning

**Status:** frozen normative P0 contract  
**Drafted:** 2026-07-16  
**Frozen:** 2026-07-16 after independent claim, mathematics, and collision red-teams  
**Scientific state:** `HOLD`; no method implementation or training is authorized  
**Scope:** the revised Horizon-Gap / horizon-sufficiency direction only

Once frozen, this document replaces informal uses of “continuation Fisher,” “trajectory
geometry,” and “coordinate-invariant trust region” for the revised project. It
does not modify the frozen SWITCH C2 preregistration or rehabilitate the stopped
nearest-neighbor coordinate-invariance line. The older CFG/SFTR discussion in
`../../RESEARCH_IDEA_ARCHIVE.md` Section 16 remains historical motivation; the
definitions below are the required mathematical contract for any new claim.

## 1. Observable process and intervention

Fix an input `x`, model parameters `theta`, a replayable intervention interface,
and an intervened latent state `z`. Let the future observable output law be

\[
P_z^{(H)}(y_{1:H}\mid x)
=
\prod_{t=1}^{H}\pi_z(y_t\mid x,y_{<t}).
\]

EOS is absorbing. All horizons must be consistent truncations of the same
stochastic decoding policy: the temperature, support, EOS semantics, latent
dynamics, and sampling rule cannot change with `H`.

If the model contains stochastic internal latent variables, a protocol must
choose exactly one target before collecting results:

1. the output-marginal law, with internal randomness marginalized; or
2. the joint law of internal randomness and observable outputs.

They are not interchangeable. A Fisher metric for the joint internal-output
trajectory can upper-bound or otherwise differ from the observable-output
Fisher and cannot silently be reported as behavioral output geometry.

For the output-marginal target, `J_t` below must differentiate the marginalized
conditional output law. Conditioning on sampled internal noise and averaging its
logit GGN generally estimates a joint/complete-data Fisher, not the marginal
output Fisher. P1 must therefore use deterministic internal latent dynamics or a
validated marginal-score estimator.

Define the finite-horizon behavioral drift

\[
D_H(z,z')
=
D_{\mathrm{KL}}\!\left(P_z^{(H)}\Vert P_{z'}^{(H)}\right).
\]

The KL chain rule gives

\[
D_H(z,z')
=
\sum_{t=1}^{H}
\mathbb E_{y_{<t}\sim P_z^{(t-1)}}
\left[
D_{\mathrm{KL}}\!\left(
\pi_z(\cdot\mid y_{<t})
\Vert
\pi_{z'}(\cdot\mid y_{<t})
\right)
\right].
\]

Therefore a valid Monte Carlo estimator samples histories from the reference
law `P_z` and teacher-forces the candidate `z'` on exactly those histories.
Independent candidate continuations do not estimate this directional KL.

## 2. Three objects that must remain distinct

Every artifact and paper table must label which of the following it estimates.

### 2.1 Fixed-history teacher-forced risk

Conditioning both models on a fixed ground-truth, dataset, or unperturbed greedy
model history produces a conditional teacher-forced diagnostic. The frozen
SWITCH C2 history is specifically an unperturbed greedy model history; it is not
dataset ground truth and not a stochastic reference sample. These diagnostics
can be useful, but they are not the expected continuation KL.

### 2.2 Same-history single-rollout chain-rule KL

Sampling one history from `P_z` and summing conditional KL terms along that
history is an unbiased but potentially high-variance Monte Carlo sample of
`D_H`. It is not the expectation itself.

### 2.3 Reference-rollout expected chain-rule risk

The target `D_H` and its local Fisher require an expectation over reference
histories. Claims about population risk, rankings, and certificates must account
for both prompt sampling and rollout-history uncertainty.

## 3. Correct finite-horizon trajectory Fisher

Let `ell_t(z,h_{t-1})` be the logits at step `t`, and define

\[
J_t(z,h_{t-1})
=
\frac{\partial\ell_t(z,h_{t-1})}{\partial z},
\qquad
C(\pi_t)
=
\operatorname{Diag}(\pi_t)-\pi_t\pi_t^\top.
\]

The observable `H`-step trajectory Fisher is

\[
F_H(z)
=
\sum_{t=1}^{H}
\mathbb E_{h_{t-1}\sim P_z^{(t-1)}}
\left[J_t^\top C(\pi_t)J_t\right].
\]

Equivalently, for trajectory score

\[
S_H(\tau)=\nabla_z\log P_z^{(H)}(\tau),
\]

\[
F_H(z)=
\mathbb E_{\tau\sim P_z^{(H)}}
\left[S_H(\tau)S_H(\tau)^\top\right].
\]

The cross-time score terms vanish in expectation by the conditional
martingale-score property. A sum evaluated along one sampled history is a
path-conditioned empirical generalized Gauss-Newton matrix, not the complete
trajectory Fisher.

For a sufficiently smooth model and a first-order retraction `R_z(v)`,

\[
D_H\bigl(z,R_z(v)\bigr)
=
\tfrac12v^\top F_H(z)v+O(\lVert v\rVert^3).
\]

Define three separate risk objects:

\[
q_K(v)=\tfrac12v^\top F_Kv,
\qquad
a_K(v)=\tfrac12v^\top(F_K+\lambda M)v,
\]

\[
d_K(v)=D_K\bigl(z,R_z(v)\bigr).
\]

The Fisher expansion controls the local quadratic proxy `q_K`, not automatically
the finite-step behavioral risk `d_K`. A finite-step behavioral certificate must
add a uniform remainder bound, for example

\[
d_K(v)
\le
q_K(v)+\frac{L_K}{6}\lVert v\rVert_M^3,
\]

over the frozen update set, or use an independently calibrated upper-confidence
line search for expected chain-rule KL. Without one of these bridges, all theorem
wording must say **local trajectory-Fisher risk**, and finite-step KL is only an
empirical validation target.

An arbitrary weighted sum of time-step terms is not generally a trajectory
Fisher. Ordinary scalar survival weights are valid only for a stopping process
independent of both model parameters and realized history. A history-dependent
stopping law requires history-conditional weights; a parameter-dependent law
also contributes its own score. All other hand-chosen weights define a weighted
behavioral-risk metric, not the original trajectory Fisher.

## 4. Horizon Gap and the actual object under test

For consistent truncations, let

\[
\Delta_t=F_t-F_{t-1}\succeq0.
\]

Then, for `h < H`,

\[
F_H-F_h
=
\sum_{t=h+1}^{H}\Delta_t
\succeq0.
\]

This monotonicity is a standard path-Fisher fact, not a novel theorem. The
scientific question is whether the tail is operationally material for natural
updates that an algorithm would actually make.

Fix a physically defined update subspace `S = Range(B)`, where `B` has full
column rank and is frozen before test outcomes. Let `M` be a transported
symmetric reference tensor whose restriction

\[
\bar M=B^\top MB\succ0
\]

is positive definite, and freeze `lambda > 0` independently of the certificate.
Define the relative Horizon Gap

\[
\gamma_{h,H}(\mathcal S)
=
\sup_{v\in\mathcal S,\,v\ne0}
\frac{v^\top(F_H-F_h)v}
{v^\top(F_h+\lambda M)v}.
\]

Equivalently, with

\[
\bar F_K=B^\top F_KB,
\]

this is the largest generalized eigenvalue of
`(\bar F_H-\bar F_h, \bar F_h+\lambda\bar M)`. The worst direction is a stress
test, not the population claim. The project must also measure, over a
preregistered natural update distribution `Q(v)`:

- short-risk versus long-risk rank disagreement;
- false-safe rate at a calibration-only threshold;
- tail-risk fraction;
- top-risk recall;
- coverage of objective-gradient, exploration, projection, adapter, or actual
  optimizer updates;
- realized finite-step KL and downstream utility.

Artificial chart directions may be used only as metamorphic correctness tests.
They cannot establish the Horizon-Gap phenomenon.

## 5. Positive semidefiniteness, null spaces, and well-posed updates

The naive problem

\[
\max_v g^\top v
\quad\text{subject to}\quad
\tfrac12v^\top F_Hv\le\epsilon
\]

need not be well posed. If `g` has a component in `ker(F_H)`, the problem is
unbounded. If `g` lies in `Range(F_H)`, a pseudoinverse direction exists but is
non-unique in the null space.

The contract therefore requires a declared, transportable reference metric that
is positive definite on the update subspace:

\[
A_H=F_H+\lambda M.
\]

Let

\[
\bar g=B^\top g,
\qquad
\bar A_H=B^\top A_HB.
\]

For `\bar g\ne0`, the regularized reduced-space trust-region direction is

\[
\bar v_H^\star
=
\sqrt{\frac{2\epsilon}{\bar g^\top\bar A_H^{-1}\bar g}}
\bar A_H^{-1}\bar g,
\qquad
v_H^\star=B\bar v_H^\star.
\]

The ambient formula is recovered when `B = I` and `M \succ 0` globally.

An alternative is a dual constraint,

\[
\tfrac12v^\top F_Hv\le\epsilon,
\qquad
\tfrac12v^\top Mv\le r^2.
\]

Allowed sources of the restricted metric `\bar M` are:

- a positive-definite metric declared directly in physical control coordinates;
- a precision/Mahalanobis metric derived from representation covariance on the
  frozen subspace; a singular covariance requires a preregistered regularized
  inverse or pseudoinverse relative to a transported base tensor; or
- a metric fixed in one reference chart and transported by the tensor law.

Setting `M = I` independently in every chart, selecting `M` from test results,
or hiding singular behavior with an unreported damping sweep is forbidden.
The value of `lambda` cannot be selected to make the Horizon Gap small. Every
result must report raw Fisher tail, regularized tail, an `M`-only/large-damping
baseline, and the direction-wise regularizer share

\[
r_{\mathrm{reg}}(v)=
\frac{\lambda v^\top Mv}{v^\top(F_h+\lambda M)v}.
\]

## 6. Coordinate transport is a correctness control, not the paper claim

For a chart change `z' = phi(z)` with Jacobian `A = D phi_z`, the corresponding
objects transform as

\[
v'=Av,
\qquad
g'=A^{-\top}g,
\]

\[
F'_H=A^{-\top}F_HA^{-1},
\qquad
M'=A^{-\top}MA^{-1}.
\]

If `B` is a basis for a reduced physical subspace, the same subspace in the new
chart is represented by `B' = AB`. Consequently,

\[
B'^\top g'=B^\top g,
\]

\[
B'^\top(F'_H+\lambda M')B'
=
B^\top(F_H+\lambda M)B.
\]

The reduced tangent update is therefore equivariant when all objects are
transported. Forbidden shortcuts include:

- drawing coordinate-isotropic Gaussian directions separately in each chart;
- performing independent Euclidean QR and calling the bases identical;
- treating the covector `g` as a tangent vector without raising its index;
- recomputing top directions separately after a chart change; and
- adding `lambda I` independently in each chart.

If a gradient-derived tangent direction is required, use the declared metric,

\[
\bar v_g=\bar M^{-1}B^\top g,
\qquad
v_g=B\bar v_g.
\]

For nonlinear `phi`, coordinate addition is only first-order equivariant because
`phi(z+v)` generally differs from `phi(z)+Av` by `O(||v||^2)`. A finite-step
invariance claim requires an induced retraction

\[
R'_{\phi(z)}(Av)=\phi\bigl(R_z(v)\bigr)
\]

and a line search using an upper confidence bound for the reference-rollout
expected chain-rule KL along the same physical path. A single realized path
cannot verify a trust-region guarantee.

These facts are standard differential geometry. They are retained to prevent an
invalid experiment, not as novelty.

## 7. Activation-space and parameter-space trust regions

An activation-space matrix `F_{z,H}(x)` controls a local intervention at one
specified interface for one input. It does not by itself control a training
update.

For trainable parameters `vartheta`, the relevant object is

\[
F_{\vartheta,H}
=
\mathbb E_{x\sim\mathcal D}
\mathbb E_{\tau\sim P_\vartheta^{(H)}}
\left[
\nabla_\vartheta\log P_\vartheta^{(H)}(\tau)
\nabla_\vartheta\log P_\vartheta^{(H)}(\tau)^\top
\right].
\]

Only when parameters affect the continuation through one exclusive interface
`z_x(vartheta)` may one write

\[
F_{\vartheta,H}
=
\mathbb E_x\left[J_x^\top F_{z,H}(x)J_x\right],
\qquad
J_x=\frac{\partial z_x}{\partial\vartheta}.
\]

Shared parameters that affect multiple recurrent states, the decoder, or other
paths require differentiation of the complete trajectory score with respect to
the parameters. Per-state activation Fishers cannot simply be added.

Accordingly, an activation experiment may establish a failure mode, but a method
paper about training additionally requires parameter-space risk control and
realized optimizer evidence.

## 8. Candidate horizon-sufficiency certificate

The method contribution cannot be “use a longer Fisher.” It must decide, without
first paying the full long-horizon cost, when a shorter horizon is adequate for
the preregistered update subspace.

### 8.1 Projected target and declared horizons

Declare three roles before data collection:

- `h` is a candidate cheap horizon from a finite set `H_short`;
- `H_cert` is the finite oracle horizon covered by the certificate; and
- `H_eval` is a separately measured behavioral endpoint.

If `H_eval > H_cert`, the certificate does not automatically cover `H_eval`.
That endpoint is empirical validation unless an additional tail guarantee spans
the difference.

For the frozen basis `B`, define

\[
\bar A_h=B^\top(F_h+\lambda M)B,
\qquad
\bar T_{h,H_{\mathrm{cert}}}
=B^\top(F_{H_{\mathrm{cert}}}-F_h)B.
\]

The exact relative tail is

\[
u_h^\star
=
\inf\left\{
u\ge0:
\bar T_{h,H_{\mathrm{cert}}}\preceq u\bar A_h
\right\}.
\]

This is `gamma_{h,H_cert}(S)` from Section 4. A matrix square-root expression is
only a computational representation; the Loewner inequality is the
coordinate-safe definition.

### 8.2 Anytime information structure

Let `F_{h,n}` be the filtration generated by all permitted prefixes through
horizon `h` from the first `n` independently seeded reference-rollout replicates,
plus frozen calibration information. A data-dependent horizon, rollout count,
or allocation rule must be a stopping time with respect to this filtration and
cannot inspect unobserved suffixes.

An admissible method returns an `F_{h,n}`-measurable `U_{h,n}`. For a bounded
rollout budget `N`, the required event is

\[
\Pr\!\left(
\forall h\in\mathcal H_{\mathrm{short}},
\forall n\le N:
\bar T_{h,H_{\mathrm{cert}}}
\preceq
U_{h,n}\bar A_h
\right)
\ge1-\delta.
\]

If `N` is unbounded, a genuine confidence sequence is required. A fixed-sample
union bound over horizons is not optional-stopping valid for adaptive rollout
counts.

### 8.3 Estimation uncertainty and an implementable decision

Plug-in matrices do not provide coverage. On one event simultaneous over all
inspected `(h,n)`, construct projected Loewner bounds

\[
0\prec\lambda\bar M
\preceq
\bar A^-_{h,n}
\preceq
\bar A_h
\preceq
\bar A^+_{h,n},
\qquad
\bar T_{h,H_{\mathrm{cert}}}
\preceq
\bar T^+_{h,n}.
\]

The known regularizer supplies a positive-definite floor on the frozen subspace.
If an implementation cannot preserve this ordering numerically, the certificate
abstains.

Then compute

\[
U_{h,n}
=
\lambda_{\max}\!\left(
(\bar A^-_{h,n})^{-1/2}
\bar T^+_{h,n}
(\bar A^-_{h,n})^{-1/2}
\right),
\]

or equivalently enforce

\[
\bar T^+_{h,n}\preceq U_{h,n}\bar A^-_{h,n}.
\]

For update coordinates `a` with `v = Ba`, the sufficient condition

\[
(1+U_{h,n})
\,\tfrac12a^\top\bar A^+_{h,n}a
\le\epsilon
\]

certifies the regularized local quadratic budget at `H_cert` on the simultaneous
event. A ranking certificate must show that update-risk intervals cannot overlap
or reverse under the preregistered practical margin.

The damping `lambda` is frozen by an independent numerical or optimization rule.
It cannot be tuned to shrink `U_{h,n}`. Raw unregularized tail, regularized tail,
the regularizer share, and an `M`-only/large-damping baseline are mandatory. If
the certificate succeeds only because regularization dominates, it has not
established horizon sufficiency.

### 8.4 Local proxy versus finite-step behavior

The decision above certifies `a_{H_cert}(v)`, a local regularized quadratic
proxy. To certify the actual finite-step `d_{H_cert}(v)`, the method additionally
needs either:

1. a uniform and checkable Taylor-remainder envelope such as

   \[
   d_{H_{\mathrm{cert}}}(v)
   \le
   q_{H_{\mathrm{cert}}}(v)
   +\frac{L_{H_{\mathrm{cert}}}}{6}\lVert v\rVert_M^3;
   \]

2. or an independently calibrated, reference-rollout expected-KL upper
   confidence line search with simultaneous coverage for inspected step sizes.

Without this bridge, use “local trajectory-Fisher horizon certificate,” not
“behavioral-risk certificate.” Finite-step expected chain-rule KL and task
utility remain independent empirical validation.

An expected-KL UCB bridge may still require sampling suffixes through `H_cert`.
All such forward calls are charged to the method. Only a structural/remainder
envelope can claim a finite-step guarantee without that suffix cost. If the UCB
bridge routinely costs as much as direct long-horizon Monte Carlo, the method has
not earned a compute advantage.

### 8.5 What can upper-bound the unseen tail

A certificate for an unobserved future requires a real envelope, not extrapolated
hope. One possible supporting condition is

\[
\bar\Delta_{t+1}
\preceq
\rho^2\bar\Delta_t,
\qquad
\bar\Delta_t=B^\top\Delta_tB,
\quad \rho<1,
\]

which yields a geometric tail bound. But `rho` and the matrix inequality must be
justified by either:

- a structural Jacobian/operator-norm envelope valid for all reachable
  histories in scope; or
- an independently calibrated upper envelope plus an explicit transfer or
  exchangeability assumption for the test population.

Fitting `rho` from the observed test prefix is invalid. A recurrent Jacobian's
average spectral radius below one is also insufficient because non-normal
transient amplification and token-dependent branching can violate the matrix
decay condition. The geometric-series calculation is background, not novelty.

### 8.6 Required theorem target

A potentially publishable theorem must provide simultaneous Loewner bounds and
an implementable `U_{h,n}` over both adaptive horizons and adaptive rollout
counts, while accounting for shared prefixes, optional stopping, and estimation
of `A_h`. It must state whether it is:

- a **conditional certificate** for a fixed prompt `x`, with probability over
  stochastic reference histories; or
- a **population certificate** for average risk over `x ~ D`.

An average population guarantee is not a per-prompt false-safe guarantee.
Hierarchical prompt inference belongs to the population experiment and cannot be
used to imply conditional coverage.

Before P2 implementation, one concentration package must be chosen, justified,
and frozen with explicit constants. Examples include conditionally bounded
projected PSD increments with a known operator-norm cap, or conditionally matrix
sub-exponential centered increments with declared variance and scale processes.
The package must also state independence or exchangeability across
reference-rollout replicates and the assumption that transfers any calibration
tail envelope to the test population. Merely postulating the simultaneous event
in Section 8.3 is not a theorem.

The method must also beat direct reference-rollout continuation-KL estimation
under matched forward calls, backward calls, wall time, and memory. An
infinite-horizon statement requires additional tail assumptions and is not
needed for the finite declared `H_cert`.

## 9. Statistical contract

1. Calibration prompts and test prompts are disjoint. Shared templates,
   checkpoints, optimizer trajectories, or source problems require a group split
   or hierarchical analysis, not a nominal row-level split.
2. Calibration rollout seeds and test rollout seeds are disjoint. Within test,
   histories used to construct `F_h/F_H_cert` are seed-disjoint from histories
   used to estimate `D_H_eval`, false-safe labels, and final rankings. Otherwise
   the result can reduce to shared Monte Carlo noise and local Fisher identity.
3. `h`, `H_cert`, `H_eval`, damping, basis `B`, update families, thresholds,
   remainder rule, and stopping budget are frozen on calibration data.
4. The calibration set must test whether `H_cert`, `2 H_cert`, or semantic
   completion yields stable risk rankings. An unconverged “long-horizon oracle”
   cannot define adequacy.
5. Prefixes `H1`, `H2`, `H3`, `H8`, and semantic boundaries are explicit
   baselines; a formatting token cannot define the short horizon.
6. Multiple horizons on one trajectory are dependent. Inference clusters or
   hierarchically bootstraps every shared source, including prompt, template,
   checkpoint, and optimizer trajectory, and controls simultaneous testing over
   horizons, models, and update families.
7. Natural-update claims require support diagnostics. At minimum, report a
   calibration-frozen representation-density or Mahalanobis measure plus decoder
   entropy or another behavior-level OOD diagnostic, so dormant-path activation
   is not mistaken for native delayed risk.
8. Ranking metrics, false-safe prevalence, top-risk recall, expected chain-rule
   KL, free-generation changes, utility, wall time, rollout count, peak memory,
   and uncertainty are all reported.
9. The matched-compute baseline includes direct Monte Carlo continuation KL, not
   only local Fisher approximations.
10. Stability diagnostics include more than average spectral radius when
    non-normal transient amplification is possible.
11. With stochastic internal latent dynamics, the marginal-output score or the
    explicitly joint target must be validated before any Fisher result is used.

## 10. Correctness lemmas versus potential contributions

The following may appear only as correctness lemmas or background:

- the trajectory-KL chain rule;
- the score/Fisher decomposition;
- `F_H - F_h \succeq 0`;
- affine or tangent-level reparameterization equivariance;
- a generalized eigenvalue formulation;
- a generic contraction/geometric-series tail bound;
- a fixed low-rank natural-gradient solver; and
- a pseudoinverse or Tikhonov treatment of Fisher null spaces.

Potentially defensible contributions require all three layers:

1. a held-out natural-update failure showing that short horizons materially
   misprice longer observable behavior in recursive latent reasoning;
2. a new compute-aware sequential estimator or horizon-sufficiency certificate;
3. an operational consequence for safe update selection or parameter training,
   replicated in a second mechanism-distinct architecture.

## 11. Explicit non-claims

The project does not claim that:

- Fisher information, natural gradients, TRPO, pullback geometry, path Fisher,
  or behavior-induced representation geometry are new;
- hidden states have no meaningful native parameterization;
- Euclidean coordinate dependence is a new discovery;
- an artificial chart change is a natural failure case;
- a single teacher-forced trajectory equals a continuation distribution;
- `H = 32` is intrinsically correct, or longer is always better;
- a local quadratic certificate is automatically a finite-step behavioral
  guarantee;
- a certified prediction horizon or Lyapunov-derived horizon is new in general;
- coordinate addition is finitely invariant under nonlinear charts;
- latent states are faithful or human-interpretable thoughts;
- a joint/internal-noise Fisher equals the observable marginal Fisher;
- increasing `lambda M` demonstrates that a short behavioral horizon is
  sufficient;
- activation-space success implies parameter-training success; or
- an infinite-horizon guarantee holds without an explicit, independently
  justified tail condition.

All continuation of this direction is governed by `P0_KILL_CRITERIA.md`.
