# P1 Baseline Contract: Cross-Checked Adapted-Metric Tangent Growth

**Drafted:** 2026-07-16  
**Status:** `PASS-DESIGN / BLOCK-EXECUTION`; independently reviewed, not execution-frozen  
**Role:** mandatory competitor, never the proposed method  
**Execution state:** `NO-GO`

This contract makes the “Certified-World-Model-style” baseline in the P1
preregistration concrete. It is motivated by [Certified World Models:
Predictability Across Configuration, Horizon, and
Resolution](https://arxiv.org/abs/2606.13092), which combines tangent/Lyapunov
growth with an adapted metric and, for non-equivariant public models, a held-out
divergence cross-check. Latent-GRPO is neither an equivariant world model nor a
uniformly hyperbolic system. Therefore the baseline below is an empirical
adaptation, not a reproduction of that paper's theorem or a “certificate.”

If this baseline matches the proposed long-horizon directional Fisher within
the frozen practical-equivalence margin at no greater wall time, the strong
Horizon-Sufficiency method direction is killed.

## 1. Fixed input and output

For each trace-stable common-proxy natural update `v` in the P1 candidate bank,
the baseline receives only:

- the frozen reference context and deterministic latent trace;
- the candidate direction `v`;
- reference histories from its own RNG pool;
- hidden-state JVPs and cheap source/trace metadata; and
- calibration-only finite-step labels for fitting the cross-check.

It may not consume held-out `s_E` labels, `q_Hcert`, `q_Hstab`, or another
baseline's predictions.

It outputs:

1. a scalar tangent-growth risk score `s_AM(v)`;
2. for each frozen evidence budget, a cross-checked upper **prediction score**
   `U_AM,h(v)` for the finite-history label statistic; and
3. `ACCEPT`, `REJECT`, or `ABSTAIN` under the same target coverage as P1.

It does not output a certified safe horizon or a confidence bound on the latent
expected KL. The evidence budget `h` says how much tangent information the
predictor consumed, not how long an update is safe to deploy.

## 2. State and tangent trace

The intervention coordinate is the action embedding itself. Let `x_0(z)=z` be
the first soft latent action input that is actually consumed after the source
stop-first execution rule. Therefore

\[
u_0(v)=D_v x_0=v,
\]

which supplies a legal denominator in the input-embedding coordinate system.
It is not replaced by the final-normalized state that generated the action,
because that earlier state is independent of the downstream intervention.

Build one ordered reference state trace from `x_0`, every later consumed
deterministic latent action embedding, and then the final-normalized residual
state `r_t` supplied to the LM head at each observable visible decision `t`.
For every later trace state `x_s`,

\[
u_s(v)=D_v x_s
\]

is computed by JVP through the complete deterministic latent suffix and visible
teacher-forced cache. The discrete latent supports, proxy sequence, stop state,
and exit trace are fixed exactly as in the primary Fisher JVP. A
`LATENT_FINISH_*`, timeout, or trace boundary forces abstention. The emission
that triggers the frozen visible token/string/length termination rule contributes
its state; later absorbing steps contribute zero.

Metric indices use frozen phase labels: `input`, later latent-depth index,
visible H1--H8 index, and visible post-H8 bin. The action-input metric is denoted
`M_in`; `M_t` below denotes the metric at the corresponding later phase. This
baseline never compares `v` with a metric fitted in an unrelated pre-action
head-state coordinate, and it never forms a full hidden-state Jacobian. Every
score is directional on the same preregistered natural update bank as P1.

## 3. Frozen metric family

Before `GO-CAL`, freeze exactly 48 calibration problem groups and split them by
frozen hash into 29 `AM-fit` and 19 `AM-conformal` groups. The count 19 is the
minimum that permits a finite one-sided 95% split-conformal quantile. All metric,
horizon, and isotonic choices use `AM-fit` only and are cross-fitted by problem
group. `AM-conformal` is opened only after those choices and fits are locked. If
eligibility or the compute cap cannot supply this split, the cross-checked
baseline and therefore the strong P1 protocol are `NO-GO`; groups are not
reused across roles. The 19 conformal groups may not tune another P1 threshold,
baseline, history count, or power choice before their residuals are recorded.
Three ordered variants are mandatory:

### 3.1 Euclidean tangent growth

\[
M_t^{E}=I.
\]

### 3.2 Robust diagonal whitening

For `input`, estimate the coordinatewise calibration median and MAD from the
unperturbed reference action embeddings `z_0(c)`. For each later frozen phase,
estimate its median `m_t` and MAD scale `a_t` from the corresponding unperturbed
reference trace states. Define

\[
M_t^{W}=\operatorname{Diag}\left(
\frac{1}{a_{t,j}^{2}+\lambda_W}
\right),
\]

where `lambda_W` is the smallest member of the frozen grid
`{1e-4,1e-3,1e-2,1e-1}` that keeps the diagonal condition number at most
`10^4`. The choice uses conditioning only, never risk prediction.

### 3.3 Adapted diagonal Lyapunov metric

Parameterize

\[
M_t^{A}=\operatorname{Diag}(\exp w_t),
\qquad
\operatorname{mean}(w_t)=0,
\qquad
\max_j w_{t,j}-\min_j w_{t,j}\le\log 10^4.
\]

Fit `w_t` on calibration tangent pairs by minimizing the frozen smooth-max
one-step expansion objective

\[
\mathcal L_M=
\frac1{G_{\rm fit}}\sum_g
\frac1\beta\log\left[
\frac1{N_g}\sum_{(f,i,t)\in\mathcal T_g}
\exp\left\{
\beta\log
\frac{\lVert u_{t+1}(v_{gfi})\rVert_{M_{t+1}^{A}}+\epsilon_M}
{\lVert u_t(v_{gfi})\rVert_{M_t^{A}}+\epsilon_M}
\right\}\right]
+\gamma_M\sum_t\lVert w_{t+1}-w_t\rVert_2^2.
\]

`T_g` is the frozen set of valid family/candidate/transition tuples in problem
group `g` and `N_g=|T_g|`. The outer mean gives every problem group equal weight;
longer traces and groups with more valid candidates cannot dominate metric
fitting. If `N_g=0`, set the fold loss to `+infinity` and declare the adapted
variant/strong P1 `NO-GO`; the group is never silently omitted.

Freeze `beta=10`, `gamma_M in {0,1e-3,1e-2,1e-1}`, `epsilon_M=1e-12`,
initialization `w=0`, optimizer, step count, dtype, and tie rule before fitting.
Choose `gamma_M` by minimum equal-problem-weight held-out squared error for
next-step log tangent expansion in leave-one-problem-group-out folds; exact ties
prefer the smaller `gamma_M`. It may not use held-out Horizon-Gap performance.
If optimization is numerically
unstable, the adapted variant abstains; it is not silently replaced by the
whitened result.

The metric is diagonal for low-compute reproducibility. No claim is made that it
is the cone metric of the cited world-model theorem.

## 4. Directional growth scores

For metric `M in {E,W,A}`, define visible-state expansion

\[
g_t^M(v)=
\frac{\lVert u_t(v)\rVert_{M_t}+\epsilon_M}
{\lVert u_0(v)\rVert_{M_{\rm in}}+\epsilon_M}
=\frac{\lVert u_t(v)\rVert_{M_t}+\epsilon_M}
{\lVert v\rVert_{M_{\rm in}}+\epsilon_M},
\]

finite-time rate

\[
\widehat\lambda_t^M(v)=\frac1t\log g_t^M(v),
\]

and transient-growth score

\[
s_{AM,H}^M(v)=
\log(\lVert v\rVert_{M_{\rm in}}+\epsilon_M)
+\max_{1\le t\le H}\log g_t^M(v).
\]

For every `h in {1,2,3,8,semantic,H_cert}`, its candidate score set is

\[
\mathcal S_h=\{s_{AM,h}^{E,W,A},
\max_{t\le h}\widehat\lambda_t^{E,W,A}\}.
\]

For `semantic`, both the maximum and tangent trace stop at the frozen legal
reference stopping time. Selection locks exactly one member of `S_h` per budget
using nested `AM-fit` problem-group cross-validation and the same finite grid
for both natural families. Its unique selection loss is

\[
\mathcal L_{CV,h}=\frac1{G_{\rm fit}}\sum_g
\frac12\sum_{f\in\{A,B\}}\frac1{|\mathcal I_{gf}|}
\sum_{i\in\mathcal I_{gf}}
\left[\log(\widehat d^{(R_E)}_{gfi}+\epsilon_d)
-\widehat f_h^{(-g)}(s_{gfih})\right]^2.
\]

An empty required problem-family cell sets the fold loss to `+infinity` and
forces `NO-GO` rather than being dropped. Minimize this loss; exact ties follow the fixed order
`s^E, s^W, s^A, max-lambda^E, max-lambda^W, max-lambda^A`. The binding raw
ranking score `s_AM(v)` is the
selected `H_cert` member; the other budgets form the declared evidence curve.
H-cert variants pay every JVP and cannot be called short-horizon merely because
they omit the vocabulary Fisher.

## 5. Finite-history divergence cross-check

The raw tangent score is not trusted as a behavioral-risk bound. Let
`dhat_gfi^(R_E)` be the exact frozen-`R_E` finite-history estimator of the P1
post-H8 endpoint for calibration problem `g`, family `f`, and candidate `i`.
For every evidence budget

\[
\mathcal H_{AM}=\{1,2,3,8,\operatorname{semantic},H_{\rm cert}\},
\]

use `AM-fit` only to select and lock one metric/score `s_h` and fit a monotone
isotonic map `fhat_h` with equal total weight per problem group:

\[
\widehat f_h:s_h(v)\mapsto
\log(\widehat d^{(R_E)}_{9:H_{eval}}(v)+\epsilon_d).
\]

Use leave-one-problem-group-out folds for all metric/map/hyperparameter
selection, then refit each chosen map once on all `AM-fit` groups and lock it.
On each untouched `AM-conformal` problem, let

\[
e_{gfih}=\log(\widehat d^{(R_E)}_{gfi}+\epsilon_d)
-\widehat f_h(s_{gfih}).
\]

The conformal unit is the problem group, not a nested candidate. Collapse each
`AM-conformal` problem to the simultaneous residual

\[
e_g^{\max}=\max_{f\in\{A,B\}}
\max_{i\in\mathcal I_{gf}}
\max_{h\in\mathcal H_{AM}}e_{gfih},
\]

where the candidate bank `I_g`, missing-candidate rule, and no-op convention are
frozen before calibration. If either required family cell is empty in a
conformal group, define that group's `e_g^max=+infinity`. With `G_conf`
conformal groups, set
`k=ceil((G_conf+1)*0.95)`. If `k>G_conf`, define `Q_.95=+infinity` and force
`ABSTAIN`; **never cap `k` at the sample maximum**. Otherwise `Q_.95` is the
`k`-th smallest `e_g^max` under the frozen finite-sample tie rule. The frozen
design uses `G_conf=19`, hence `k=19`. Do not refit the map or metric after
opening these groups. The resulting split-conformal prediction score is
simultaneous within a future problem group across the frozen candidate bank,
both families, and every inspected evidence budget:

\[
U_{AM,h}(v)=
\exp\{\widehat f_h(s_h(v))+Q_{.95}(e^{\max})\}-\epsilon_d.
\]

`epsilon_d` is fixed from numerical identity-KL error before calibration. The
fit/conformal assignment, every per-budget map, residual quantile, feature
standardization, and all fold assignments are frozen before test. The semantic
member uses the legal reference-filtration stopping time frozen in the main P1
protocol; a hindsight parser is ineligible.

This coverage target is the future, same-`R_E` **noisy finite-history label
statistic**, not the unobserved expected KL. It is conditional on problem-group
exchangeability, the frozen label RNG/count, and candidate-bank construction.
`U_AM,h` is therefore an empirical upper prediction score, not a certificate,
not a max-T truth bound, and not evidence that the true mean risk is below
`tau_d`. All false-safe truth states continue to use P1's independent
high-precision simultaneous label procedure. No selected or "safe" horizon is
derived from the collection of `U_AM,h` values.

The baseline abstains rather than extrapolates when:

- the common structural trace gate fails;
- adapted-metric conditioning or optimization fails;
- calibration conformal coverage is below `0.90` in either family;
- the test score is outside the calibration score range enlarged by the frozen
  5% margin; or
- the selected metric/horizon is unstable across calibration folds.

Abstention is charged as rejection and counts against coverage.

## 6. Predictor and decision comparison

The baseline participates in all P1 endpoints:

- rank candidates by the calibration-frozen primary `s_AM,Hcert` and separately
  by `U_AM,Hcert`;
- accept the lowest `U_AM,Hcert` scores at the exact common coverage `c_star`;
- report post-H8 Spearman, top-20% recall, simultaneous false-safe prevalence,
  selected-update full-support risk, and deployment-law utility; and
- report the complete per-budget prediction/abstention curve, with semantic kept
  separate from the numerically ordered budgets.

The strongest calibration-selected raw or cross-checked member is the binding
adapted-metric competitor. No member is dropped because it outperforms the
proposed method.

P1's long directional Fisher must exceed this baseline's test Spearman by at
least `0.03` with a one-sided problem-cluster lower confidence bound above zero.
If the baseline is within `0.03`, has equal or better recall/false-safe risk, or
matches the consequence result at no greater cost, the proposed method fails
its mandatory control gate.

## 7. Matched-compute contract

Record separately for every variant:

- latent and visible forward tokens;
- JVP/VJP/backward calls;
- hidden-state materialization;
- calibration fitting and cross-validation time;
- reference sampling and forcing;
- wall time, peak allocated/reserved VRAM, CPU RAM, and disk; and
- abstention/support/trace diagnostics.

The binding wall-time comparison is end-to-end cold-start cost for the complete
frozen calibration plus held-out study. Partition cost once into `C_common`
(candidate/source traces, splits, and finite-step labels required by both
methods), `C_P1-exclusive`, and `C_AM-exclusive`. Add the identical `C_common`
to both method totals; a tensor generated once and reused is common, never
charged only to the baseline. The no-greater-wall-time kill rule uses
`C_common+C_method-exclusive` at the same frozen held-out candidate count.

Also report an amortized per-test-candidate time whose denominator is that exact
held-out count, plus method-specific marginal inference. No larger hypothetical
deployment denominator is allowed, and marginal inference cannot replace the
binding cold-start comparison. Equal generated/forced tokens and equal
derivative calls are secondary slices. The baseline may batch or cache exactly
when P1 can use the same optimization. Metric selection, fitting, and conformal
calibration are in `C_AM-exclusive`; they are not free preprocessing.

## 8. Validation tests

Before checkpoint use, fake/tiny models must pass:

1. zero direction gives zero tangent growth;
2. scalar linear recurrence recovers its exact finite-time exponent;
3. a non-normal two-state recurrence exposes transient growth missed by spectral
   radius;
4. coordinate scaling is corrected by the corresponding adapted metric;
5. any frozen visible termination event zeros all later contributions;
6. trace/support boundary forces abstention;
7. cross-fitting never reads its held-out problem labels;
8. monotone maps and problem-max conformal quantiles reproduce exact toy
   group-simultaneous coverage; and
9. compute counters match instrumented calls.

## 9. Freeze blockers

This contract passed design review but remains execution-`BLOCK` until:

- a final independent mathematics/implementation review confirms that the
  frozen code and hashes match this design-reviewed metric/conformal contract;
- a fake-model implementation passes Section 8;
- the compute microbenchmark confirms all H-cert variants are measurable under
  the P1 cap;
- exact `AM-fit`/`AM-conformal` assignment, fold, optimizer, quantile, tie, and
  abstention settings are hashed; and
- the main P1 preregistration links the frozen hash of this contract.

No checkpoint baseline result may be opened from this draft.
