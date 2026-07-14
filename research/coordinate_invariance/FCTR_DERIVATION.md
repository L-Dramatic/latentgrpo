# Functional Continuation Trust Region: Equivariance Contract

This note fixes the mathematical contract implemented by `fctr.py`. It does
not claim that natural gradients, Fisher geometry, or output-space trust
regions are new.

## Local problem

At a latent state `z`, let `g_z` be the gradient of a scalar objective and let
`G_z` be a positive-definite local metric induced by an observable
continuation distribution. The local FCTR problem is

```text
maximize_delta    g_z^T delta
subject to        0.5 delta^T G_z delta <= epsilon.
```

Its solution is

```text
delta_z = sqrt(2 epsilon / (g_z^T G_z^-1 g_z)) G_z^-1 g_z.
```

The implementation permits a regularizer only as another metric tensor that
is transported with the chart. A numerical term `lambda I` is coordinate
dependent under a non-orthogonal chart and is therefore not an invariant
repair.

## Exact linear-chart equivariance

For an invertible affine chart `u = A z + b`, covectors, metrics, and tangent
vectors transform as

```text
g_u = A^-T g_z,
G_u = A^-T G_z A^-1,
delta_u = A delta_z.
```

Substitution gives

```text
G_u^-1 g_u = A G_z^-1 g_z,
g_u^T G_u^-1 g_u = g_z^T G_z^-1 g_z.
```

Therefore solving the same functional trust-region problem in `u` and
transporting its step back with `A^-1` recovers the native step exactly up to
numerical precision. A coordinate-Euclidean step instead follows
`A^-1 A^-T g_z` after transport and is invariant only for orthogonal charts
up to an overall scalar convention.

## Smooth nonlinear charts

For `u = phi(z)`, replace `A` locally by the Jacobian `J_phi(z)`. The same
identities hold in the tangent space. A finite additive step in chart
coordinates differs from the exponential or retraction of the transported
tangent by second-order chart-curvature terms. Consequently, nonlinear tests
must report the step radius and verify first-order convergence rather than
claim finite-step identity.

## What metric is allowed

Any metric derived from an observable output distribution transforms
covariantly when implemented correctly. This includes:

- next-token pullback Fisher;
- H2/H3 prefix Fisher;
- a finite-horizon path Fisher;
- an exact local Hessian of continuation KL under regularity conditions.

This creates a strict novelty boundary: reparameterization invariance alone
does not justify the multi-horizon FCTR method. FishBack-style next-token
geometry is already an invariant baseline. A real-model FCTR gate must show
that a longer-horizon metric changes a consequential update decision that H1,
H2/H3, ordinary sequence KL, whitening, and retuned coordinate thresholds do
not already explain.

## Evidence ladder

1. The deterministic toy gate validates tensor transport and solver controls.
2. A public-checkpoint gate must demonstrate a moderate-chart operational
   failure, with no-op, orthogonal, precision, and scalar-retuning controls.
3. The multi-horizon metric must add value beyond short-prefix functional
   baselines under matched compute.
4. Only then may FCTR enter training, followed by a second latent architecture.

Failing item 2 stops the method branch. Failing item 3 retains at most a
critical analysis and forbids presenting FCTR as the necessary correction.
