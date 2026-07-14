# SSG-PO Probability Contract

## Setup

For a fixed eligible vocabulary `C`, canonicalize finite model logits as

```text
l_i = raw_l_i - log sum_(j in C) exp(raw_l_j),  i in C.
```

Draw independent standard Gumbels and form complete scores

```text
X_i = l_i + sigma G_i.
```

Let `h: R -> (a,b)` be the same strictly increasing differentiable bijection
for every item and policy. The implemented transform is

```text
h(x) = m + s tanh(x/s),  m=(a+b)/2, s=(b-a)/2.
```

Store the ordered Top-K indices `S=(s_1,...,s_K)` and selected transformed
scores `Y_r=h(X_sr)`. The executed latent mixture is a deterministic function
`softmax(Y/tau)`.

## Proposition 1: exact support preservation

For every pair of vocabulary items, strict monotonicity gives

```text
X_i > X_j  iff  h(X_i) > h(X_j).
```

Gumbel scores tie with probability zero, so the complete ordered Top-K support
under `Y` equals that under ordinary unbounded Gumbel Top-K almost surely. The
squash controls mixture score range without changing support exploration.

## Proposition 2: exact selected-score density

Write `x_r=h^{-1}(y_r)`. For a valid descending action, independence and the
Top-K order event give

```text
p_l(S,y)
 = product_r [f_G((x_r-l_sr)/sigma) / sigma * |d x_r / d y_r|]
   product_(j in C minus S) F_G((x_K-l_j)/sigma).
```

The second product integrates every unselected score below the K-th selected
threshold. For the implemented transform,

```text
x = s atanh((y-m)/s),
|dx/dy| = 1 / (1-((y-m)/s)^2).
```

This is a normalized density with respect to counting measure over ordered
supports and Lebesgue measure over all `K` stored selected scores. The action is
augmented because downstream computation depends only on their softmax.

## Proposition 3: common support across policies

For finite logits, the Gumbel density and CDF are strictly positive at every
finite argument. Therefore every policy has positive density on the same action
space

```text
{(S,y): S distinct and eligible, b>y_1>...>y_K>a}.
```

Any two such SSG policies are mutually absolutely continuous. Their
importance ratio is defined almost everywhere on old-policy samples.

## Proposition 4: hard clipping breaks this contract

If additive noise is clipped before it is added to a policy location,

```text
X_i = l_i + sigma clip(G_i, c_low, c_high),
```

then `X_i` has positive atoms at `l_i+sigma c_low` and
`l_i+sigma c_high`. A policy update that changes `l_i` moves those atoms. An
old atom location is generally neither a new atom nor a positive-Lebesgue-mass
set under the new continuous component. It consequently has positive old mass
and zero new mass.

Thus ordinary old-to-current importance sampling over selected scores is not
defined on all positive-mass rollouts. The statement concerns this explicit
selected-score action representation. It does not claim that every possible
coarsening or alternative augmented representation is singular.

## Scope limitations

- The candidate mask must be fixed independently of the optimized policy during
  a likelihood-ratio update. Dynamic top-p adds a separate support problem.
- Tanh saturation can harm gradients near the bounds and is an empirical gate,
  not dismissed by the probability proof.
- A valid likelihood does not imply better reasoning. Real-model rollout and
  matched-compute training gates remain necessary.
