# LEPO Contract Derivation

This note distinguishes a correct archive contract from a non-likelihood
optimization surrogate in the released LEPO implementation.

## 1. Source-faithful action law

Let `l` be the raw model logits and let `C(l)` be the support retained by the
released TopK-then-TopP generation processors. With `m = |C(l)| <= K`, one
latent action is

```text
z_C = softmax((l_C + g_C) / tau),   g_i iid Gumbel(0, 1),
z_j = 0 for j not in C.
```

The model executes `z E`, where `E` is the vocabulary embedding matrix. The
released configuration uses the same `K=30` for filtering and archiving, so
the archived top-k vector has unit mass and reconstructs `z` almost surely.

Conditional on a fixed support `C`, `z_C` has a Concrete density on the open
`(m-1)`-simplex:

```text
log p_C(z | l)
  = log Gamma(m) + (m-1) log tau
    + sum_i [l_i - (tau+1) log z_i]
    - m log sum_j exp(l_j - tau log z_j).
```

Its score with respect to the active logits is

```text
s_exact(z, l_C) = 1 - m softmax(l_C - tau log z_C),
E[s_exact] = 0.
```

The zero-mean identity follows from normalization of the density and is
verified numerically by the exact control in the audit suite.

## 2. Released soft-label surrogate

For a trajectory advantage `A`, the paper's Eq. 9 and the released code use

```text
J_sur(l; z, A) = A sum_i z_i log softmax(l)_i.
```

Its gradient with respect to the full raw logit vector is

```text
s_sur(z, l) = z - softmax(l).
```

This is not the Concrete score above. The distinction is structural, not a
numerical integration artifact. For every filtered-out token `j not in C`,

```text
z_j = 0  almost surely,
E[s_sur,j] = -softmax(l)_j < 0.
```

Thus the surrogate has a nonzero expected score whenever the generation
filters exclude positive model probability. Inside `C`, it also generally
differs from the exact Concrete score at finite temperature.

This observation does not by itself prove harmful bias. LEPO explicitly calls
Eq. 9 a soft surrogate, and group-centered advantages can cancel some
reward-independent components. For an ordinary reward-minus-baseline weight,
the falsifiable question is whether

```text
E[A(z) s_sur(z, l)]
```

is sufficiently aligned with the exact-density score direction

```text
E[A(z) s_exact(z, l_C)]
```

on real model states and rewards.

### 2.1 What group centering cancels

The pointwise nonzero surrogate mean must not be confused with the actual
group-aggregated update at a shared state. For a group of first latent actions
sampled from the same prompt state, let the normalized advantages satisfy
`sum_r A_r = 0` and let `p = softmax(l)` be common to the group. Then

```text
sum_r A_r s_sur(z_r, l)
  = sum_r A_r (z_r - p)
  = sum_r A_r z_r.
```

The full-vocabulary tail term cancels exactly at this shared state. The
constant part of the exact Concrete score also cancels:

```text
sum_r A_r s_exact(z_r, l_C)
  = -m sum_r A_r softmax(l_C - tau log z_r).
```

Therefore the first-state question is not whether the excluded model mass
creates a constant drift. It is whether the reward-weighted action statistic
`sum_r A_r z_r` is aligned with the reward-weighted exact statistic above.
Stage B2 tests precisely this reduced comparison.

The cancellation generally does not extend to later latent steps. Once prior
continuous actions differ, rollout `r` reaches its own state, logits `l_r`,
support `C(l_r)`, and probability vector `p_r`; group centering only gives
`sum_r A_r = 0`, not `sum_r A_r p_r = 0`. Later-step tail, support, and history
effects must consequently be audited as state-dependent sequence effects.

### 2.2 Exact density is not automatically an unbiased policy gradient

Stage B2 deliberately mirrors the pinned trainer by dividing each centered
group reward by that group's Bessel-corrected sample standard deviation. This
normalizer is itself a nonlinear function of all sampled rewards. Consequently,
even when `s_exact` is the exact Concrete score, the estimator

```text
mean_r [(R_r - mean(R)) / (std(R) + epsilon)] s_exact(z_r)
```

is not generally an unbiased estimator of the gradient of unnormalized
expected reward. Mean-only centering has a simple finite-group scale factor;
the random standard-deviation denominator destroys that factorization.

Accordingly, every B2 label of `exact` means **exact-density score under the
source-faithful group advantage**, not an exact or unbiased policy gradient.
The comparison remains controlled because the exact and released-surrogate
directions use the identical sampled actions, rewards, groups, and advantage
normalization. Their candidate updates are then evaluated on disjoint actions
with exact fixed-support importance ratios, which directly tests local utility
without assuming that either estimated direction is unbiased.

## 3. One-hot limit and filtering

Without filtering, as `tau -> 0`, the Gumbel-Softmax action approaches a
one-hot categorical sample and `z - softmax(l)` recovers the ordinary
categorical score. With TopK/TopP filtering enabled, the sampled categorical
law is instead normalized on `C(l)`, while the released target still subtracts
the unfiltered full-model softmax. The strict recovery statement therefore
requires either disabled filtering or a likelihood normalized on the same
filtered support.

## 4. Dynamic support

If `C(l_old) != C(l_new)`, old and new filtered action laws need not be mutually
absolutely continuous. A direct exact likelihood ratio is then zero or
undefined on actions whose support membership changed. LPCA reports this
separately from surrogate bias: support churn is not a defect in an objective
that never claims to use an exact ratio, but it constrains any proposed exact
repair.

## 5. Dual execution histories

The transformer consumes the continuous embedding `z E`, while the released
id stream advances with `argmax(l_C)` when latent sampling is disabled. The id
stream is visible to stopping criteria, decoding, reward text, and any
history-sensitive logits processor. The two histories agree only in degenerate
one-hot cases. Public-checkpoint experiments must measure the actual effects of
this split rather than treating mode disagreement alone as a failure.
