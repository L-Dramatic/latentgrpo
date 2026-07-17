# OMPI M0 Claim Contract

Decision date: 2026-07-18

## Proposed claim under audit

OMPI-R samples private latent paths from a frozen behavior proposal, scores every
distinct visible response under every path, averages those likelihoods over
paths, reward-tilts the resulting candidate distribution, and fits it by cross
entropy. Its claimed new mechanism is cross-path outcome credit through mixture
responsibilities.

## Frozen identity test

Let

\[
m_j(\theta)=\log\frac{1}{K}\sum_i \exp \ell_{ij}(\theta),\qquad
q_j\propto\exp(m_j(\theta_{old})+R_j/\beta).
\]

The reported OMPI-R objective is

\[
-\sum_j q_j\log\operatorname{softmax}(m(\theta))_j.
\]

This is exactly TPO on candidate logits `m`. The all-pairs responsibility
gradient

\[
\frac{\partial L}{\partial \ell_{ij}}=(p_j-q_j)\rho_{ij}
\]

is the ordinary chain rule through the JEPO/IWAE-style log-mean-exp marginal.

### KILL condition M0-A

KILL OMPI-R as a standalone optimizer if its loss and likelihood-matrix
gradient are exactly those of TPO applied to an empirical latent marginal. A
different application domain or notation is not a distinct optimization
method.

## Frozen objective-scope test

The full observable policy is

\[
\bar\pi_\theta(y|x)=\int p_\theta(y|x,z)\,\mu_\theta(dz|x).
\]

OMPI-R freezes `mu` to a behavior proposal. It therefore optimizes a
fixed-proposal response surrogate. It omits the policy-dependent proposal/score
term and must not claim to optimize the full current observable marginal.

### KILL condition M0-B

KILL any theorem or abstract claim that silently identifies the fixed-proposal
surrogate with policy improvement of the full `theta`-dependent marginal.

## Frozen identifiability test

High responsibility ESS, high normalized responsibility entropy, and high
off-source mass are not sufficient evidence of meaningful cross-path credit.
They are maximal when all paths induce exactly the same response distribution,
which is the latent-irrelevance null.

The minimum existence gate must require both:

- non-degenerate cross-path support for the same response; and
- non-zero path effect on the normalized visible-candidate distribution.

Candidate-set path/candidate mutual information is included as the first null
control. It is a diagnostic, not yet a final scientific metric.

### KILL condition M0-C

KILL Gate 1A as specified in the external report if an identical-row
likelihood matrix passes its frozen ESS, entropy, and off-source thresholds.

## Authorization

- Exact CPU identity and null-control tests: authorized.
- Checkpoint inference: not authorized before all M0 failures are resolved by a
  genuinely different claim.
- GPU training: forbidden under the current OMPI-R claim.

