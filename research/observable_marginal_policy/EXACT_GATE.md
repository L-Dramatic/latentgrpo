# OMPI-R Exact Identity Gate

Decision date: 2026-07-18

## Frozen results

| Check | Result |
|---|---:|
| OMPI-R loss minus TPO-on-marginal loss | `0.0` |
| Maximum likelihood-gradient difference | `0.0` |
| Irrelevance-null median responsibility ESS, `K=4` | `4.0` |
| Irrelevance-null median normalized entropy | `1.0` |
| Irrelevance-null median off-source mass | `0.75` |
| Irrelevance-null path/candidate mutual information | `0.0` nats |
| Irrelevance-null candidate-distribution change | `0.0` L1 |

All six focused tests pass.

## Finding 1: the proposed optimizer is an exact composition

Substituting OMPI-R's empirical response-marginal logit

\[
m_j=\log\frac{1}{K}\sum_i\exp\ell_{ij}
\]

into TPO gives the reported target, loss, candidate-logit gradient, and
responsibility-weighted likelihood gradient exactly. The responsibility term is
the chain rule through the log-mean-exp; it is not an additional estimator.

The strongest baseline is therefore not `diagonal TPO`. It is **TPO using the
same empirical marginal candidate logits**, and that baseline is OMPI-R itself.
An aliasing toy can show that marginalization beats a diagonal approximation,
but it cannot establish independent method novelty against this baseline.

## Finding 2: Gate 1A confounds aliasing with irrelevance

When every latent path induces the same visible-response distribution, all
paths receive uniform responsibility for every response. This is maximal ESS,
entropy, and off-source mass even though the path has no behavioral effect and
the marginal candidate policy is unchanged.

The external report's frozen Gate 1A would pass this null:

- ESS `4.0 >= 1.50`;
- normalized entropy `1.0 >= 0.25`;
- off-source mass `0.75 >= 0.25`.

A valid existence test needs two axes: within-response multi-path support and a
non-zero path intervention effect on visible behavior. Path/candidate mutual
information is included only as an initial null control; a future checkpoint
test would also need length, duplicate, and candidate-truncation controls.

## Finding 3: OMPI-R is a fixed-proposal surrogate

The full response marginal depends on both the visible decoder and the latent
path distribution. Freezing the behavior proposal removes the proposal/score
gradient. Updating shared decoder parameters does not restore that missing
term. OMPI-R may be described as a replay or generalized-EM-like surrogate, but
not as an exact policy-improvement update for the full current response policy.

## Gate decision

The external report pre-registered a stop before coding if the proposal reduced
to standard TPO plus latent marginalization. That condition is met exactly.
Running the report's larger CPU family or checkpoint gate cannot distinguish
OMPI-R from its strongest composed baseline, so those runs are not authorized.

