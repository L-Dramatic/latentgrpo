# PCMC M0 Claim Contract

Decision date: 2026-07-18

## Proposed claim under audit

PCMC trains a contextual latent barycenter to approximate, with one forward
pass, a weighted mixture of the continuation policies produced by hard token
components. The target is meant to repair the behavioral non-closure of an
arithmetic embedding mixture.

## M0-A: target semantics

The arithmetic mixture and randomized hard-branch mixture are two different
policies. The latter is a justified target only if the support weights are
interpreted as uncertainty over a mutually exclusive hard branch. Soft
Thinking and related methods instead interpret the mixed embedding as a new
continuous concept. PCMC must present this as a tested design contract, not as
a self-evident ground truth.

KILL if the paper calls non-equality to the hard-branch mixture a bug without
showing that replacing it causally improves held-out reward at matched compute.

## M0-B: method identity

For a fixed teacher mixture, the proposed one-step loss

\[
D_{KL}\left(\sum_i q_i P_i\;\Vert\;P_{b_\phi}\right)
\]

is standard distribution or policy distillation. A set-conditioned latent
adapter changes the student parameterization but not the distillation
objective.

KILL as a standalone method if the independent remainder is only `SCM + KL
consistency loss` or an ordinary same-model ensemble distillation adapter.

## M0-C: sequence contract

Matching the initial one-step teacher mixture does not imply matching the
mixture of autoregressive sequence laws. A branch selected once induces
history-dependent posterior weights after tokens are observed. Static weights
can create cross-branch sequences that every teacher branch assigns zero
probability.

Any multi-step theorem must define the target conditional at every reachable
history using posterior-updated branch weights. A telescoping bound that assumes
small error at every history cannot be used to justify a loss measured only at
the initial state.

KILL the one-step-only method claim if held-out multi-step divergence or reward
does not improve beyond same-parameter distillation baselines.

## M0-D: evidence sequence

PCMC may proceed only in this order:

1. causal non-closure: hard-branch intervention beats arithmetic execution on
   natural mixtures, with entropy, length, and prompt difficulty controlled;
2. oracle existence: a constrained latent vector closes most of the one-step
   gap without leaving the natural hidden-state radius;
3. sequence validity: posterior-aware or rollout-level distillation reduces
   held-out multi-step divergence and impossible cross-branch behavior;
4. amortization: a same-parameter adapter captures a material fraction of the
   oracle gain and improves final reward at matched training and inference
   compute.

Checkpoint inference is not authorized until the concrete Gate A protocol,
sample-size analysis, source adapters, and fail-closed output manifest exist.
Training is not authorized at M0.

