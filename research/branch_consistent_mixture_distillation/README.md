# Branch-Consistent Mixture Distillation (BCMD)

Status: `HOLD / GATE -1 SOURCE COLLISION AND ORACLE REQUIRED`

## One-sentence claim

BCMD trains one merged soft-token transition to approximate the predictive
distribution of its corresponding weighted hard-token branches, amortizing
local parallel exploration into one forward transition without allowing the
mixture to become trivially one-hot.

This is a new method candidate, not a continuation or renamed version of
LRC-Bench or PCMC.

## Problem

Current stochastic soft-reasoning methods usually execute

`p_soft = f(h, sum_i w_i e_i)`.

The probabilistic hard-branch teacher is

`p_branch = sum_i w_i f(h, e_i)`.

Because a transformer is nonlinear, these are not generally equal. A low
Gumbel temperature can hide the mismatch by making one weight nearly one;
raising the temperature can increase out-of-distribution drift or destabilize
training. Outcome-only RL does not specify which behavior the merged state
should preserve.

## Proposed method

For each distributed training action, BCMD adds a stop-gradient local teacher:

`L_branch = KL(stopgrad(p_branch) || p_soft)`.

The full candidate objective is the source method's RL loss plus:

1. branch-consistency distillation on a top-vocabulary support;
2. a frozen minimum effective-support constraint, so one-hot actions cannot
   satisfy the loss trivially; and
3. the source method's reference-policy KL, preserving its stability control.

The branch teacher is used only during training. Inference retains one merged
soft transition. A cheaper learned teacher or top-2 approximation is not part
of Gate -1.

## Why it may be new

- Multiplex Thinking introduces sampled branch-and-merge actions and on-policy
  optimization, but its headline objective does not establish equality between
  merged execution and the corresponding hard-branch predictive mixture.
- Existing collapse papers diagnose soft-to-hard equivalence; they do not train
  the merged transition against an explicit hard-branch teacher.
- CODI and related distillation methods align latent and explicit reasoning at
  trace, representation, or answer boundaries rather than at the stochastic
  one-step branch contract above.

These distinctions remain provisional until the pinned source audit is done.

## Frozen Gate -1

BCMD is promoted to a GPU pilot only if all conditions hold:

1. no direct paper or official implementation already optimizes the same
   weighted hard-branch-to-merged-transition KL;
2. at least two independently released stochastic mixture checkpoints have
   median effective support above `2.0` on a label-blind 64-prompt split;
3. both have median native branch-consistency JS above `0.01`, leaving a real
   target to repair;
4. a constrained embedding oracle reduces held-out median branch JS by at least
   60% and reaches at most `0.02`, without reducing effective support below
   `2.0` or exceeding a `1.25x` native perturbation radius; and
5. the oracle beats temperature-only, top-1, entropy-only, random-direction,
   and unconstrained embedding baselines.

SofT-GRPO is retained as a negative applicability control. It cannot count as a
positive family while its native effective support is approximately one.

## Permanent KILL conditions

- exact objective collision in Multiplex Thinking, CoT2, MoT-G, or another
  primary source;
- only Latent-GRPO satisfies the distributed-action precondition;
- the oracle cannot beat a temperature sweep;
- one-step repair does not predict a lower four-step teacher discrepancy; or
- gains appear only after changing the support threshold or oracle constraints.

## Expected paper type and compute

If Gate -1 passes, this is a method paper. The source/oracle gate needs inference
only. A credible method paper would later require training at least two 1B-7B
families and equal-FLOP comparisons against discrete GRPO, the source soft-RL
method, Multiplex Thinking, and temperature/support regularization.
