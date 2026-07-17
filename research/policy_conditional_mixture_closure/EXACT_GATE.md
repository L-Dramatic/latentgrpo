# PCMC Sequence-Closure Exact Gate

Decision date: 2026-07-18

## Counterexample

A private branch is selected once with equal probability. Branch A emits `00`
and branch B emits `11`. The teacher sequence mixture is therefore

\[
P_T(00)=P_T(11)=0.5,
\]

with zero probability for `01` and `10`.

A student can match the initial next-token distribution exactly with
`P(0)=P(1)=0.5` but reuse the static mixture independently at the second step.
Its sequence law is uniform over all four sequences.

## Results

| Metric | Result |
|---|---:|
| Initial one-step KL | `0.0` nats |
| Sequence total variation | `0.5` |
| Student mass on impossible cross-branch sequences | `0.5` |
| Sequence TV with posterior-updated branch conditionals | `0.0` |

## Interpretation

The report's one-step closure objective can be a useful local distillation
signal, but it does not support the sequence-level claim by itself. The stated
telescoping bound remains mathematically valid only if its per-history errors
are actually controlled against the correct posterior-updated teacher. That is
a stronger and more expensive condition than the proposed first loss.

This counterexample does not KILL the observed non-closure phenomenon. It KILLs
the inference that one-step matching alone repairs multi-step latent reasoning.

