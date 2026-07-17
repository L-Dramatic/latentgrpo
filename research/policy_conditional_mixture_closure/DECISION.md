# PCMC Research Decision

Decision date: 2026-07-18

## Verdict: HOLD, mechanism gate only

| Gate | Result | Reason |
|---|---|---|
| Exact identity with one prior method | NOT PROVEN | The domain-specific target/adapter combination is not algebraically identical to SCM. |
| Independent loss novelty | FAIL | The frozen one-step objective is ordinary teacher-distribution KL distillation. |
| Target semantics | UNRESOLVED | Hard-branch mixture is a design contract, not the unique semantics of a soft embedding. |
| One-step to sequence implication | FAIL | Exact counterexample gives zero initial KL but `0.5` sequence TV. |
| Causal non-closure evidence | NOT RUN | No natural-checkpoint evidence yet. |
| Constrained oracle barycenter | NOT RUN | No evidence that a one-forward solution exists in the natural latent region. |

## Authorization

- Preserve PCMC as the only current mechanism hypothesis from the external
  report.
- Authorize protocol and source-adapter preparation on CPU.
- Do not authorize checkpoint inference until that protocol has fixed sample
  sizes, controls, manifests, and shutdown-safe execution.
- Do not authorize training.

## Promotion rule

Only causal Gate A and constrained-oracle Gate B can promote PCMC from HOLD.
Passing one-step KL alone is insufficient. Failure of either gate KILLs PCMC as
the next AAAI method direction without threshold relaxation.

