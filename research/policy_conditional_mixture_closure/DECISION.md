# PCMC Research Decision

Decision date: 2026-07-18

## Verdict: KILL PCMC as the next general AAAI method direction

| Gate | Result | Reason |
|---|---|---|
| Exact identity with one prior method | NOT PROVEN | The domain-specific target/adapter combination is not algebraically identical to SCM. |
| Independent loss novelty | FAIL | The frozen one-step objective is ordinary teacher-distribution KL distillation. |
| Target semantics | UNRESOLVED | Hard-branch mixture is a design contract, not the unique semantics of a soft embedding. |
| One-step to sequence implication | FAIL | Exact counterexample gives zero initial KL but `0.5` sequence TV. |
| Checkpoint A0: Latent-GRPO | PASS | Calibration Q75 JS is 0.06386 nats; 64 confirmation prompts exceed it. |
| Checkpoint A0: SofT-GRPO | FAIL | Calibration Q75 JS is 5.37e-8 nats, about 93,000 times below the frozen 0.005 threshold. |
| Causal non-closure evidence | BLOCKED BY A0 | The frozen protocol requires both checkpoint families to pass before A1. |
| Constrained oracle barycenter | NOT RUN | No evidence that a one-forward solution exists in the natural latent region. |

## Authorization

- Preserve the Latent-GRPO non-closure measurements as a diagnostic asset and
  negative/heterogeneous result.
- Do not reinterpret the Latent-only pass as a PCMC method win. The source-
  faithful SofT action is almost one-hot and closes trivially.
- Do not run A1, Gate B, adapter training, LoRA, or RL for PCMC.
- Do not authorize training.

## Promotion rule

The frozen promotion rule required both checkpoint families to pass A0. SofT
failed by orders of magnitude after an unmodified-source sampler replay audit,
so A1 and Gate B remain closed without threshold relaxation or checkpoint
cherry-picking.

## Evidence

See `A0_RESULT_2026-07-18.md` and
`artifacts/pcmc_gate/local_a0/a0_decision.json`.
