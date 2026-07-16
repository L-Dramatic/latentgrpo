# M0 Claim Contract: Stratified Gumbel Group Exploration

Date frozen: 2026-07-16

## Candidate claim

Couple Gumbel randomness across the G rollouts of one latent-GRPO group so that
every rollout remains marginally on-policy, while the group covers more distinct
reasoning paths and produces more informative reward variation at fixed rollout
cost.

## Required gates

| ID | Requirement | Status needed |
|---|---|---|
| C1 | Every trajectory has the exact original conditional sampling law despite autoregressive divergence | Must pass |
| C2 | Coupling improves informative-group probability or estimator variance without lowering expected reward | Must pass |
| C3 | The actual group-relative gradient remains unbiased, or a correction removes coupling bias with no material extra rollout cost | Must pass |
| C4 | The method has a contribution beyond Arithmetic Sampling, RQMC policy gradients, generic antithetic sampling, and recent GRPO diversity methods | Must pass |

## Frozen kill conditions

Kill before model inference if:

1. group baselines become correlated with the trajectory score and induce a
   nonzero cross term;
2. unbiasedness requires discarding group-relative advantages or drawing an
   independent baseline group;
3. the surviving method is only RQMC/antithetic policy gradient applied to LLM
   uniforms;
4. sequence-level stratification is already covered by Arithmetic Sampling;
5. diversity comes from forcing low-probability local tokens and has no
   coherence-preserving mechanism distinct from prior work.

## Decision rule

- `GO`: C1-C4 pass.
- `REDESIGN`: C1/C2 pass and one cheap, exact C3 correction remains novel.
- `KILL`: C3 or C4 fails after the exact one-step gate.
