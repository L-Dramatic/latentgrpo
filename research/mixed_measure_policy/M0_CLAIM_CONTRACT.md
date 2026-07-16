# M0 Claim Contract: Mixed-Measure Latent Policy Optimization

Date frozen: 2026-07-16

## Candidate claim

The released Latent-GRPO sampler should be treated as a policy over a joint
latent action containing an ordered token support, continuous mixture weights,
clipping events, and a proxy token. A new optimizer would use the correct action
law and outperform the released selected-score surrogate without changing the
rollout policy.

## Required contributions

| ID | Required result | Gate |
|---|---|---|
| C1 | Source-faithful proof that the executed action law has discrete, continuous, and atomic components | Must pass |
| C2 | A precise failure mode of the ordinary old/new PPO ratio that is not merely the fixed-bound clipping case handled by CAPG | Must pass |
| C3 | A sampler-preserving gradient or trust-region estimator valid for black-box terminal rewards, with practical cost and a falsifiable advantage over the released surrogate | Must pass |
| C4 | A novelty boundary not subsumed by CAPG, HPO, LEPO, generic boundary-corrected reparameterization, or weak-derivative policy gradients | Must pass |

## Forbidden claim inflation

- Do not claim that mixed discrete-continuous policies are new.
- Do not claim that clipping atoms or a Lebesgue-plus-Dirac measure are new.
- Do not claim that absence of a PPO ratio makes every policy gradient
  impossible.
- Do not call a density-contract diagnosis an optimization method.
- Do not use a changed sampler as evidence for a sampler-preserving method.

## Frozen kill conditions

Kill the method direction before GPU use if any of the following holds:

1. the only correction is CAPG with renamed variables;
2. the estimator requires a differentiable simulator or differentiable reward
   unavailable in outcome-supervised LLM reasoning;
3. augmenting the action with base noise makes its policy score zero or changes
   the executed action when reevaluated under the new policy;
4. unbiased repair requires boundary enumeration, counterfactual rollout, or
   parameter-space finite differences whose cost is not competitive with GRPO;
5. the audit establishes C1/C2 but cannot establish C3/C4.

## Decision rule

- `GO`: C1-C4 pass and a CPU exact gate can compare the proposed estimator with
  finite differences.
- `REDESIGN`: C1/C2 pass and one concrete, bounded-cost C3 estimator remains.
- `KILL`: C3 or C4 has no defensible candidate after adversarial audit.
