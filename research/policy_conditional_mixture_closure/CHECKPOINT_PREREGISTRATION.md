# PCMC Checkpoint Gate A/B Preregistration

Frozen: 2026-07-18

## Claim under test

PCMC is viable only if a natural soft latent action is not closed under the
checkpoint's one-step transition and that non-closure predicts a causal loss
in answer quality. One-step divergence is a screening variable, not the paper
claim. No result in Gate A authorizes training.

## Assets and statistical units

- Checkpoints: `Latent-GRPO-Llama-1B` and `SofT-GRPO-Qwen-1.5B`, each pinned by
  model SHA-256 and clean source commit.
- Data: all 500 MATH-500 examples at the pinned Arrow revision and SHA-256.
- Split: a label-blind salted hash fixes 250 calibration and 250 confirmation
  prompts. The answer field is not read during A0 selection.
- A0 unit: one prompt and the first source-native soft action at its frozen
  seed. There is no entropy-based action search.
- A1 unit: one selected confirmation prompt. The eight continuation seeds are
  paired across all four interventions; prompt-level means, not rollouts, are
  the independent analysis units.

## Source-native action contracts

Both checkpoints use their own training-time prompt template and stochastic
soft sampler. The exact Top-p, Top-k, softmax temperature, Gumbel law, clipping,
noise scale, and structural end token are frozen in
`configs/pcmc_gate_ab_v1.json` and enforced by code.

The structural `</think>` component is measured before intervention. Events
with more than 0.01 structural-end mass are ineligible. Otherwise the end
component is removed and content weights are renormalized. This conditions A0
on continuing thought and prevents termination probability from masquerading
as content-mixture non-closure.

## A0: one-step closure screen

For content token embeddings `E_i` and frozen action weights `q_i`, compare:

- arithmetic transition: `p(. | sum_i q_i E_i)`;
- hard-branch teacher: `sum_i q_i p(. | E_i)`.

The event score is Jensen-Shannon divergence in nats at temperature 1.0. KV
caches for the arithmetic and hard-branch calls must be deep-copy disjoint.

A checkpoint advances only when all are true:

- at least 70% of its 250 calibration prompts are eligible;
- calibration Q75 JS is at least 0.005 nats;
- at least 48 confirmation prompts exceed that frozen Q75 threshold.

At most 64 high-gap and 64 low-gap confirmation prompts are selected by a
second salted hash. Both checkpoints must advance. Failure by either checkpoint
KILLs PCMC before continuation generation.

## A1: causal continuation test

Frozen interventions are arithmetic, randomized hard branch, top-1, and
temperature-0.5 sharpened soft action. Each prompt-method pair receives eight
paired visible-continuation seeds and at most 256 visible tokens.

Each checkpoint must independently satisfy every condition:

- high-gap randomized-hard minus arithmetic accuracy is at least 3 points;
- its prompt-bootstrap 95% lower bound is greater than 0;
- gap versus reward-loss Spearman is at least 0.25 with lower bound above 0.10;
- partial Spearman controlling action entropy, maximum weight, effective
  support, and prompt length is at least 0.15 with lower bound above 0;
- randomized hard exceeds top-1 and sharpened by at least 1 point each.

Bootstrap count is 10,000. No threshold may be relaxed after A0 is observed.
Both checkpoints must pass to authorize B0.

## Gate B

B0 asks whether a projected-Adam embedding oracle can substantially close the
one-step gap inside a frozen norm/trust region. B1 then tests posterior-updated
hard-branch sequence mixtures for 16 visible steps. B0 and B1 remain blocked
until A1 passes, and their failure KILLs the method direction.

## Evidence boundaries

- `engineering-preflight`: sampler, tokenizer, cache isolation, memory, and
  timing only; never scientific evidence.
- `A0`: checkpoint phenomenon screen only; never a causal or method claim.
- `A1`: causal mechanism evidence; still not a trained-method result.
- `B0/B1`: method-feasibility gates; only their joint pass may authorize method
  implementation and training.

Invalid, incomplete, hash-mismatched, optional, or out-of-order work is an
operational block, not a favorable or unfavorable scientific result.
