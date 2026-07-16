# P1 CPU/Fake-Model Contract Preflight

**Date:** 2026-07-16  
**Scope:** CPU-only toy, source-derived, direct-source sampler, recursive-adapter,
and source-likelihood replay tests; no checkpoint loaded  
**Verdict:** `PASS-CPU-FAKE-CONTRACT / NO-GO-CHECKPOINT-PREFLIGHT`

## What was implemented

- [`p1_fake_preflight.py`](p1_fake_preflight.py) defines an immutable toy
  request state, source-style generic stop checks, joint latent action execution,
  a differentiable recursive latent closure, and a directional JVP helper.
- [`p1_source_sampler_contract.py`](p1_source_sampler_contract.py) is a scalar
  CPU reimplementation of the pinned source sampler branch at
  `sglang/srt/layers/sampler.py` (official-source commit `c0994f…`), including
  its top-p support law, clamped Gumbel perturbation, 524 fallback, proxy, and
  soft-embedding selection rules.
- [`p1_official_sampler_replay.py`](p1_official_sampler_replay.py) loads the
  *unmodified* pinned `sampler.py` with only its import-time serving-framework
  dependencies stubbed, then executes `Sampler.forward` on a one-item CPU fake
  request.  This bypasses the Windows-only `resource` import blocker without
  replacing the sampler body.
- [`p1_recursive_source_adapter.py`](p1_recursive_source_adapter.py) composes
  the direct sampler replay with the audited scheduler order and source
  weighted-embedding semantics.  It is deliberately locked to literal id 524,
  uses candidate-owned fake caches, and never loads a model.
- [`p1_official_objective_replay.py`](p1_official_objective_replay.py) executes
  the unmodified released Gumbel-likelihood function on CPU with only unrelated
  package imports stubbed.  The unavailable Flash-Attention kernel is replaced
  solely for the hard-token algebraic branch.
- [`p1_source_objective_contract.py`](p1_source_objective_contract.py) is the
  independent formula/gradient oracle for that likelihood term.
- [`p1_official_ppo_replay.py`](p1_official_ppo_replay.py) executes the
  unmodified released PPO `compute_policy_loss` body on CPU with its exact
  released masked-mean dependency supplied as a minimal stub.
- [`p1_source_ppo_contract.py`](p1_source_ppo_contract.py) is the independent
  token-mean, dual-clip, negative-advantage-weighted PPO loss oracle.
- [`p1_exact_law_contract.py`](p1_exact_law_contract.py) enumerates finite
  autoregressive joint laws and checks that chain-rule and direct joint KL
  agree, while exposing truncated-support failures as infinite KL.
- [`p1_source_advantage_contract.py`](p1_source_advantage_contract.py) mirrors
  the released eight-member GRPO include-overlong first-mask/winner routine and
  the actor's separate max-length advantage-zeroing branch.
- [`test_p1_fake_preflight.py`](../../tests/test_p1_fake_preflight.py) exercises
  the P1 rules without a model checkpoint or CUDA.
- [`test_p1_source_sampler_contract.py`](../../tests/test_p1_source_sampler_contract.py)
  independently reconstructs the source formulas under fixed random seeds.
- [`test_p1_official_sampler_replay.py`](../../tests/test_p1_official_sampler_replay.py)
  directly compares that contract with the unmodified source forward pass.
- [`test_p1_recursive_source_adapter.py`](../../tests/test_p1_recursive_source_adapter.py)
  verifies stop-first execution, literal-524 exit, independent recursive cache
  closure, RNG isolation, hard-token sentinels, and a four-later-action JVP.
- [`test_p1_source_objective_contract.py`](../../tests/test_p1_source_objective_contract.py)
  compares forward values and gradients with the pinned source likelihood.
- [`test_p1_source_ppo_contract.py`](../../tests/test_p1_source_ppo_contract.py)
  compares the pinned PPO loss and gradients, including self-ratio identity.
- [`test_p1_exact_law_contract.py`](../../tests/test_p1_exact_law_contract.py)
  proves the finite-vocabulary support and chain-rule density contract exactly.
- [`test_p1_source_advantage_contract.py`](../../tests/test_p1_source_advantage_contract.py)
  checks the released first-mask winner and max-length branches.

The module is intentionally **not** a source-equivalent Latent-GRPO adapter and
does not produce scientific measurements. It makes the design contract
executable before source/checkpoint integration begins.

## Passed contracts

1. A proposed proxy is appended before generic stop handling.
2. Length, request stop ids, request EOS ids, tokenizer EOS ids, tokenizer
   additional-stop ids, stop strings, and abort all preempt latent execution.
3. `ignore_eos=True` disables token/EOS checks but does not disable stop strings.
4. A non-stopping proxy equal to the latent-end id consumes the hard end-token
   embedding and switches to visible mode.
5. A non-stopping nonterminal proxy consumes the proposed soft embedding.
6. Candidate closure recomputes at least four later latent actions instead of
   teacher-forcing the reference suffix.
7. A directional JVP through the five-action toy closure matches a central
   finite-difference derivative.
8. A narrow top-p value still admits at least `max_topk` raw candidates, exactly
   as the pinned sampler requires.
9. Fixed-seed Gumbel scores, selected top-k ids, mixture weights, and proposed
   embedding agree with an independent source-formula reconstruction.
10. The sampler's non-obvious fallback is preserved: if **raw** top-1 is 524,
    it discards the noisy choice and returns the raw top-k mixture.  Explicitly
    disabling noise or leaving latent mode does the same.
11. The reporting distribution at `temperature` is recorded separately; it is
    not silently substituted for the raw-logit fallback mixture law.
12. The direct-source replay confirms that the checkout is exactly commit
    `c0994fb781a2d180662bb522d8ff3e8638dcf56d` and, with a fixed RNG seed,
    matches selected ids, perturbed scores, mixture weights, raw-top-k audit
    fields, and final proxy id for noisy, raw-524, and non-latent fallback
    fake requests.
13. The sampler has a binding source quirk: its noisy fallback checks literal
    `524`, whereas scheduler execution reads the configured latent-end id. A
    deliberately mismatched fake configuration proves these differ. The new
    recursive adapter therefore rejects any non-524 end id; the real tokenizer
    and serving configuration must later prove the same literal alignment.
14. The fake recursive adapter consumes a non-stopping end proxy as hard
    `E_524` before reporting the latent-to-visible boundary. Generic stopping
    preempts it and leaves the candidate cache unadvanced. A forced first
    embedding then recomputes its own later source actions and cache rather
    than teacher-forcing the reference suffix.
15. The released source Gumbel likelihood's forward value and
    advantage-dependent straight-through gradient route agree with an
    independent CPU oracle. The finite-difference JVP uses a 1e-2 local step
    because the pinned sampler explicitly casts logits to float32; a 1e-5 step
    is numerically invalid after repeated top-k/softmax operations.
16. The released PPO loss's token-mean aggregation, asymmetric clipping,
    dual-clip lower bound, negative-advantage weight, metrics, and gradients
    agree with an independent CPU oracle. At frozen self-ratio it yields ratio
    one, zero PPO KL, and zero clipping, as required before any Family-B
    hidden-state counterfactual can be constructed.
17. Finite-vocabulary enumeration proves that the full-support chain-rule KL
    equals direct joint KL, and that substituting a truncated visible top-p law
    creates an infinite-KL support violation. The visible top-p boundary keeps
    an item when its preceding cumulative mass equals `top_p`, exactly as its
    source helper's strict `>` exclusion rule specifies.
18. The released include-overlong first-mask winner routine and the separate
    actor max-length zeroing branch are executable on a synthetic group of
    eight. This preserves the source's group-level score normalization,
    best-positive-path selection by mean old log-probability, and the fact that
    the actor's overlong branch zeroes an entire advantage row in-place.

## Test evidence

Run on CPU-only PyTorch `2.11.0+cpu` with Python `3.14.0`:

```text
python -m pytest -q tests\test_p1_fake_preflight.py \
  tests\test_p1_source_sampler_contract.py \
  tests\test_p1_official_sampler_replay.py \
  tests\test_p1_recursive_source_adapter.py \
  tests\test_p1_source_objective_contract.py \
  tests\test_p1_source_ppo_contract.py \
  tests\test_p1_exact_law_contract.py \
  tests\test_p1_source_advantage_contract.py \
  tests\test_behavioral_geometry_joint_kl.py \
  tests\test_behavioral_geometry_analysis.py

53 passed, 8 subtests passed
```

## What this does not establish

- full serving-stack equivalence: the direct replay intentionally stubs only
  import-time framework objects and uses a one-item all-greedy fake batch;
- batch-level source-field behavior, including the source's `[0]` indexing for
  temperatures/noise scales and request batching interactions;
- exact replay of the pinned PPO surrogate;
- Flash-Attention kernel-level equivalence, actual rollout/reward data,
  optimizer update, or the required source-objective hidden-state gradient for
  Family B;
- actual `524` stop-id and decoded stop-string behavior under the checkpoint
  tokenizer/configuration;
- same-history reference/candidate forcing on the real model;
- full-support or truncated-law density correctness on the real model;
- tail/trace coverage, 8-GB differentiable replay feasibility, compute cap,
  power, calibration, held-out results, or training behavior.

Therefore this result does **not** authorize `GO-CHECKPOINT-PREFLIGHT`.
The source-only fake/preflight design scope is now exhausted. The next gate is
an explicit checkpoint-preflight decision: bind this adapter to actual request
state, cache, decoder, tokenizer, and checkpoint logits, but only after the
remaining authorization ledger is reviewed and an explicit `GO-CHECKPOINT-
PREFLIGHT` is recorded. No real checkpoint may be touched before that gate.
