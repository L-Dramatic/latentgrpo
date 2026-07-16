# P1 Source-Only Completion Audit

**Date:** 2026-07-16  
**Bound source:** official Latent-GRPO commit `c0994fb781a2d180662bb522d8ff3e8638dcf56d`  
**Scope:** CPU-only and fake-request preflight; no checkpoint, GPU, calibration,
held-out group, risk label, or training run  
**Decision:** `PASS-SOURCE-ONLY-FAKE / NO-GO-CHECKPOINT-PREFLIGHT`

## Completed, reproducible contracts

| Contract | Evidence |
|---|---|
| Stop-first request semantics | `p1_fake_preflight.py`; abort, length, every token/EOS source, stop string, and `ignore_eos` tests |
| Official sampler | Unmodified pinned `Sampler.forward` replayed on a CPU fake request with fixed RNG |
| Literal end-id invariant | Direct replay proves noisy fallback is hard-coded to `524`; the recursive adapter rejects any other configured end id |
| Recursive execution | Candidate-owned fake cache, hard `E_524` consumption, independent future top-k actions, timeout, isolated RNG, and four-later-action JVP |
| Source Gumbel likelihood | Direct replay of the released likelihood body plus independent forward/backward contract |
| PPO and advantages | Direct replay of released PPO loss and include-overlong first-mask winner, plus token-mean/dual-clip/max-length contracts |
| Density/support | Exact finite-vocabulary enumeration of joint KL, chain-rule KL, and truncated-support failure |

The complete command in [`P1_CPU_FAKE_PREFLIGHT_REPORT.md`](P1_CPU_FAKE_PREFLIGHT_REPORT.md)
currently reports `53 passed, 8 subtests passed`.

## Why this is still not a checkpoint authorization

The following conditions require real checkpoint/request state and remain
unverified by design:

1. fixed-seed identity replay of checkpoint logits, selected ids, perturbed
   scores, mixtures, proxy stream, and visible ids;
2. actual tokenizer/configuration proof that id 524 is in vocabulary, is the
   intended latent-end token, is absent from every EOS/stop/additional-stop set,
   and cannot form a frozen stop string under actual decoder state;
3. actual serving request/KV-cache and latent-to-visible replay, including
   source position alignment and decoder updates;
4. checkpoint embedding/LM-head/vocabulary identity and absence of hidden logit
   scale, bias, softcap, or normalization not represented in the source-gradient
   construction;
5. measured 8-GB JVP feasibility, semantic-tail eligibility, numeric compute
   cap, and all execution-freeze hashes; and
6. real-model source-objective hidden-state gradient, policy self-ratio, and
   exact Flash-Attention/CUDA behavior.

These are not superficial missing tests. They are the boundary between a valid
fake-source contract and a scientific measurement. The current authorization
therefore remains `NO-GO-CHECKPOINT-PREFLIGHT`.

## Required next authorization

Before reading/loading checkpoint tensors or running a model forward pass,
record an explicit `GO-CHECKPOINT-PREFLIGHT`. Its scope must be limited to the
six identity/tokenizer/cache/head/compute checks above. It must not open
calibration groups, held-out groups, risk labels, training, or GPU rental.
