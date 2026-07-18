# Four-Method Checkpoint Smoke Preregistration

Frozen: 2026-07-18, before loading the CODI checkpoint.

## Purpose

This smoke gate establishes that all four Gate 0 methods have a pinned real
checkpoint which loads, accepts a source-native prompt, and produces finite,
repeatable native computation. It is an engineering prerequisite only.

Existing immutable engineering evidence is reused for Coconut, Latent-GRPO,
and SofT-GRPO. Reuse is permitted only by exact artifact hash and imports no
prior scientific conclusion. CODI receives one new native-load run because its
official checkpoint was acquired after those artifacts were produced.

## CODI protocol

- Source: official commit `2c2314662c63e9f482ebc46614ffe9af17a241e5`.
- Checkpoint: official `zen-E/CODI-gpt2` revision
  `fd641b3d3edc59e4f534b55588e906588c9e36bb`.
- Base: local `openai-community/gpt2` snapshot
  `607a30d783dfa663caf39e06633721c8d4cfcd7e`.
- Official architecture flags: BF16, rank-128 LoRA with alpha 32, target
  modules `c_attn/c_proj/c_fc`, 768-dimensional projection with layer norm,
  and six inference latent iterations.
- Prompt: `Question: What is 2 + 3?`, tokenized natively, with the official
  bottom-of-thought id appended under `remove_eos=True` semantics.
- Device: the available local CUDA GPU. No cloud GPU is required.

The runner performs two identical evaluation-mode recurrences from fresh
caches and records only latent hashes and first visible-step logits. It does
not decode an answer.

## Required gates

All are mandatory:

1. source, checkpoint, base-model, and linked-evidence hashes match;
2. `peft==0.15.2`, matching the official requirements;
3. checkpoint load reports zero missing and zero unexpected keys;
4. every prompt id is inside the resized embedding range;
5. exactly six latent embeddings are executed per replay;
6. latent states and first visible-step logits are finite;
7. both replays are bitwise identical;
8. peak allocated CUDA memory is at most 6000 MiB.

Any failure is `HOLD_CHECKPOINT_SMOKE`. No tolerance, dependency version,
prompt, source flag, or missing-key allowance may be changed after observing a
failure. A packaging repair requires a new versioned protocol.

## Interpretation boundary

A pass authorizes freezing a tiny checkpoint-state intervention preflight. It
does not establish causal use, a contract failure, task performance, or paper
viability. Training remains unauthorized.
