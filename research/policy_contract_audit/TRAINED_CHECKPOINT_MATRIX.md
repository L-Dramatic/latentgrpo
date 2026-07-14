# LPCA Trained-Checkpoint Matrix

Verification date: 2026-07-14. Every remote model is pinned to an immutable
Hugging Face revision. Repository `main` revisions are not valid experiment
inputs.

## Available official targets

| Method | Selected checkpoint | Frozen revision | Size | Audit role |
|---|---|---|---:|---|
| SofT-GRPO | `zz1358m/SofT-GRPO-master`, subfolder `saved_weight/Deepeeek-Qwen-1.5B+SofT-GRPO` | `128d4dbf8ba0f0765d48d39ab41b51be6e879a12` | 3,554,214,752 weight bytes | Compute-compatible trained-policy audit of auxiliary Gumbel action, soft execution, component aggregation, and gauge/null directions. |
| Latent-GRPO | `DJCheng/LLaMA3.2-1B-Instruct-Latent-GRPO-Top10` | `0db191bfd8240199db894de4de800788c845cb18` | 3,014,249,734 repository bytes | Compute-compatible trained-policy audit of clipping atoms, post-noise support selection, omitted selection events, and advantage-dependent backward semantics. |
| Latent-GRPO | `DJCheng/Qwen2.5-Math-7B-Latent-GRPO-4k-Top10` | `b7e1186dd9778b2dcd2ca415d406acbcb9e2fd97` | 15,247,181,586 repository bytes | Higher-compute replication after the 1B protocol is frozen and passes controls. |
| LEPO | none found | n/a | n/a | Public Qwen2.5-3B base-model first-action audit and sequential stress test only; no trained-policy claim is allowed. |
| NF-CoT | none found | n/a | n/a | Formal exact-likelihood positive control until official code and weights appear. |
| Dropout-GRPO | paper states that reference code is released, but no repository link was discoverable | n/a | n/a | Formal replayable-action positive control until an official source can be pinned. |

The SofT-GRPO model repository bundles several baselines and trained models;
only the selected 1.5B subfolder is an input to the compute-compatible audit.
The full 52.2GB repository must not be treated as one checkpoint.
The selected weight file passes the reported `3,554,214,752` byte count and
SHA-256 verification
`eff1ab0b6cca014cc4c766c237672384f2b6eb7f4e871c59d27c5e5a28d6dd93`.
Safetensor metadata exposes 339 tensors with `151936 x 1536` input and output
embeddings. The native tokenizer contains 151,665 entries with maximum id
151,664, so its prompt ids fit the checkpoint vocabulary without repair.

The pinned 1B Latent-GRPO files pass byte-count and SHA-256 verification. Its
tokenizer defines one added `<|compress_token|>` at id `128256`, while the
model embedding and LM head both contain `128256` rows (valid ids
`0..128255`). The ordinary chat template does not emit this token. Audits must
assert that every prompt id is below the model vocabulary size and must not
silently resize the official checkpoint; any protocol requiring the added
token needs a separately justified packaging repair.

Both selected small checkpoints pass the frozen native-load smoke protocol
(`trained_checkpoint_smoke_v1`): every weight, vocabulary, prompt-range,
finite-logit, and repeatability gate passed. Peak allocated GPU memory was
3,415 MiB for SofT-GRPO and 2,866 MiB for Latent-GRPO on the RTX 4060 Laptop
GPU. The Latent-GRPO global tokenizer warning remains visible; only the frozen
prompt was established to be in range.

## Execution order

1. Download only the pinned files and verify the model config, tensor index or
   safetensor, tokenizer, and reported byte count.
2. Load each model with its native tokenizer and run a no-effect forward-pass
   smoke test. This phase may repair file layout and compatibility only.
3. Replay the pinned source sampler on deterministic synthetic logits and on a
   single hidden-effect checkpoint state. Require reconstruction,
   normalization, seed determinism, and source-equivalence controls.
4. Freeze dataset selection, prompts, temperatures, seeds, action counts,
   metrics, and thresholds in a new canonical-hash configuration.
5. Run the 1B/1.5B trained-checkpoint prevalence and consequence audit.
6. Promote to the 7B Latent-GRPO replication only if the smaller run passes all
   controls and exposes a nontrivial, interpretable effect.

## Comparison boundary

This is not a leaderboard comparison. The selected checkpoints use different
base models, upstream data, and latent objectives. Cross-method claims are
restricted to whether each released policy contract is internally coherent and
whether its own semantic distinctions have operational effects. Raw reward or
accuracy differences between checkpoints are confounded and cannot establish
method superiority.

## Stop conditions

- A model that cannot be tied to both an official checkpoint revision and a
  pinned official sampler source is excluded from source-level claims.
- A sampler replay that fails old-policy identity, action reconstruction, or
  deterministic seed replay is repaired before any effect is inspected.
- A mathematically exact reference is not promoted as a method merely because
  it disagrees with a released surrogate.
- A 7B replication is skipped if the 1B audit has no stable effect or if its
  effect disappears under prompt-clustered uncertainty analysis.
