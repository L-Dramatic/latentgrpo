# P1 Checkpoint Identity and Execution Preflight

**Date:** 2026-07-16  
**Authorized scope:** GO-CHECKPOINT-PREFLIGHT only  
**Decision:** FAIL-CHECKPOINT-PREFLIGHT / NO-GO-CAL / NO-GO-HELD-OUT  
**Scientific status:** No P1 phenomenon, intervention, calibration, held-out,
training, or optimization result was collected.

## 1. Scope and decision rule

This is an engineering identity preflight on the local checkpoint
E:\LantentGRPO\_models\Latent-GRPO-Llama-1B. It used a fixed five-item synthetic
marker suite only:

    <think>
    0 0 0 <think>
    alpha <think>
    alpha beta gamma <think>
    synthetic marker <think>

There was no benchmark, question, answer, reward, calibration split, held-out
split, human annotation, parameter update, or training run. The suite and the
fixed finite-difference grid are embedded in
[checkpoint_identity_preflight.py](checkpoint_identity_preflight.py), which
loads local files only and writes no artifact during execution.

The gate passes only if all of the following hold:

1. Checkpoint/tokenizer/head identity is compatible with the source adapter.
2. Source-sampler replay and independently owned recursive caches reproduce.
3. A natural five-action synthetic closure admits a stable, fixed-branch
   directional finite-difference check at the predeclared primary epsilon 0.01.
4. Real request termination semantics are verified in the pinned serving
   runtime.

Items 1--2 pass. Item 3 fails, and item 4 is not executable on the present
Windows/runtime combination. Therefore calibration and all scientific P1 work
remain blocked. This is a **preflight failure**, not a claim that the horizon
sufficiency hypothesis is false.

## 2. Immutable input binding

The frozen P0 files were read only and their hashes remained unchanged:

| P0 artifact | SHA-256 |
|---|---|
| P0_CLAIM_CONTRACT.md | 78c95be7d81be5322058e7b64a763986d50205b5ae3c4f7fcd8892a86539e60c |
| P0_MATHEMATICAL_SPEC.md | 9665d75e5082b9e428d6814200bec1045e124fdfb287bd613abdb4a9acf15452 |
| P0_COLLISION_MATRIX.md | f87321987982b91e5321307d869c5ad975f628ab7567c3e42d614fe39ba5558e |
| P0_KILL_CRITERIA.md | 5a1a093fef3482d30355bbec3522dc9cc8f71f2a527bfd18be7e8bb49e469b7c |
| P0_SWITCH_C2_COMPATIBILITY.md | b0031f6578886b20434b0d4b12b8d3d241cdb1a87f274b7ac3a32f68d8ba639a |

The pinned official-source revision is
c0994fb781a2d180662bb522d8ff3e8638dcf56d. The local checkpoint files were
bound before loading:

| File | SHA-256 |
|---|---|
| model.safetensors | ab6dcd385f01eb2a2aab4d449a5e922ebbd672c4c3189c01480de80a74379833 |
| config.json | bd6c347cd69ca9fa1da4ef3fa321ba252ed18f0776ea0ab1d35586cf1e726f01 |
| generation_config.json | b746ceb4d99812126ef4d0bac3987d197b2a32fd6de9e751c5432fe95e1e2b16 |
| special_tokens_map.json | 7b3ca363d95921e9897d55df44701d26412cb0e33165812af96721ca26e59267 |
| tokenizer_config.json | 2f8991526d072bf96e727dd57763206a643d8e14849c02f89e56ad69c050ae36 |
| tokenizer.json | 4af0b6021b16d8b2e8b75e783099483a1070d25156dc4930b10b4fad4a9219c5 |

The safetensors header contains 147 tensors. The byte ranges for
lm_head.weight and model.embed_tokens.weight are BF16 [128256, 2048] and
byte-identical (tensor SHA-256
b111063841df6851fa28395ca43f6570c81486a16d4830d0026cb62cf14a7b5d).

## 3. Passing identity, sampler, and recursive-execution checks

| Check | Result | Interpretation |
|---|---:|---|
| Input/output embedding shapes | [128256, 2048] / [128256, 2048] | Matches config.json. |
| Tied input/output storage | true | The loaded head is tied to the input embedding. |
| LM-head bias | absent | No unmodelled bias term. |
| Selected direct-head logit error | 0.0 | Manual head rows exactly match model logits on audited ids. |
| Repeated synthetic prefill error | 0.0 | The same local model prefill is bit-identical. |
| Fixed-seed noisy source sampler | identical | The unmodified pinned sampler body replays deterministically on real checkpoint logits. |
| Independent deterministic recursive closures | 0.0 logit difference | Candidate and reference run separately but reproduce. |
| Cache storage | disjoint | Candidate and reference do not alias KV-cache tensors. |
| Source exit replay | [12, 19, 524] then visible | A non-stopping 524 is consumed as hard E_524; no extra latent action is sampled after it. |

The tokenizer is internally compatible with the source convention in the
limited Hugging Face audit: </think> tokenizes to [524, 27963, 29], and token
524 decodes to </. It is neither the HF EOS (128009) nor an HF special id. The
checkpoint has 128256 embedding/head rows.

One strict guard remains important: the tokenizer additionally exposes
<|compress_token|> as id 128256, which is **outside** the model vocabulary
(0..128255). The adapter must reject or avoid this tokenizer-only id; it must
never remap it into a model token or use it as a latent action.

## 4. JVP feasibility gate: failed

The first predeclared synthetic prompt with at least five natural latent
actions was 0 0 0 <think>. Its reference proxies were
[12, 12, 12, 12, 12]. Parameters were frozen; only the first latent-mixture
embedding was perturbed along a deterministic unit direction. The scalar target
was logit 0 at the resulting closure endpoint.

The backpropagated directional derivative was **1.1420258**. The primary
finite-difference rule was fixed before the diagnostic grid:

    epsilon = 0.01
    accept only if the full proxy/support/exit trace is unchanged and abs_error <= 0.1

At the primary step, the requested finite difference was **3.125** and the
absolute error was **1.9829742**. The proxy sequence and exit flag were stable,
but the source top-k **support** was not. Hence the two perturbed runs did not
remain on the same differentiable source-sampler branch.

The complete, predeclared diagnostic grid was retained; it was not used to
select a replacement passing step:

| Requested epsilon | FD / requested epsilon | Effective scale / requested | Abs. error | Proxy and exit stable | Top-k support stable |
|---:|---:|---:|---:|:---:|:---:|
| 0.100 | 0.9375 | 0.997 | 0.205 | yes | no |
| 0.050 | 1.2500 | 0.987 | 0.108 | yes | no |
| 0.020 | 1.5625 | 0.975 | 0.420 | yes | no |
| 0.010 (primary) | 3.1250 | 0.979 | 1.983 | yes | no |
| 0.005 | 0.0000 | 1.057 | 1.142 | yes | no |
| 0.002 | -15.6250 | 0.655 | 16.767 | yes | no |
| 0.001 | 31.2500 | 0.359 | 30.108 | yes | no |

The shrinking effective perturbation below 0.002 confirms a BF16 quantization
floor. It cannot, however, explain away the failure: the top-k support changes
at every recorded step, including 0.1. Thus this preflight does not establish a
source-faithful, smooth five-action derivative. The nearer error at 0.05 is
still above the frozen tolerance and has unstable support, so it is explicitly
**not** treated as a pass.

## 5. Runtime and serving-semantic blockers

The local isolated runtime successfully loaded the checkpoint on an 8-GB RTX
4060 Laptop GPU, but it is not source-equivalent:

| Component | Pinned official environment | Executed local environment |
|---|---|---|
| Python | 3.11.13 | 3.10.9 |
| PyTorch | 2.6.0 | 2.5.1+cu124 |
| Transformers | 4.51.1 | 4.55.4 |
| Serving package | full source-pinned SGLang | direct sampler-body replay with narrow import stubs |

The run peaked at 2,521,968,128 allocated bytes (2.35 GiB) and
2,543,845,376 reserved bytes (2.37 GiB), leaving 4,860,149,760 free bytes at
the end of this check. This proves only that this narrow BF16 engineering
preflight fits. It does not license training, full source serving, suffix
rollouts, or FP32 model differentiation on an 8-GB card.

The full SGLang serving stack cannot be imported on this Windows host because
it imports the Unix-only resource module. Consequently, this gate could not
observe an actual Req.check_finished() call, the complete runtime
EOS/stop/additional-stop sets, or decoder stop-string matching. The HF
tokenizer evidence above is necessary but not a replacement for those request
semantics.

## 6. Required repair before any calibration authorization

Do not begin GO-CAL or use any calibration prompt. The next work must be an
engineering repair with a new explicit authorization and all of these exit
criteria:

1. **Source-equivalent request replay.** Run the pinned revision in a Linux or
   WSL environment with Python 3.11.13, PyTorch 2.6.0, Transformers 4.51.1,
   and the actual SGLang package. On the same synthetic suite, capture the
   real request stop/EOS/additional-stop state, decoder behavior, and the
   append-check-finish-update ordering. Prove that 524 is absent from every
   stopping mechanism for the relevant request and that E_524 is consumed
   exactly once before visible decoding.
2. **Mathematically explicit JVP branch.** Either define a conditional JVP
   with a recorded, frozen top-k support and a support-margin certificate, or
   remove JVP dependence from the proposed P1 mechanism. A differentiable
   surrogate may not be silently substituted for the released sampler; any
   such conditional branch must be declared and separately validated against
   the source action law.
3. **Repeat the fixed derivative gate.** Use the predeclared primary step and
   report proxy, support, exit, effective perturbation, finite difference, and
   analytic derivative together. Passing requires a stable declared branch and
   the frozen tolerance; a more favourable grid point cannot be chosen after
   observation.
4. **Reassess resources only after items 1--3.** If the repair requires full
   FP32 model differentiation or full serving caches, obtain a larger, clean
   GPU environment before running it. Do not gamble on the current 8-GB card.

Until those conditions pass, the defensible statement is narrowly positive:
the local checkpoint can replay the source sampler, tied head, cache ownership,
and the hard-524-to-visible transition under an isolated adapter. It is not yet
a valid execution substrate for the P1 natural-update experiment or an AAAI
method claim.

