# P1 Runtime-Repair and Conditional-JVP Gate

**Date:** 2026-07-16  
**Authorization:** GO-RUNTIME-REPAIR  
**Decision:** PASS-REQUEST-SEMANTICS / FAIL-CONDITIONAL-JVP / NO-GO-CAL

This report supersedes only the Windows-runtime and request-semantics blockers in
[P1_CHECKPOINT_PREFLIGHT_REPORT.md](P1_CHECKPOINT_PREFLIGHT_REPORT.md). It does
not rewrite that report's failed dynamic JVP result, alter any frozen P0 file,
or report a scientific P1 outcome.

## Scope

All executions used only the pinned local checkpoint/tokenizer and synthetic
marker/token states. No benchmark item, answer, reward, calibration split,
held-out split, human annotation, optimizer update, or training was used.

The repair created an isolated WSL2 environment in the user's Linux home
directory. It does not modify the Windows Python installation, model files, or
official source checkout.

## 1. Core runtime binding

| Component | Required source environment | Executed repair environment |
|---|---|---|
| Platform | Linux serving runtime | WSL2 Ubuntu |
| Python | 3.11.13 | 3.11.13 |
| PyTorch | 2.6.0 | 2.6.0+cu124 |
| Transformers | 4.51.1 | 4.51.1 |
| SGLang source | pinned local revision | editable local pinned source, 0.4.6.post1 |
| TorchAO | 0.9.0 | 0.9.0 |
| compressed-tensors | 0.11.0 | 0.11.0 |
| sgl-kernel | 0.1.1 | 0.1.1 |

CUDA is available to this environment on the RTX 4060 Laptop GPU. The real
source class sglang.srt.managers.schedule_batch.Req imports successfully.

The only pip-check warning is that decord 0.6.0 declares this platform
unsupported. It is a video-decoding package and is not imported or used by the
request/tokenizer/JVP gates. It remains a full-runtime reproducibility note,
not a substitute for resolving a failure in the tested path.

## 2. Actual SGLang request-semantics gate: passed

The executable evidence is
[p1_linux_request_semantics_preflight.py](p1_linux_request_semantics_preflight.py).
It loads the local tokenizer through the source runtime and calls the actual,
unmodified Req.check_finished() and Req.update_latent_info() methods.

| Item | Observed result |
|---|---|
| </think> ids | [524, 27963, 29] |
| 524 decode | </ |
| Model EOS | 128009 |
| Runtime additional-stop ids | [128008] |
| Tokenizer-only compress id | 128256, outside model rows 0..128255 |
| Default request with emitted 524 | Does **not** finish |
| Default request after update_latent_info | Leaves latent mode and forces [prob=1,0,0], [id=524,-100,-100] |
| Explicit stop_token_ids=[524] | Finishes as a token stop |
| Explicit stop string </ | Finishes as a string stop |
| ignore_eos=True plus token-stop 524 | Does not finish; source then exits latent mode |
| Emitted model EOS 128009 | Finishes as a token stop |

The actual decode source order is:

    append(next_token_id)
    check_finished()
    update_latent_info(...)

A precise source detail matters: the pinned scheduler calls
update_latent_info even if check_finished() has just marked the request
finished. This mutates terminal-request bookkeeping once, but no later decode
step is scheduled. The implementation must therefore state both facts:
generic stop mechanisms prevent continuation, while a terminal request may
still receive this one bookkeeping mutation. It must not describe the source
as skipping update_latent_info entirely after a stop.

**Operational rule for every later experiment:** reject a request configuration
if 524 appears in its explicit stop-token ids, its decoded stop strings, EOS
set, or additional-stop set. For this tokenizer 524 is absent from the model
EOS and runtime additional-stop sets; 128008 is the actual additional stop.

## 3. Fixed-support conditional-JVP repair: failed

The original real-checkpoint JVP failed because small perturbations changed the
source sampler's dynamic top-k support. The repair did not erase that failure.
Instead it introduced an explicit conditional derivative in
[p1_fixed_support_jvp_preflight.py](p1_fixed_support_jvp_preflight.py):

1. Record all five top-k supports from one natural synthetic no-noise trace.
2. At each later step, gather exactly those recorded ids from the current
   logits, softmax them, and form the latent mixture.
3. Never reselect top-k ids inside the conditional branch.
4. Separately record whether a normal dynamic replay would have retained the
   recorded support.

At the reference trace the conditional recurrence exactly reproduces the
pinned source replay:

| Identity check | Error |
|---|---:|
| Initial source mixture vs conditional mixture | 0.0 |
| Five-action endpoint logits, source vs conditional | 0.0 |

The reference proxy sequence is [12, 12, 12, 12, 12]. This validates the
conditional definition at its reference point. It does **not** assert that
finite perturbations stay on that source branch.

For a fixed 257-logit cosine projection, the analytic directional derivative
was 0.9756885. The predeclared conditional grid and result are:

| Epsilon | Finite difference | Relative error | Dynamic support retained (+ / -) |
|---:|---:|---:|:---:|
| 0.20 | 0.9827268 | 0.007 | no / no |
| 0.10 | 0.7745594 | 0.206 | no / no |
| 0.05 | 0.8958340 | 0.082 | no / no |
| 0.02 | 0.6294399 | 0.355 | no / no |
| 0.01 | 2.0104289 | 1.061 | no / no |

The acceptance rule was frozen in code before execution: both 0.10 and 0.05
must have relative error no greater than 0.10. The 0.10 result fails, so this
is FAIL-CONDITIONAL-JVP. The apparently good 0.20 or 0.05 cells cannot be
selected after observation.

This separates two distinct findings:

- **Definition success:** a fixed-support conditional recurrence is source
  exact at its reference trace.
- **Numerical failure:** BF16 endpoint computation is too quantized/non-smooth
  for the frozen finite-difference gate on this hardware, even after discrete
  support selection is removed from the finite branch.

The dynamic support changes in every grid cell are retained as a separate
mechanism warning. A future conditional derivative can only be called
conditional; it cannot be sold as a finite source-path derivative without an
independent support-stability result.

## 4. Resource decision

The successful BF16 checks peak at approximately 2.35 GiB allocated and 2.36
GiB reserved. At the FP32 feasibility check, the GPU had 5,977 MiB free.

The BF16 safetensors checkpoint is 2,996,982,344 bytes. A full FP32 parameter
copy alone is approximately 5.582 GiB, before CUDA allocator overhead,
activations, cache, logits, or the existing background GPU consumer. Therefore
an FP32 recurrence/JVP is unsafe on this current 8-GB GPU. It was deliberately
not attempted.

## 5. Binding next decision

Calibration remains forbidden. There are only two scientifically honest paths:

1. **FP32 repair path:** obtain a clean GPU with at least 16 GB usable memory,
   recreate the same pinned Linux runtime, and rerun the unchanged conditional
   grid. It passes only if the full frozen coarse-grid rule passes. The
   dynamic-support diagnostic remains reported even if this numerical gate
   passes.
2. **No-JVP redesign path:** remove every JVP-dependent mechanism and claim
   from P1, then submit the redesign to a fresh source/estimand review before
   any calibration work. A finite-difference result from the current BF16
   runtime may not be used as supporting evidence.

Until one path passes its stated gate, the project is NO-GO-CAL. The request
semantics repair is real progress, but it does not establish the scientific
phenomenon or license a method result.

