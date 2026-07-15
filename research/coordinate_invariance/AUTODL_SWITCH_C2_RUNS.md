# AutoDL SWITCH C2 Run Ledger

This ledger records operational attempts around the frozen SWITCH C2 protocol.
Scientific outputs remain authoritative only when their configuration and
implementation hashes pass the frozen collector checks.

## Attempt 1 - 2026-07-15

- Project ref: `switch-c2-frozen-v2`
- Project commit: `68fdf904635acd9c247beb155597d5c2f491c91c`
- GPU: NVIDIA H20, 97,871 MiB visible VRAM
- Driver / PyTorch: 580.65.06 / 2.8.0+cu128
- Data disk: 150 GiB class, 149.5 GiB free at resource check
- Managed start: 2026-07-15T01:57:24Z
- Managed stop state: `FAILED`, exit code 1
- Failure time: 2026-07-15T02:01:38Z
- Failure boundary: frozen environment preparation, before checkpoint identity
- Failure cause: the instance had no route to `huggingface.co` while resolving
  the pinned MATH-500 revision (`Errno 101`, network unreachable)
- Scientific interpretation: none; no identity, eligibility, calibration, or
  test artifact was produced
- Returned bundle SHA-256:
  `95e3af381496e550c901176c224e78ccca62029801fb1520bb7ecb5ab76b5618`
- Bundle verification: tar stream read successfully, 33 members
- Evidence retrieved: 2026-07-15T02:03:13Z
- Shutdown verification: SSH endpoint was closed for five consecutive checks
  from 2026-07-15T02:05:19Z through 2026-07-15T02:09:37Z

Remediation for the next attempt:

1. Source AutoDL's `/etc/network_turbo` when present.
2. Probe the official pinned Hugging Face dataset endpoint with bounded retries
   before entering the frozen runner.
3. Keep a locally verified offline cache for the pinned dataset, base model,
   and adapter as a fallback.
4. Reuse the existing virtual environment, source checkout, journals, and
   config-bound artifacts on the persistent data disk.

## Attempt 2 - 2026-07-15

- Project ref: `switch-c2-frozen-v2`
- Project commit: `68fdf904635acd9c247beb155597d5c2f491c91c`
- Managed start: 2026-07-15T04:10:39Z
- Managed stop state: `FAILED`, exit code 1
- Failure time: 2026-07-15T04:10:53Z
- Failure boundary: frozen local tests, before checkpoint identity or weight
  download
- Failure cause: the frozen source SHA-256 was computed from a clean Windows
  CRLF checkout, while the cloud Linux checkout contained the same pinned Git
  blob with LF line endings
- Equivalence diagnosis: the local file contained 1,377 CRLF sequences; after
  CRLF-to-LF normalization it was byte-identical to Git blob
  `eb976d634c06570e0a25b63c3b16c44490a799aa`
- Canonical blob SHA-256:
  `468e4fc05361e4b6150c2a645dddbb3d3ea9a4e60935808804a71bac494615e7`
- Scientific interpretation: none; the fail-closed release test prevented all
  checkpoint-dependent stages
- Returned bundle SHA-256:
  `6803158f3d920e086a40a22def7c0a300acdeb72b5c92be43cc6729e4e7b9029`
- Bundle verification: tar stream read successfully, 33 members
- Evidence retrieved: 2026-07-15T04:11:08Z
- Shutdown verification: SSH endpoint was closed for five consecutive checks
  from 2026-07-15T04:14:15Z through 2026-07-15T04:18:33Z

Remediation:

1. Canonicalize every source pin to bytes read directly from the fixed Git
   commit rather than host-dependent checkout bytes.
2. Require the worktree file, after CRLF normalization, to equal that Git blob.
3. Add the Git blob object ID to the model-source identity record.
4. Rebuild all dependent config, implementation, artifact, and release hashes
   before issuing `switch-c2-frozen-v3`.

## Attempt 3 - 2026-07-15

- Project ref: `switch-c2-frozen-v3`
- Project commit: `974de2153e3315939abbf32a0e316d0bd16c8b92`
- GPU: NVIDIA H20, 97,871 MiB visible VRAM
- Driver / PyTorch: 590.48.01 / 2.8.0+cu128
- Data disk: 100 GiB class, 99.5 GiB free at resource check
- Managed start: 2026-07-15T09:40:30Z
- Managed stop state: `FAILED`, exit code 1
- Failure time: 2026-07-15T09:40:47Z
- Failure boundary: checkpoint-identity runtime initialization, before model or
  adapter weight download
- Passed controls: official Hugging Face network probe, frozen project/source
  identity, CUDA/BF16/resource checks, pinned MATH-500 resolution, and all 21
  runner-local tests
- Failure cause: PyTorch 2.8 lazily initializes CUDA, but its peak-memory reset
  path does not initialize the allocator. The identity runner called
  `reset_peak_memory_stats(cuda:0)` before any tensor or model had initialized
  the allocator, yielding `RuntimeError: Invalid device argument`.
- Scientific interpretation: none; no checkpoint identity, eligibility,
  calibration, or test artifact was produced
- Returned bundle SHA-256:
  `c9bc2fa68368fa94f762526a1c0775aa19be718b569e65eddea1e4eb1c15a4bc`
- Bundle verification: tar stream read successfully, 33 members
- Evidence retrieved: 2026-07-15T09:41:52Z
- Shutdown verification: SSH endpoint was closed for five consecutive checks
  from 2026-07-15T09:44:02Z through 2026-07-15T09:48:20Z

Remediation:

1. Explicitly initialize CUDA immediately before clearing and resetting memory
   statistics in the identity, eligibility, and scientific runners.
2. Preserve the reset before model loading so peak allocated memory retains its
   original meaning.
3. Add an order-sensitive regression test for all three checkpoint-dependent
   entrypoints.
4. Rebind the three implementation hashes without changing any model, prompt,
   split, estimator, threshold, or decision rule, then issue
   `switch-c2-frozen-v4`.
