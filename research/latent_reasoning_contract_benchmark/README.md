# Latent Reasoning Contract Benchmark (LRC-Bench)

Status: `DETERMINISTIC A0 PASS / CALIBRATION EVIDENCE ONLY`

LRC-Bench is the current candidate direction after the optimizer and estimator
branches in this repository failed their frozen gates. It asks whether a
released latent-reasoning system is specified and implemented coherently from
source and checkpoint provenance through recurrence, execution, learning, and
outcome relevance.

## Narrow thesis

Latent reasoning is not one homogeneous mechanism. A defensible evaluation must
first reconstruct what each release actually computes. LRC-Bench therefore
uses six typed layers:

1. provenance;
2. recurrent state construction;
3. executed object and control flow;
4. learning semantics;
5. stochastic policy measure, only where applicable; and
6. causal or outcome relevance.

The fifth layer is explicitly `N/A` for deterministic Coconut and CODI. It is
required for stochastic Latent-GRPO and SofT-GRPO. This prevents an invalid
comparison in which deterministic hidden-state recurrence is judged by a
continuous-action density that the method never defines.

## What is not claimed

LRC-Bench is not the first benchmark to find shortcuts, weak causal use,
non-sequential computation, unstable latent dynamics, or failures at larger
reasoning depth. Those claims collide with direct 2026 work. It is also not a
method paper at Gate -1 and does not claim that contract violations harm final
accuracy.

The possible contribution is narrower: a source-pinned, executable conformance
benchmark that connects mechanism specification to runtime consequences and
extends the audit to stochastic latent-RL sampler, storage, execution, and
surrogate contracts.

## Gate sequence

- `Gate -1`: exact collision boundary, four auditable methods, pinned source
  and checkpoints, typed schema, no GPU.
- `Gate 0`: source-native fixture replay and exact positive/negative controls,
  no training.
- `Gate A`: checkpoint interventions and strongest simple baselines on frozen
  native tasks.
- `Gate B`: cross-method prediction test: contract metrics must explain
  operational effects beyond entropy, max weight, latent depth, and no-latent
  ablations.
- `Gate C`: only after Gate B, decide whether a repair method is justified.

## Current assets

- Coconut: official source and a pinned independent reproduction checkpoint;
  source-native checkpoint replay already exists elsewhere in this repository.
- CODI: official source and official public checkpoint, now pinned, downloaded,
  and hash verified locally.
- Latent-GRPO: official source and official 1B checkpoint; source-native sampler
  and checkpoint audits already exist.
- SofT-GRPO: official source and official 1.5B checkpoint subfolder; source-
  native sampler and checkpoint audits already exist.

Run the zero-GPU readiness audit with:

```powershell
python -m research.latent_reasoning_contract_benchmark.source_preflight `
  --output artifacts/latent_reasoning_contract_benchmark/gate_minus_one_v1.json
```

Passing Gate -1 means only that the experiment can be run without a provenance
or category error. It is not evidence that the paper thesis works.

Gate 0 now also passes. The runner executes unmodified pinned Coconut and CODI
recurrence classes on a cache-aware toy causal model, executes the unmodified
Latent-GRPO and SofT-GRPO sampler files, and compares each against an
independently owned oracle. All reconstruction, normalization, replay, and
source-equivalence controls pass. This validates instrumentation only; no task
effect has been inspected.

```powershell
_research_env/Scripts/python.exe -m `
  research.latent_reasoning_contract_benchmark.gate_zero `
  --output artifacts/latent_reasoning_contract_benchmark/gate_zero_fixtures_v1.json
```

The four-method checkpoint smoke also passes. Coconut, Latent-GRPO, and
SofT-GRPO are linked by immutable prior engineering artifacts. The official
CODI checkpoint loads with zero missing or unexpected keys and executes six
repeatable latent steps under the official BF16 inference path. This remains a
readiness result: no intervention effect, task accuracy, or paper claim has
been measured.

```powershell
_research_env/Scripts/python.exe -m `
  research.latent_reasoning_contract_benchmark.checkpoint_smoke `
  --output artifacts/latent_reasoning_contract_benchmark/checkpoint_smoke_v1.json
```

The frozen deterministic A0 calibration now passes on 32 label-blind selected
GSM8K questions for both Coconut and CODI. Native replay is exact, and
equal-depth state interventions produce nontrivial paired target-NLL effects.
The untouched confirmation split has not been read. This result validates the
intervention measurement but is not a paper-level causal or novelty claim.
