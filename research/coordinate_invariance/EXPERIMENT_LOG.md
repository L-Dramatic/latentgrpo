# Coordinate-Invariance Experiment Log

This log is append-oriented. Failed gates remain recorded with their original thresholds and are not rewritten as passes.

## 2026-07-14: CPU toy contract

- Artifact: `artifacts/coordinate_invariance/toy_contract_v1.json`
- Status: **Pass**
- Evidence level: implementation contract only
- Result: exact round trips, behavior identity, and functional-distance identity passed; orthogonal Euclidean neighbor flips were zero; anisotropic and nonlinear positive controls were detected.
- Interpretation: the basic chart and metric code behaves as mathematically expected. This is not model evidence.

## 2026-07-14: Public GPT-2 Coconut integration smoke

- Artifact: `artifacts/coordinate_invariance/public_gpt2_coconut_smoke_v1.json`
- Checkpoint: `connordilgren/gpt2-gsm8k-coconut`, `checkpoint_33`
- Checkpoint SHA-256: `f7a9b5fda7c5c2afa972aaa22ac9aef13d6f083e202fe8a09649d810e3593213`
- Coconut source: `facebookresearch/coconut` at `27273cb8cca4bb763c041a63b036d0c3b7cbbb48`
- Status: **Pass**
- Evidence level: real-checkpoint integration contract
- Result: the official forward, identity chart runner, and condition-number-12 affine chart runner were bitwise identical in filled embeddings and logits. Six latent steps were captured and replayed. The matched factual replay divergence was zero.
- Interpretation: chart insertion and replay do not manufacture behavioral differences.

## 2026-07-14: Preregistered GPT-2 Coconut coordinate pilot v1

- Config: `research/coordinate_invariance/configs/public_gpt2_coconut_pilot_v1.json`
- Artifact: `artifacts/coordinate_invariance/public_gpt2_coconut_pilot_v1.json`
- Trace: `artifacts/coordinate_invariance/traces/public_gpt2_coconut_pilot_v1/`
- Data: 64 deterministically selected examples from the local GSM8K evaluation file
- Status: **Fail under the frozen gate**
- Instrument status: **Pass**
- Phenomenon status: **Fail**

### Passed controls

- All no-op chart embeddings and logits remained within tolerance.
- All charted latent values recovered bitwise before model consumption.
- Orthogonal Euclidean neighbor flip rate was exactly zero.
- Affine interpolation differed from native interpolation by at most `1.67e-12`.
- Matched continuation replay and explicit compute accounting completed without numerical errors.

### Failed preregistered conditions

- Condition-number-4 random affine charts produced a mean Euclidean neighbor flip rate of `0.09375`, with 95% bootstrap interval `[0.046875, 0.14082]`; the frozen lower-bound requirement was `0.10`.
- Condition-number-12 random affine charts produced a mean flip rate of `0.22656`, with interval `[0.15625, 0.30469]`; the frozen lower-bound requirement was `0.25`.
- The thresholds were not changed after observing the result.

### Behavioral signal among observed flips

- The behavior chart provided 13 flipped queries, above the minimum audit count of 12 but below the desired count of 20.
- Identity-selected and affine-selected neighbor replacements had mean direct continuation divergence `1.326`, interval `[0.370, 2.541]`.
- Their generated continuation mismatch rate was `0.692`, interval `[0.500, 0.846]`.
- Equal-native-norm identity versus affine noise had mean direct divergence `3.05e-4`, interval `[1.25e-4, 5.55e-4]`.
- Native versus nonlinear interpolation had mean direct divergence `7.42e-6`, interval `[1.12e-6, 1.85e-5]`, but no sampled-token mismatch at the tested horizon.

### Decision

The gate is not reclassified as a pass. The result says that random moderate affine charts change fewer nearest-neighbor decisions than anticipated, even though the decisions that do change can have large downstream consequences. This is a mixed signal, not yet a reason to implement FCTR.

## 2026-07-14: Post-gate chart regularity diagnostic

This diagnostic is exploratory and used the pilot traces. It is not independent confirmation.

- `sinh scale=0.10` caused a 67.2% neighbor flip rate, but its local Jacobian condition number was too strong: median `4.97`, 95th percentile `17.46`, maximum `86.16`. It is rejected as the main evidence chart.
- `sinh scale=0.05` caused a 28.1% flip rate, with median local condition `1.73`, 95th percentile `3.04`, maximum `6.60`.
- `sinh scale=0.04` caused a 23.4% flip rate, with median local condition `1.45`, 95th percentile `2.19`, maximum `3.99`.
- `sinh scale=0.03` caused a 14.1% flip rate, with median local condition `1.24`, 95th percentile `1.62`, maximum `2.45`.

### Next gate

Freeze `sinh scale=0.04` as a moderate nonlinear chart and test it only on examples excluded from pilot v1. The new gate must report that this chart was selected after pilot inspection. It must pass no-op identity, local-condition bounds, an orthogonal negative control, held-out neighbor-flip frequency, and held-out continuation consequences. Failure stops the current nearest-neighbor evidence line; it will not trigger a stronger chart or looser threshold.

## 2026-07-14: Disjoint GPT-2 Coconut nonlinear holdout v1

- Config: `research/coordinate_invariance/configs/public_gpt2_coconut_holdout_v1.json`
- Artifact: `artifacts/coordinate_invariance/public_gpt2_coconut_holdout_v1.json`
- Trace: `artifacts/coordinate_invariance/traces/public_gpt2_coconut_holdout_v1/`
- Data: 100 deterministically selected examples with zero overlap with the 64-example pilot
- Chart: `sinh scale=0.04`, selected after the pilot and frozen before this holdout
- Status: **Fail under the frozen gate**
- Decision: **Stop the nearest-neighbor evidence line**

### Passed controls and consequence checks

- The identity no-op recovered every consumed latent bitwise and produced zero embedding and logit error.
- The orthogonal negative-control neighbor flip rate was exactly zero.
- The nonlinear chart remained moderate over the evaluated states: local Jacobian condition-number median `1.397`, 95th percentile `2.147`, maximum `2.977`.
- Sixteen queries changed nearest neighbor, so the minimum behavioral audit size of 15 was met.
- On those flips, direct continuation divergence had mean `4.082` and 95% bootstrap interval `[1.396, 7.410]`, well above the frozen lower-bound requirement of `0.001`.
- Generated continuation mismatch rate was `0.719`, interval `[0.563, 0.875]`, above the frozen lower-bound requirement of `0.10`.

### Failed preregistered condition

- The nonlinear neighbor flip rate was `0.16` over 100 held-out examples, with 95% bootstrap interval `[0.09, 0.23]`.
- Its frozen lower-bound requirement was `0.10`; the observed lower bound missed it by `0.01`.
- The threshold is not relaxed and the result is not rounded into a pass.

### Interpretation

Moderate smooth reparameterizations do alter some Euclidean nearest-neighbor choices, and the altered choices can have large downstream effects. However, the held-out frequency estimate did not clear the prespecified robustness threshold. This makes nearest-neighbor instability useful as an illustrative failure case, but too fragile to carry the paper's central empirical claim. Per the frozen decision rule, no stronger chart is selected and no second architecture is spent on this line.

## 2026-07-15: FCTR coordinate-transport toy gate v1

- Config: `research/coordinate_invariance/configs/fctr_toy_gate_v1.json`
- Config file SHA-256:
  `a3118511d214f3e3988f64e277afae3fa4836625180c2fb1f90aac84bfb2cff8`
- Canonical config SHA-256:
  `4fbab76baf0224885b7aa86a59510771eca74e268ea3b050df04b95434eb337d`
- Artifact: `artifacts/coordinate_invariance/fctr_toy_gate_v1.json`
- Artifact SHA-256:
  `66dc3deebee61fb1a70b5b8e47de5dadee1bd1f0b352e3b074da293a6c83277a`
- Status: **Pass**
- Evidence level: deterministic implementation contract only

All seven frozen controls passed. Identity and orthogonal Euclidean updates
transported within `6.10e-16`. Under condition-number 4 and 12 affine charts,
coordinate-Euclidean update relative errors were `0.783` and `1.936`, with
transported direction cosines `0.820` and `0.674`. FCTR steps transported
within `9.75e-16`, preserved the `0.001` functional budget within `2.17e-19`,
and preserved predicted gain within `6.94e-18`.

The observed Euclidean chart dependence and FCTR invariance are expected
mathematical controls. They do not establish a latent-reasoning failure or an
advantage over next-token FishBack. No training or cloud run is authorized by
this result.

## 2026-07-15: SWITCH source and checkpoint screen

- Official source: `LARK-AI-Lab/SWITCH` at
  `d8d97cdc6276fcfa6e48f6a6b19ce472c7b87fcd`
- Official adapter: `LARK-Lab/SWITCH-Phase3-GRPO-LoRA-Qwen3-8B` at
  `246fee75d774c02a110ea8608ac841a916dd5d35`
- Base model: `Qwen/Qwen3-8B` at
  `b968826d9c46dd6066d109eabc6255188de91218`
- Official source tests: `26 passed`

SWITCH provides the needed second hidden-recurrence architecture and a
nontrivial visible continuation. The paper-final launcher disables optional
latent noise and latent replay by default, so its released result must not be
described as Gaussian latent policy optimization. The pinned base and adapter
contain about 24.2 GB of weight files before runtime state. Their download and
the 40/80 GB GPU run remain deferred until the local differentiable Coconut
integration gate and a frozen SWITCH protocol pass.

## 2026-07-15: FCTR Coconut C1a engineering stop

- Config: `research/coordinate_invariance/configs/fctr_coconut_smoke_v1.json`
- Canonical config SHA-256:
  `2285801576ed776be85d5c386c14e9d743deb115a38e557725443987b3a8fa70`
- Artifact: none
- Scientific status: **Not evaluated**
- Engineering status: **Stopped before the first Jacobian**

The selected efficient SDPA backend lacked the second derivative needed by
PyTorch's JVP implementation. No effect, chart comparison, or gate value was
produced. The preregistered repair changes only the attention backend to the
mathematically equivalent eager implementation; all scientific choices and
thresholds remain unchanged in C1b.

## 2026-07-15: FCTR Coconut C1b engineering stop

- Config: `research/coordinate_invariance/configs/fctr_coconut_smoke_v1b.json`
- Canonical config SHA-256:
  `ee8d5701f6e2253d74efb8ec0a57de5ecd3658148db8339d1cf1f3d78ed7b0be`
- Artifact: none
- Scientific status: **Not evaluated**
- Engineering status: **Stopped during the first chart comparison**

Eager attention enabled the native Jacobian measurement. The run then stopped
on a CPU/CUDA device mismatch in the analytic Jacobian transport comparison,
before any report was written. C1c moves only that analytic comparison to CPU
float64. Its scientific configuration and thresholds are unchanged.

## 2026-07-15: FCTR Coconut C1c differentiable integration gate

- Config: `research/coordinate_invariance/configs/fctr_coconut_smoke_v1c.json`
- Config file SHA-256:
  `45fe953d4f4be07338e9b5f198c79cb83d7adff1dfa22dcabdc867ff0fa96fda`
- Canonical config SHA-256:
  `f3ce1bacda69040d8190d6e55d3e57bcb159cd2ca2a5610b5e700566333e8245`
- Runner script SHA-256:
  `389f9ba43b7f6a4219e5402495959d8f2d1ac1350b79705382f1c518dc9b135c`
- Artifact: `artifacts/coordinate_invariance/fctr_coconut_smoke_v1c.json`
- Artifact SHA-256:
  `4ed3b83325ea7d9d73f6546058fd2980fc394305bdcab61d9cdec37488e6e495`
- Status: **Pass: 11 of 11 frozen gates**
- Evidence level: real-checkpoint differentiable integration contract only

The three-token target was `### 18` followed by EOS at latent pass 3. The
autograd directional derivative agreed with central finite differences to
`3.86%`, below the frozen `10%` ceiling. Identity errors were zero. Across the
orthogonal and condition-number 4/12 charts, Jacobian error was at most
`1.97e-6`, gradient error `5.64e-7`, metric error `1.25e-4`, and FCTR step
transport error `1.99e-5`, all below their frozen ceilings. Coordinate-Euclidean
updates changed by relative L2 `1.30` and `0.97` under the anisotropic charts,
passing the positive control.

The run used 27 function evaluations, 270 model forward calls, 33.4 seconds,
and 911.5 MiB peak allocated memory on the local RTX 4060 Laptop GPU. This
validates the real-model differentiation and transport implementation. It does
not authorize FCTR training because this checkpoint's visible continuation is
the already-diagnosed fixed delimiter plus short answer.

## 2026-07-15: LPCA Stage B2 portfolio decision transferred

The policy-contract branch completed its frozen Stage B2 reward-gradient audit
before FCTR was revived. All numerical controls passed and the exact versus
released-surrogate gradient directions were materially different, but the
exact candidate was operationally worse:

- exact gain: `0.0063270`;
- released-surrogate gain: `0.0079559`;
- paired exact-minus-surrogate difference: `-0.0016289`;
- prompt-clustered 95% interval: `[-0.001942, -0.001349]`;
- exact-minus-surrogate difference was negative at every frozen temperature.

Matched exact-density-substitution training is permanently unauthorized. This
does not erase the policy-contract mismatch, but it prevents LPCA's repair from
being presented as a performance method. The result is preserved in
`research/policy_contract_audit/EXPERIMENT_LOG.md` and is not reused as positive
evidence for FCTR.

## 2026-07-15: SWITCH source-equivalence preflight v1

- Config: `research/coordinate_invariance/configs/switch_c2_source_preflight_v1.json`
- Canonical config SHA-256:
  `42ce262f98692a392ea33c04c63a2a941d55590e35def44cfb761a5574790431`
- Artifact: `artifacts/coordinate_invariance/switch_c2_source_preflight_v1.json`
- Artifact SHA-256:
  `3fae56acf8332a10d136e4b897c2653fc5be307a09a4eb5c2c479349bfa88d9b`
- Status: **Pass**
- Evidence level: source and release-metadata contract only

The source, evaluator choice, tokenizer IDs, adapter metadata, minimum dwell,
and paper-final launcher defaults all matched their pins. The preflight also
records an important boundary: the RL implementation decides exit by argmax
after minimum dwell and force-appends `</swi>` with `sampled_mask=0`. The end
marker is not a sampled GRPO action. This is retained as policy-contract
evidence and prevents broad claims about SWITCH sampling that the source does
not support.

## 2026-07-15: SWITCH C2 protocol and implementation freeze

- Identity config file SHA-256:
  `8f6f7bcc1edf52db3a39b8bc58585709e7526cbd46d07aa521e305ec379c1519`
- Identity canonical config SHA-256:
  `e2df1ad9f577c8d44c97e8738575ea383ae1e75287206792aca42c56e8da09b5`
- Scientific config file SHA-256:
  `a73759e1e0fed584d33234b1431d7ba282908229ba8217bdd1188a85fcc23800`
- Scientific canonical config SHA-256:
  `603629a1eab5bc0c68af70fd160904e1dda30c388a5ec4fae312585e46202c9d`
- Protocol: `research/coordinate_invariance/SWITCH_C2_PREREGISTRATION.md`
- AutoDL handoff:
  `research/coordinate_invariance/AUTODL_SWITCH_C2_RUNBOOK_ZH.md`
- Local project suite: **124 passed**
- Scientific status: **Not run**

The implementation now includes a hash-verified paper-final loader, official
versus audited identity smoke, resumable all-500 prompt assignment, immutable
16/32 calibration/test separation, gradient-aligned four-dimensional probes,
L1/L3/L4 and V1/V3/V8/semantic/V32/V64 metrics, whitening, exact V8 scalar
retuning, prompt bootstrap, free-rollout consequence checks, and implementation-
bound journals. A detached DynamicCache prefix snapshot was checked for
independent restoration, and the full pipeline passed a differentiable small-
model end-to-end test.

No Qwen3-8B or SWITCH weight has been downloaded and no checkpoint effect has
been observed. A high-memory cloud GPU is now authorized for C2 only. The
frozen runner requires at least 78 GiB visible VRAM; H20 96 GB is preferred.
This is inference/autodiff measurement, not training. C2 failure stops FCTR;
C2 success authorizes only a matched-compute estimator before any training.
