# FCTR Revival Contract

Frozen on 2026-07-15 after the nearest-neighbor line failed and before any
optimizer-level real-model effect was inspected.

## Prior evidence that remains binding

The disjoint Coconut holdout changed Euclidean nearest neighbors on 16 of 100
examples, and changed choices had large continuation effects. Its bootstrap
lower bound nevertheless missed the frozen frequency threshold. That line is
stopped permanently: no stronger chart, looser threshold, or second
architecture will be used to relabel it as a pass.

The corrected BCG pilot also showed that the apparent H1 blind spot on the
available Coconut checkpoint was largely a fixed `###` prefix. Cumulative H2
and H3 predicted the terminated continuation well. Coconut therefore remains
an engineering target, not the scientific model that can revive FCTR.

## New question

The only permitted revival question is optimizer-level:

> Does a coordinate-defined latent update produce materially different
> observable behavior under an exact, moderate reparameterization, and does a
> longer-horizon functional metric add operational value beyond H1/H2/H3,
> FishBack, whitening, and scalar retuning?

Showing that Euclidean gradients are not affine invariant is a mathematical
positive control, not a paper result. Showing that any output Fisher is
invariant is also insufficient because next-token FishBack already provides
that correction.

## C0: deterministic solver gate

- Config: `configs/fctr_toy_gate_v1.json`
- Canonical config SHA-256:
  `4fbab76baf0224885b7aa86a59510771eca74e268ea3b050df04b95434eb337d`
- Identity and orthogonal charts are negative controls.
- Condition-number 4 and 12 affine charts are Euclidean positive controls.
- Every FCTR step must transport with relative L2 error at most `1e-10`.
- Every quadratic budget must close within `1e-12`.
- The strongest anisotropic chart must change the coordinate-Euclidean update
  by relative L2 at least `0.1`.

A pass validates only the implementation and derivation.

## C1: local Coconut differentiable integration gate

The public GPT-2 Coconut checkpoint will be used to establish that gradients,
short-prefix Fisher metrics, and FCTR steps can be measured through an exact
hidden-state feedback intervention without manufacturing an effect.

Frozen implementation record, written before running C1:

- Config: `configs/fctr_coconut_smoke_v1.json`
- Config file SHA-256:
  `22c8798d139a3097b40c6e481ac39c6489c80f28b5a163ef660ef7b37e2a63c9`
- Canonical config SHA-256:
  `2285801576ed776be85d5c386c14e9d743deb115a38e557725443987b3a8fa70`
- Base model revision:
  `openai-community/gpt2@607a30d783dfa663caf39e06633721c8d4cfcd7e`
- Coconut source commit:
  `27273cb8cca4bb763c041a63b036d0c3b7cbbb48`
- Checkpoint SHA-256:
  `f7a9b5fda7c5c2afa972aaa22ac9aef13d6f083e202fe8a09649d810e3593213`
- Data SHA-256:
  `d6cdd4c3c48de84873fbfb0f0005d30e9c5527c430feb206608749aa48dfd6e5`
- Example index `0`, latent pass `3` of six, target template `### {answer}`
  followed by EOS, and a three-token H1/H2/H3 horizon.
- A seeded three-dimensional orthonormal update subspace is measured inside
  the 768-dimensional hidden state. Charts are identity, seeded orthogonal,
  and seeded affine maps with condition numbers 4 and 12.
- Trust budget is `1e-4`; the coordinate-Euclidean RMS step is `1e-2`.
- Relative-error ceilings are `1e-3` for Jacobian and gradient transport and
  `2e-3` for metric and FCTR transport. Finite-difference relative error must
  be at most `0.1`; identity and orthogonal Euclidean controls at most `1e-3`;
  the strongest anisotropic Euclidean mismatch at least `0.1`.

The first execution of that frozen config ended before any Jacobian or effect
was produced. PyTorch reported that the Transformers-selected efficient SDPA
backend does not implement the second derivative required by
`torch.autograd.functional.jvp`. No artifact was written and no gate value was
observed.

The permitted differentiability-only repair is frozen as C1b:

- Config: `configs/fctr_coconut_smoke_v1b.json`
- Config file SHA-256:
  `6d7dad495f5be83edc07dab3a6fa0f5cba4e64f9bba737a7873b002e9d309075`
- Canonical config SHA-256:
  `ee8d5701f6e2253d74efb8ec0a57de5ecd3658148db8339d1cf1f3d78ed7b0be`
- The experiment name gains suffix `v1b` and the sole runtime change is
  `attention_implementation: eager`. Eager attention computes the same
  attention equations using differentiable primitive operations.
- Model, checkpoint, data, example, target, subspace, charts, seeds, update
  scales, and every threshold are byte-for-byte unchanged from C1a.

C1b completed the native Jacobian and reached the first chart comparison, then
stopped before writing an artifact because the independently created chart
matrix was on CPU while the measured Jacobian was on CUDA. No gate report was
available. C1c freezes the permitted device-placement repair:

- Config: `configs/fctr_coconut_smoke_v1c.json`
- Config file SHA-256:
  `45fe953d4f4be07338e9b5f198c79cb83d7adff1dfa22dcabdc867ff0fa96fda`
- Canonical config SHA-256:
  `f3ce1bacda69040d8190d6e55d3e57bcb159cd2ca2a5610b5e700566333e8245`
- Runner script SHA-256:
  `389f9ba43b7f6a4219e5402495959d8f2d1ac1350b79705382f1c518dc9b135c`
- The only code change moves the native Jacobian to CPU float64 before its
  analytic matrix product. The only config change is the `v1c` experiment
  suffix. All model, data, chart, update, seed, and threshold fields remain
  unchanged.

Required controls:

1. Native and no-op chart objectives and logits agree within the frozen
   float32 tolerance.
2. Autograd and finite-difference directional derivatives agree.
3. Native and charted objective gradients obey covector transport.
4. H1, H2, and H3 pullback metrics obey metric transport.
5. FCTR steps transport while coordinate-Euclidean steps retain the expected
   anisotropic positive-control mismatch.
6. All answer tokens are valid pre-EOS tokens; no post-EOS term is used.

This stage may repair differentiability, cache handling, and numerical
precision only. It cannot authorize FCTR training because the checkpoint lacks
a nontrivial visible continuation.

## C2: SWITCH scientific gate

SWITCH is the preferred second representation because its released Qwen3-8B
checkpoint uses genuine hidden-state recurrence inside visible long-form math
reasoning. Source and model revisions are pinned in `SOURCE_MANIFEST.json`.

Before downloading weights or renting a GPU, C1 must pass and the C2 config
must freeze:

- prompt selection and dataset revision;
- target latent transition and minimum latent dwell;
- exact chart matrices and condition numbers;
- H1/H2/H3, semantic-prefix, and longer-horizon estimators;
- gradient subspace or Fisher-vector-product solver;
- proposal step grid, independent held-out utility, bootstrap unit, and all
  thresholds;
- token, model-call, backward-call, wall-time, and peak-memory accounting.

The full method gate requires all of the following:

1. No-op, orthogonal, precision, and transport controls pass.
2. Moderate charts alter a real update decision after scalar learning-rate or
   trust-radius retuning, not only its coordinate norm.
3. The change affects held-out continuation utility or correctness.
4. H1/H2/H3, semantic-prefix KL, FishBack, whitening, and ordinary sequence KL
   do not already remove or predict the effect.
5. A longer-horizon functional step reduces chart sensitivity under matched
   model calls and improves or preserves held-out utility.

Failure of item 2 or 3 stops the method paper. Failure of item 4 means the
simpler baseline wins and FCTR must not be claimed. A pass authorizes only a
matched-compute V32 estimator. That estimator must preserve the C2 advantage
before a small training pilot and later cross-architecture replication.

## Compute decision

No cloud GPU was authorized for C0 or C1. C1c, the SWITCH source preflight, the
source-equivalent adapter tests, the identity-smoke contract, and the complete
C2 protocol are now frozen. A cloud rental is therefore authorized for C2
only, not for training. The runner requires at least 78 GiB visible VRAM;
H20 96 GB is preferred, with A100/H100 80 GB as alternatives.
