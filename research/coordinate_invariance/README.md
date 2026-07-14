# Coordinate-Invariance Research Harness

This package is the evidence-first substrate for the reparameterization audit and FCTR direction. It is intentionally separate from the legacy LatentGRPO implementation.

## Current contracts

- exact affine and analytic nonlinear coordinate charts;
- round-trip, Jacobian, and condition-number diagnostics;
- coordinate-defined nearest-neighbor, noise, and interpolation operations;
- categorical and multi-horizon functional divergence;
- matched-RNG continuation comparison without leaking global RNG state;
- integrity-checked latent trace persistence and replay;
- explicit model-call, token, rollout, backward, and wall-time accounting;
- a deterministic CPU toy gate with positive and negative controls.
- an exact local FCTR solver with covector/metric/tangent transport controls;
- a frozen FCTR revival contract that separates local Coconut integration from
  a later long-continuation SWITCH scientific gate.

## Run the engineering gate

```powershell
python -m research.coordinate_invariance.toy_gate `
  --output artifacts/coordinate_invariance/toy_contract_v1.json
```

The toy gate is not scientific evidence. A pass only shows that the harness can preserve behavior under a no-op chart, retain Euclidean geometry under an orthogonal control, and detect mathematically expected coordinate dependence under anisotropic or nonlinear charts.

The optimizer-level solver gate is similarly limited:

```powershell
python -m research.coordinate_invariance.fctr_toy_gate `
  --config research/coordinate_invariance/configs/fctr_toy_gate_v1.json `
  --output artifacts/coordinate_invariance/fctr_toy_gate_v1.json
```

Its derivation and strict prior-art boundary are recorded in
`FCTR_DERIVATION.md`; the real-model execution boundary is frozen in
`FCTR_REVIVAL_PREREGISTRATION.md`.

The local real-checkpoint differentiable integration gate is:

```powershell
python -m research.coordinate_invariance.fctr_coconut_smoke `
  --config research/coordinate_invariance/configs/fctr_coconut_smoke_v1c.json `
  --output artifacts/coordinate_invariance/fctr_coconut_smoke_v1c.json
```

C1c passed all frozen transport and numerical controls on the public GPT-2
Coconut checkpoint. This is still an integration result, not method evidence:
the checkpoint has only a trivial visible continuation, so the next scientific
gate must use the pinned SWITCH long-continuation architecture.

The paper-final SWITCH gate is frozen in `SWITCH_C2_PREREGISTRATION.md`. Its
checkpoint identity, resumable 500-prompt eligibility scan, calibration/test
separation, exact-KL scalar-retuning control, and held-out decision rules are
implemented by the `switch_checkpoint_identity_smoke`,
`switch_c2_eligibility_scan`, and `switch_c2_scientific_gate` modules. The
AutoDL handoff and one-command runner are in `AUTODL_SWITCH_C2_RUNBOOK_ZH.md`
and `run_switch_c2_autodl.sh`. No SWITCH checkpoint effect has been observed
locally; these files are a pre-execution contract.

## Evidence boundary

Do not cite the toy result as evidence that a trained latent reasoning model fails. Scientific evidence begins only after an exact chart is inserted around a real latent interface and the following are verified:

1. native and charted no-op logits are equal within a preregistered precision tolerance;
2. coordinate sensitivity appears under moderate, well-conditioned charts;
3. the sensitivity changes continuation behavior or a training decision;
4. the effect survives precision, whitening, ordinary-KL, and retuning controls;
5. a functional correction reduces the effect under matched compute;
6. the conclusion reproduces in a second latent representation.
