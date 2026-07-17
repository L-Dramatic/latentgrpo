# Policy-Conditional Mixture Closure Audit

This directory audits PCMC, the backup proposed by the external ChatGPT Pro
report after OMPI-R failed its method-identity gate.

## Reproduce

```powershell
_research_env\Scripts\python.exe -m research.policy_conditional_mixture_closure.sequence_gate
_research_env\Scripts\python.exe -m pytest -q tests\test_policy_conditional_mixture_closure_gate.py
```

The current status is `KILL_PCMC_GATE_A0`: the complete frozen A0 passed on
Latent-GRPO but failed on SofT-GRPO by orders of magnitude. A1, Gate B, and
training are not authorized.

## Frozen checkpoint gate

- `CHECKPOINT_PREREGISTRATION.md` freezes the scientific estimand, split,
  interventions, thresholds, and KILL rules.
- `checkpoint_gate_runner.py` implements asset audit, GPU engineering preflight,
  and crash-safe A0 collection.
- `gate_a_analysis.py` is the only A0/A1 decision path.
- `ops/` contains the persistent managed runner and independent watchdog.

The managed scripts remain as reproducibility infrastructure. The completed A0
does not authorize rerunning with changed thresholds or a Latent-only claim.
