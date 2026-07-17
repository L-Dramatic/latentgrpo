# Policy-Conditional Mixture Closure Audit

This directory audits PCMC, the backup proposed by the external ChatGPT Pro
report after OMPI-R failed its method-identity gate.

## Reproduce

```powershell
_research_env\Scripts\python.exe -m research.policy_conditional_mixture_closure.sequence_gate
_research_env\Scripts\python.exe -m pytest -q tests\test_policy_conditional_mixture_closure_gate.py
```

The current status is `HOLD`: CPU protocol preparation is allowed; checkpoint
inference and training are not yet authorized.

