# Observable-Marginal Policy Audit

This directory archives the independent ChatGPT Pro report and subjects its
primary recommendation, Observable-Marginal Policy Improvement (OMPI), to a
zero-GPU method-identity gate.

## Reproduce

```powershell
_research_env\Scripts\python.exe -m research.observable_marginal_policy.exact_identity_gate
_research_env\Scripts\python.exe -m pytest -q tests\test_observable_marginal_policy_gate.py
```

## Files

- `CHATGPT_PRO_REPORT_2026-07-17.md`: unmodified external report supplied by the
  user.
- `M0_CLAIM_CONTRACT.md`: pre-registered identity, scope, and identifiability
  gates.
- `exact_identity_gate.py`: dependency-free executable counterexample.
- `EXACT_GATE.md`: frozen result and interpretation.
- `INDEPENDENT_AUDIT.md`: literature collision and reviewer assessment.
- `DECISION.md`: current authorization decision.

This audit does not modify the LatentGRPO implementation and does not treat a
toy identity check as model-performance evidence.

