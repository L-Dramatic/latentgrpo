# Score-Squashed Gumbel Experiment Log

## 2026-07-14: support/concentration gate v1

### Frozen question

Can the fixed score squash preserve standard Gumbel Top-K support, maintain a
valid exact likelihood, and match the released hard-clipped sampler's mixture
concentration without widespread saturation?

### Protocol

The preregistered configuration crosses three logit concentrations and five
seeds at `V=256`, `K=10`, 16,384 paired draws per scenario, and the inherited
fixed bounds `[-1.5,3.0]`. Its SHA-256 is
`c62139fb723c6263f13ad5d12455a9742e4bf251341967d44caa384cf6cf1e51`.
Seven checks are conjunctive and unchanged after the run.

### Result

**Status: fail.** Six checks passed and the mixture-matching check failed.

- SSG/unbounded ordered-support mismatch: exactly `0`.
- Exact ratio mean error: at most `0.00918`.
- Exact score mean norm: at most `0.0589`.
- Hard clipping changed ordered support in `93.64%-99.98%` of paired draws.
- At least `18.58%` of hard-clipped selected noises occupied the upper atom.
- SSG saturation fraction: at most `0.101%`.
- SSG support-inclusion entropy gain: `0.540` nats on average and positive in
  every scenario.
- Hard-clip/SSG mixture entropy gap: up to `0.315`, above the frozen `0.15`.
- Hard-clip/SSG 95th-percentile maximum-weight gap: up to `0.414`, above the
  frozen `0.08`.

The mismatch is largest for diffuse logits: hard clipping produces much more
uniform mixtures than the fixed score squash. This defeats the preregistered
claim that the construction separates support preservation from comparable
mixture-range control.

### Decision

Stop SSG-PO. Do not tune the squash bounds or temperature after this result and
do not proceed to checkpoint or training experiments. Preserve the exact
density, common-support theorem, support-diversity benefit, and failed
mixture-control hypothesis in the broader policy-contract audit.

Artifact: `artifacts/score_squashed_gumbel/support_concentration_gate_v1.json`.
