# Independent Audit of the Next-Direction Red-Team Report

Audit date: 2026-07-18

External artifact SHA-256:
`8DFF3F2165DF385162D5CAE6E0E1BE399D3FA76092C0A0C7E2EAE8796359410D`

Frozen repository commit:
`6f32fc37527f923e064927c8930c2c9a3d9f64a2`

## Verdict

Accept the report's portfolio-level conclusion:

> `NO-GO: there is no sufficiently strong new candidate in this portfolio.`

Strengthen its Top-1 decision from "allow a sacrificial CPU Gate 0" to:

> `KILL PTPU before implementation.`

The reason is method identity, not weak expected performance. PTPU does not
specify an estimator or identification result that survives subtraction of PUM
and standard off-policy evaluation (OPE). A synthetic experiment cannot repair
that failure.

## What the external report did well

- It read the frozen repository and preserved all prior terminal decisions.
- It searched a broad 2024-2026 neighborhood and rejected 23 of 26 candidates.
- It named the strongest composed baselines and direct-rollout oracle.
- It exposed token-support, moving-policy, foreign-prefix, and long-horizon
  identifiability risks.
- It refused to authorize training or large-GPU expansion.

These are useful research-audit assets. The error is the final exception that
still authorizes PTPU's CPU gate.

## Exact PUM collision

[PUM](https://arxiv.org/abs/2606.07190) already defines, for student policy
`pi_s`, the prefix-conditioned solve probability

```text
q_s(x, p) = Pr_{c ~ pi_s(. | x, p)}[verifier(x, p + c) = 1].
```

This is the same object as PTPU's headline estimand `V^pi(h)`, with notation
changed from `(x, p, pi_s)` to `(h, pi)`. PUM then forms a gain profile across
policies, explicitly separates shared intrinsic utility from policy-dependent
variation, and reports capability-aware student weighting for downstream
policies of different strengths. The [official PUM project
page](https://zhiqix.github.io/pum-project-page/) also lists policy dependence
as a dedicated analysis.

Therefore, neither "prefix value depends on the continuation policy" nor
capability-aware aggregation is an unoccupied contribution.

## Exact OPE reduction

For a fixed prefix `h`, define an episodic MDP whose initial state is `h`, whose
action is the next token, whose deterministic transition appends that token,
and whose only nonzero reward is the terminal verifier output. Then

```text
V^pi(h) = E_{c ~ pi(. | h)}[R(h + c)]
```

is exactly the ordinary value of target policy `pi` from initial state `h`.
Estimating it from trajectories logged by policies `mu_k` is standard
multi-logger conditional OPE. [Sequential doubly robust
evaluation](https://proceedings.mlr.press/v48/jiang16.html) already targets a
new policy's value from another policy's data, and
[MAGIC](https://proceedings.mlr.press/v48/thomasa16.html) already mixes model
and importance-sampling estimates to reduce error.

The report's only proposed increment is

```text
S(h, pi) = (ESS, max weight, distribution distance, coverage),
```

followed by abstention. This is a diagnostic contract, not a new estimator. The
report provides no new identifying assumption, partial-identification bound,
multi-logger weighting rule, confidence construction, or theorem that exploits
an LLM-specific structure. The shown DR equation is a generic baseline.

The collision is reinforced by:

- [GenAC](https://arxiv.org/abs/2604.10701), which conditions a critic on the
  current actor to track policy changes;
- [`V_0`](https://arxiv.org/abs/2602.03584), which treats dynamic policy
  capability as explicit context for any-policy value prediction;
- [MinPRO](https://arxiv.org/abs/2601.22718), which directly studies the
  instability of cumulative prefix importance ratios in off-policy LLM RL;
- [OPE beyond overlap](https://proceedings.mlr.press/v235/khan24b.html), which
  provides actual partial-identification bounds under explicit smoothness
  assumptions rather than an ESS-based abstention label.

## Why the proposed CPU Gate 0 has no decision value

1. The finite DAG is constructed to contain rank inversions, so detecting them
   verifies the generator, not a natural LLM phenomenon.
2. The report repeatedly compares a "candidate estimator" but never defines
   one beyond DR/MAGIC/FQE plus support diagnostics.
3. Bias, interval coverage, and unsupported-state abstention are standard OPE
   properties. Passing them would not establish an AAAI contribution.
4. A 15% RMSE improvement over generic baselines is not preregistrable until a
   fixed candidate formula and a non-adversarial data-generating family exist.
5. Real utility remains governed by the direct target-policy oracle. The CPU
   gate cannot show that cross-policy prefix selection has a nontrivial natural
   ceiling.

The correct pre-compute gate is exact method subtraction. PTPU fails it.

## Practical dominance of direct rollout

PTPU requires access to target-policy token likelihoods to compute importance
ratios. In the proposed setting the target model and deterministic verifier are
available, so direct target continuations are also available. K=2 or K=4 fresh
rollouts are unbiased, avoid long-horizon weight products, and naturally handle
the target policy's reachable-prefix distribution. Any OPE method would need to
beat this baseline after counting target-model forward calls, verifier calls,
and abstention coverage.

The proposed Qwen base, Qwen instruct, and distilled reasoning policies may
share a tokenizer, but shared tokenization does not imply usable trajectory
overlap. Long completions multiply modest token-level mismatch into low
effective sample size. This is an expected failure mode, not a new method gap.

## AAAI assessment after subtraction

| Dimension | External report | PTPU as method |
|---|---:|---:|
| Repository grounding | 9/10 | 9/10 |
| Literature breadth | 8/10 | 4/10 independent remainder |
| Problem importance | 8/10 | 8/10 |
| Method novelty | n/a | 2/10 |
| Identification depth | 7/10 audit | 3/10 method |
| Natural oracle evidence | 0/10 | 0/10 |
| AAAI main-track readiness | n/a | 2/10 |
| Next justified compute | n/a | 0 GPU-hours |

## Preserved value

- Keep PUM, policy-conditioned critics, standard OPE, and K=2/4 fresh rollout
  as mandatory baselines for future prefix-value claims.
- Keep the report's 26-candidate collision table as a negative search map.
- Keep cross-policy rank reversal only as an already anticipated analysis
  question; do not promote it to a method claim.
- Do not rename PTPU, combine it with LRPE/APVC, or lower its novelty bar.
