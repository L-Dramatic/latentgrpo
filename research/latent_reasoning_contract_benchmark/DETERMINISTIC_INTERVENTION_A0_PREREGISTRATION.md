# Deterministic Intervention A0 Preregistration

Frozen: 2026-07-18, after checkpoint readiness and before any checkpoint
intervention effect was produced.

## Question

Can the LRC-Bench runtime apply source-faithful, equal-depth state
interventions to both Coconut and CODI, reconstruct each native run, and detect
checkpoint-level outcome sensitivity without changing the visible target?

This is an instrumentation and calibration gate. A pass is not novel evidence
that latent states are causal; that broad result already has direct prior art.
It only authorizes the untouched confirmation and stochastic-contract phases.

## Frozen Data

The source is `openai/gsm8k`, configuration `main`, test split, immutable
revision `740312add88f781978c0658806c59bc2815b9866`. Selection sorts
`SHA256(salt + NUL + question)` without reading labels or model outputs. The
first 32 selected rows are calibration and the next 32 are an untouched
confirmation set.

Prompt manifest SHA-256:
`0ec53ccd19acbf5d2b1a4bd8fc0f512bc741c2d6b92c223712f7562c4dd2d1e7`.

The two methods share questions and numeric answers but retain their native
target formats:

- Coconut: `### {answer}` followed by EOS;
- CODI: `The answer is: {answer}` followed by EOS.

Raw NLL or accuracy is never compared across methods. All effects are paired
within one method, checkpoint, prompt, and target sequence.

## Frozen Conditions

- `native_live`: released recurrence computes all six executed states.
- `native_replay`: a fresh cache receives the six captured native states; this
  is a mandatory reconstruction control.
- `repeat_first`: the first native state is executed six times.
- `reverse_steps`: the same six native states and norms are executed in reverse
  order.
- `zero_matched_depth`: six zero states preserve latent depth.
- `norm_matched_random`: six deterministic isotropic states match the native
  per-step L2 norms.
- `no_latent`: the source-compatible zero-depth path, analyzed only as a simple
  depth baseline because its hidden-token count differs.

Every condition uses the same question and teacher-forced target tokens. The
four equal-depth effects preserve six executed latent positions. No answer is
decoded, and no generated-token budget can confound the primary outcome.

## Outcomes And Controls

Primary outcome: target-token mean negative log likelihood. Secondary outcome:
mean teacher-forced KL from the native predictive distribution. Native-state
norm, step change, intervention displacement, token counts, logits, runtime,
and peak allocated CUDA memory are recorded.

The run is `HOLD` if any output is non-finite, source/checkpoint/prompt hashes
drift, target tokens differ, an equal-depth condition does not execute exactly
six states, native replay differs by more than `1e-6` NLL or `1e-4` in any
scored logit, or peak allocated CUDA memory exceeds 2500 MiB.

## Calibration Decision

For each method independently, at least one of the four equal-depth effects
must meet all of:

1. absolute mean NLL delta at least `0.01` nats/token;
2. mean teacher-forced KL at least `1e-4`;
3. absolute NLL delta above `1e-4` on at least 75% of prompts.

Controls passing with signal in both methods is `PASS_A0_SIGNAL`. Controls
passing without that signal is `KILL_DETERMINISTIC_EFFECT_BRANCH`; it does not
erase the separately motivated stochastic policy-contract branch. Calibration
cannot alter prompts, target formats, conditions, thresholds, or aggregation.

No training or cloud GPU is authorized by this gate.
