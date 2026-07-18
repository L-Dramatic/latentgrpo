# Gate 0 and Gate A Preregistration

Status: frozen design skeleton after Gate -1. Exact prompt hashes are frozen
only after CODI checkpoint acquisition and before any intervention effect is
read.

## Gate 0: source-native fixture replay

No training and no task-level result inspection.

For every method, construct one deterministic synthetic state and one real
checkpoint state. Require:

1. exact source revision and checkpoint hash;
2. native prompt/tokenizer range validity;
3. recurrence reconstruction within dtype tolerance;
4. identical no-effect replay under fixed seed or deterministic recurrence;
5. explicit separation of executed state, stored state, proxy/control state,
   and scored state;
6. source-equivalence against an independently implemented fixture.

Any failed control is engineering `HOLD`, not scientific evidence. Three
failed repair attempts on the same source contract exclude that method.

## Gate A: checkpoint intervention pilot

Use 64 hash-selected native-task prompts per method, split 32 calibration and
32 untouched confirmation prompts. Keep the native number of latent steps.
For stochastic methods use 16 paired seeds per prompt; deterministic methods
use one replay per prompt. Report paired prompt-cluster uncertainty.

### Interventions

- `native`: exact released recurrence;
- `no_latent`: bypass latent recurrence with the source-compatible shortest
  path;
- `repeat_first`: repeat the first latent state for all latent positions;
- `permute_steps`: reverse or hash-permute recurrent states where defined;
- `matched_random`: norm- and layer-matched random state control;
- `contract_specific`: alter stored/scored state while preserving executed
  state, or vice versa, only when the source exposes that distinction.

An intervention that changes token count, answer decoding, prompt text, or
final-answer budget relative to its paired native run is invalid.

### Outcomes

Primary outcome: paired change in native reference-answer negative log
likelihood, standardized by the prompt-level native distribution within each
method. Secondary outcomes: exact answer accuracy, output KL, stop time,
support changes, clipping decisions, and gradient direction where applicable.

### Strong simple baselines

- no-latent ablation;
- latent depth;
- hidden-state norm and step-to-step change;
- entropy, maximum mixture weight, and effective support for stochastic methods;
- input-embedding and last-input-token controls;
- Four Axioms metrics where computationally compatible;
- method identity and native accuracy without contract features.

## Promotion to Gate B

All source controls must pass. In addition, confirmation data must show both:

1. at least two distinct operational contract failures or nontrivial contract
   sensitivities across at least two mechanism families; and
2. cross-validated contract features improve held-out prediction of absolute
   standardized intervention effect by at least `0.05` in Spearman correlation
   and `5%` relative MAE over the strongest simple baseline.

Both thresholds must hold under prompt-cluster bootstrap with a 95% lower bound
above zero. Results from calibration choose no threshold or intervention.

## KILL at Gate A

Kill the benchmark-paper thesis if contract features do not beat simple
baselines, if only Latent-GRPO is positive, if effects disappear on untouched
confirmation prompts, or if fewer than three methods pass source-native replay.
Preserve the source manifest, fixtures, and negative results.

GPU is authorized only after Gate 0 passes for all four methods. Matched
training remains unauthorized through Gate A.
