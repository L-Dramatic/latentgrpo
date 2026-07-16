# P1 Source and Compute Feasibility Audit

**Audit date:** 2026-07-16  
**Scope:** stochastic natural-update phenomenon gate for Horizon Sufficiency  
**Decision to redraft P1:** `GO-REDRAFT`  
**Decision to freeze or execute P1:** `NO-GO-EXECUTION`  
**GPU decision:** `NO-GO`  

This is a post-freeze feasibility audit, not part of the immutable P0 contract.
It does not change the frozen SWITCH C2 experiment and does not authorize a
checkpoint run, held-out measurement, certificate implementation, or training.

## 1. Binding recommendation

Do **not** make Qwen3-8B SWITCH the first P1 platform. Also do not promote the
cheapest GPT-2 Coconut checkpoint merely because it is convenient: its observed
continuation is normally a fixed `###` delimiter, a short answer, and EOS, so it
does not supply the nontrivial semantic tail needed to test a Horizon Gap.

The lowest-risk order after the independent P1 red-team is:

1. use Coconut only to validate natural-update extraction, same-history replay,
   and expected-Fisher accounting on a real but short-support checkpoint;
2. make the already local official Latent-GRPO 1B checkpoint the preferred P1
   candidate **only if** an early-action deterministic-recursion adapter and
   frozen tail/trace gates pass;
3. test source exploration actions first, followed only on a clean frozen pass
   by an objective-derived source-surrogate hidden-state counterfactual;
4. treat SofT-GRPO 1.5B only as nearby same-mechanism robustness;
5. promote a qualified CODI checkpoint ahead of SofT for the first low-compute
   mechanism-distinct scientific replication; and
6. reserve SWITCH for a stronger hidden-recurrence replication after the
   small-model phenomenon survives.

The preferred P1 platform sequence is therefore:

| Priority | Platform and update family | Role | Current decision |
|---:|---|---|---|
| 0 | GPT-2 `Coconut-OPT` | real-checkpoint engineering and estimator control | `GO-PREFLIGHT`, but `NO-GO` as P1 primary on the current checkpoint |
| 1 | official Latent-GRPO 1B `LatentPolicy-ACTION` + `LatentPolicy-SOURCE-GRAD` | preferred low-compute P1 candidate | `CONDITIONAL`; recursive adapter, token, trace, tail, and compute gates missing |
| 2 | public CODI GPT-2 `CODI-OPT` | preferred low-compute cross-mechanism candidate if it has adequate visible support | `CONDITIONAL`; checkpoint and adapter missing |
| 3 | official SofT-GRPO 1.5B action/objective families | nearby same-mechanism robustness check | `CONDITIONAL`; P1 adapter missing and does not count as mechanism-distinct by itself |
| 4 | SWITCH Qwen3-8B stochastic first-segment replay | strongest hidden-recurrence replication | `DEFER` |

This table is scientific priority, not engineering order. Engineering remains
cheapest-first: Coconut, Latent-GRPO, SofT-GRPO, CODI, SWITCH. The earlier
delimiter-confounded GPT-2 BCG pilot is not positive evidence. Its
24 reference continuations all ended after `###`, one short answer, and EOS;
H2/H3 already recovered the apparent tail. Therefore the current Coconut
checkpoint cannot count toward the two-family P1 claim. It remains useful for
engineering controls, and a different Coconut checkpoint could be reconsidered
only after a frozen semantic-tail eligibility gate passes.

## 2. Why the frozen SWITCH C2 cannot be relabeled

The frozen runner is explicitly greedy. `SwitchAuditRunner.run` has no
temperature or generator argument and uses `argmax` at the initial decision,
after a latent block, and after ordinary visible tokens
(`real_models/switch.py:622-632, 669-670, 769-770, 789-790`). Its replay plan is
built from that factual continuation (`real_models/switch.py:329-368`).

Useful numerical machinery already exists:

- `SwitchDifferentiableReplay` differentiates through the first intervened
  latent input, all remaining factual latent steps, and a fixed teacher-forced
  visible history (`real_models/switch.py:371-603`);
- the four-dimensional JVP loop is implemented in
  `switch_c2_scientific_gate.py:149-189`;
- `prefix_geometry` constructs a single-history categorical pullback Fisher in
  `switch_c2_geometry.py:56-118`; and
- `JointContinuationEvaluator` already defines the correct sample/force
  same-history KL contract in `joint_kl.py:15-32, 94-112, 221-299`.

What does not exist is a SWITCH policy implementing that sample/force contract,
an expected-Fisher aggregator over independent histories, or a valid treatment
of stochastic EOS and future `<swi>` events. The old implementation hashes also
bind `real_models/switch.py` (`switch_c2_scientific_gate.py:69-79`), so P1 must
use new modules and new hashes rather than mutate C2.

## 3. Smallest defensible SWITCH estimand

If SWITCH is used later, the first implementable target is not the unrestricted
deployment process. It is a stopped process:

> continuation risk after the first latent block, conditional on the factual
> first-block dwell, observed until a fixed visible horizon or the first EOS or
> new `<swi>` boundary, whichever occurs first.

EOS and a new `<swi>` must be recorded as absorbing stopping events. Rejecting
or resampling those paths would estimate a selected conditional distribution
and bias the reference-history expectation. Continuing through a new `<swi>`
would instead require differentiable variable-control-flow replay of future
latent blocks and is currently `NO-GO` for the first P1.

This stopped first-visible-segment target must be named explicitly. It cannot be
marketed as a guarantee for the full recursive deployment trajectory.

## 4. Natural update families

### 4.1 Engineering family: Coconut-OPT

The trained public checkpoint is already present at
`_models/gpt2-gsm8k-coconut/checkpoint_33` and passes the repository's pinned
SHA-256 contract. The adapter already supports latent capture/override with
gradients and stochastic sample/force replay
(`real_models/coconut.py:84-168, 240-426, 428-600`).
The pinned upstream training path uses AdamW (`_external/coconut/run.py:228-236`)
and applies its supervised loss backward and optimizer step
(`_external/coconut/run.py:363-370`); the curriculum constructs the latent-token
and supervised-label pattern in `_external/coconut/dataset.py:230-299`.

For a calibration-only optimizer batch `B`, define a real micro-update using
the official curriculum loss and AdamW rule:

\[
\theta^+ = \operatorname{AdamW}(\theta, \nabla_\theta L_B),
\qquad
v(x,t)=z_{\theta^+}(x,t)-z_\theta(x,t).
\]

On held-out prompts, both activations are injected back into the same frozen
reference model `theta` for the P1 risk comparison. Behavior of the fully
updated parameter model `theta+` belongs to the later parameter bridge, not P1.
Optimizer batches, held-out prompts, and Fisher/evaluation histories must be
disjoint. Multiple prompt effects induced by one optimizer step remain one
optimizer-trajectory cluster for splitting and bootstrap.

This is a valid natural update, but the present checkpoint's visible suffix is
too short to answer the P1 scientific question. Use it to validate update
extraction, support diagnostics, exact enumeration on a small model, and
compute accounting. Do not count it as a primary natural family.

### 4.2 Preferred P1 platform: official Latent-GRPO 1B

The official Latent-GRPO source contains top-k/Gumbel latent-policy action
records. A candidate exploration update begins from the source-proposed sampled
mixture/proxy pair and the single official no-noise deterministic reference at
the same first latent context. It becomes an executed soft action only when its
proxy first survives the generic source stop check and is then nonterminal.
Conditional on no generic stop, proxy 524 invokes the scheduler's hard one-hot
transition, so the proposed soft mixture is recorded but not consumed. This is
algorithmic exploration, not an isotropic stress direction.

The released GSM8K configuration allows 128 response tokens and switches from
latent to explicit reasoning at the declared latent-end token
(`_external/official_latent_grpo/Latent-GRPO-gsm8k-llama3.sh:15, 43-53`). The
pinned 1B checkpoint already passes native-load, finite-logit, and repeatability
smokes with about 2.9 GiB peak allocated memory on the local RTX 4060. This does
not establish differentiable-replay memory, but makes it the best current
low-compute candidate.

The 128-token limit is total response length, not 128 visible tokens after
latent closure. The P1 support scan must count latent steps, the end transition,
formatting, visible tokens, and EOS inside that same cap; extending it would be
a separately named analysis configuration rather than released evaluation.

Source replay is pinned to commit
`c0994fb781a2d180662bb522d8ff3e8638dcf56d`. The noisy sampler contract is not
generic "top-10 Gumbel": it uses raw full-softmax probabilities, a top-p `0.95`
eligibility prefix with at least ten tokens, a full-vocabulary Gumbel draw
clamped to `[-1.5,3]`, the released one-sided shift and unit scale, perturbed
top-10 selection, and unit Gumbel-softmax temperature. The clean top-1 id gates
the noisy branch. Exact replay must use a homogeneous batch layout because the
source reads batch-wide and element-zero sampling state.

The red-team rejected the former final-action interface. The scientifically
defensible candidate starts at the first nonterminal deterministic soft action,
injects one natural action, and then recursively recomputes every later
deterministic mixture, proxy id, exit decision, and cache state. The last-action
variant is only a non-recursive specificity control.

The injected source action is joint and stateful. In the pinned source, every
proxy is appended and `check_finished()` is called before latent-state update
(`scheduler_output_processor_mixin.py:246-259`). That check applies total
response length, `ignore_eos`, sampling stop ids, request/tokenizer EOS ids,
tokenizer additional-stop ids, and stop strings even during latent mode
(`schedule_batch.py:682-725`). Therefore the exact execution order is:

1. append proxy and apply the frozen source stop semantics;
2. on a token/string/length stop, record an absorbing `LATENT_FINISH_*` endpoint,
   consume no soft mixture, and produce no visible continuation;
3. otherwise consume one-hot `E_524` and enter visible mode if the proxy is 524;
4. otherwise consume the continuous mixture and remain latent.

The freeze must bind all stop-id/string sets, `ignore_eos`, decoder state, and
confirm that 524 is neither a generic stop id nor a stop-string trigger under
the frozen decoded suffix. The same order applies to each later recursively generated proxy. A candidate whose injected proxy differs
from the reference, or terminates generically in latent mode, remains a
structural branch endpoint and cannot be silently treated as a smooth activation
delta.

Two constructions are proposed on this platform with one common deterministic
reference action:

- `LatentPolicy-ACTION`: a source-proposed noisy top-k/Gumbel joint action minus
  the official no-noise deterministic action; its soft mixture executes only
  for a nonterminal, non-stopping proxy; and
- `LatentPolicy-SOURCE-GRAD`: an advantage-weighted source-surrogate gradient
  pulled through the frozen LM head to its final-normalized input state, mapped
  back through the deterministic action rule. The released actor detaches the soft
  embedding, so this is explicitly an objective-derived activation
  counterfactual, not a pathwise answer gradient, head update, or optimizer
  step.

For source equivalence, that hidden state is specifically the final-RMSNorm
output directly supplied to the biasless tied head. The objective is the
negative first-update policy loss with old=current self-ratio one, one PPO
epoch, group size eight, the pinned first-mask/winner advantage branch,
token-mean aggregation, clip `0.2/0.2` and dual clip `3.0`,
`norm_adv_by_std_in_grpo=True`, `neg_adv_weight=1`,
`algorithm.use_kl_in_reward=False`, entropy zero, and no actor KL loss. The audited Gumbel surrogate
implementation is mandatory; its ordinary-token-logprob fallback is a hard
failure.

They must be reported separately and cannot be pooled into a larger apparent
sample. A later mechanism-distinct checkpoint remains mandatory even if both
pass.

It may become a primary family only if all of the following pass before P1 is
frozen:

1. the local sampler is source-equivalent to the pinned official implementation;
2. checkpoint/tokenizer identity is valid without silently resizing a missing
   or out-of-range compress token;
3. same-history stochastic sample/force continuation is implemented; and
4. the selected action distribution was genuinely part of the declared
   algorithm, with temperature, top-k, and RNG law frozen;
5. candidate latent closure naturally exits at high frozen coverage and the
   local interpretation survives a support/proxy/exit trace-stability gate; and
6. the primary temperature-only full-softmax visible analysis law and the
   exact extended-real top-k/top-p sensitivity both pass sample/force tests.

The tokenizer adds `<|compress_token|>` at id 128256 while the checkpoint has
valid embedding ids only through 128255. The ordinary frozen smoke prompt did
not emit it, but any P1 protocol that requires that id is `NO-GO`; silent
embedding resize is forbidden. Probability support is exactly model rows
`0..128255`, regardless of `len(tokenizer)`.

### 4.3 Replication and fallback families

The local official SofT-GRPO 1.5B checkpoint is the preferred nearby-policy
engineering robustness check because it has already passed native-load and
prompt-range controls without the Latent-GRPO tokenizer mismatch. It still
requires a source-equivalent P1 sample/force adapter and a semantic-tail support
gate. Its vocabulary-mixture mechanism and source lineage are too close to
Latent-GRPO to count as the mechanism-distinct replication required by P3.

Scientifically, a qualified `CODI-OPT` is preferred before SofT as the first
low-compute cross-mechanism candidate: an actual CE-plus-distillation optimizer
micro-step on a public CODI checkpoint. It may replace a failed family only
according to a source-only rule frozen before any P1 risk outcome. Two Coconut
curriculum stages may not be counted as two independent families.

### 4.4 Excluded primary families

- the three random orthogonal axes in the old SWITCH consequential basis;
- arbitrary Gaussian or isotropic tangent noise;
- coordinate transformations;
- the repository's custom LatentGRPO loss until its policy objective and KL
  semantics pass a separate source-validity audit; and
- post-hoc amplified, norm-searched, or outcome-selected updates.

These may be negative or stress controls only.

## 5. Static SWITCH compute audit

Let `L` be the factual first-block latent dwell. One 64-visible-token replay
uses approximately `L + 64` sequential single-token model forwards; an H8
objective replay uses `L + 8`. With the current four-dimensional subspace, one
exact single-history categorical GGN uses four H64 JVP replays.

Under the frozen C2 loops and assuming every candidate remains inside radius:

- each calibration prompt performs 170 H64-equivalent replays plus one H8
  replay; the first four prompts perform eight extra finite-difference replays;
- each test prompt performs 76 H64-equivalent replays plus one H8 replay and
  four or five free rollouts; and
- the 16/32 scientific block contains at most 5,184 H64 replays, of which only
  192 are the four JVPs used to build the factual-path GGN.

The eligibility stage also scans all 500 ordered MATH-500 prompts and permits up
to 1,024 generated tokens per prompt (`switch_c2_eligibility_scan.py:262-384`).

For `D` subspace dimensions, `R_F` construction histories, horizon `H`, and
latent dwell `L`, the expected-Fisher core costs approximately

\[
D R_F (L+H)
\]

token-level forwards per prompt, before natural-update labels, direct-MC
baselines, or UCB line search. At `D=4`, `R_F=8`, `H=64`, and `L>=4`, this is at
least 2,176 token-level forwards per prompt.

Replacing only the old factual GGN gives a misleadingly modest total-call
increase because most old calls remain greedy. If construction and consequence
labels both use `R` independent stochastic histories, a naive stochastic clone
of C2 is approximately `R` times the scientific-stage cost: roughly 8x, 16x, or
32x for `R=8,16,32`, before extra UCB calls.

The only plausible SWITCH P1 is a reduced protocol with approximately 8-12
natural update candidates and no old coordinate-chart search. Its leading
replay count is then approximately

\[
4R_F + mR_E,
\]

where `m` is the number of natural updates and `R_E` is a seed-disjoint label
history count.

## 6. Memory boundary

Sequential streaming should make history count primarily a time multiplier,
not a peak-memory multiplier. A valid implementation must accumulate only small
CPU float64 sufficient statistics and immediately release each history's logits,
Jacobians, graph, and KV cache.

The existing 78 GiB requirement is an execution floor, not a measured peak.
No Qwen3-8B JVP checkpoint measurement completed in the prior H20 attempts.
Therefore:

- sequential, one-history-at-a-time geometry on a 96 GB H20 is conditionally
  plausible;
- batched histories, retained `R x H x V x D` Jacobians, or multiple live KV
  caches are `NO-GO`; and
- no GPU-hour or rental-cost promise is defensible before an R=1/2/4 calibration
  microbenchmark measures wall time, peak VRAM, and fragmentation.

## 7. Sample-size boundary

The old 16 calibration / 32 test prompts are not automatically valid for P1.
Additional histories reduce within-prompt Monte Carlo noise but do not increase
the number of independent prompt clusters.

For the frozen mean Spearman-margin target `0.10` with a 95% lower bound of
`0.03`, a rough normal approximation at a true mean of `0.10` gives

\[
n \gtrsim \left(\frac{1.96\,\sigma_{\text{prompt}}}{0.07}\right)^2.
\]

This is approximately 32, 71, or 126 test prompts for prompt-level standard
deviations 0.20, 0.30, or 0.40. The variance is currently unknown. Sixteen
calibration prompts would also have to choose horizons, rollout counts, update
scales, baselines, and support thresholds, so inheriting that number would be
under-justified.

This audit initially suggested 32 calibration groups, but the v2 adapted-metric
contract supersedes that lower planning value: a finite 95% problem-group
split-conformal cross-check needs 19 untouched conformal groups, so P1 now
requires exactly 48 calibration groups (29 fit, 19 conformal) and retains a
64--128 risk-test planning range subject to the frozen power rule. A suggested
history calibration grid is `R in {4,8,16,32}` with separate construction and
label seed pools. The test range remains planning guidance, not a frozen result.

## 8. Minimum new components

Before any held-out execution, P1 needs:

1. a `Coconut-OPT` engineering extractor that freezes batch, optimizer-state
   semantics, and exact induced activation deltas;
2. source-equivalent Latent-GRPO 1B action/objective extraction plus an
   early-action deterministic recursive-closure and same-history sample/force
   adapter;
3. a frozen semantic-tail support scan that cannot inspect geometry or update
   consequences;
4. an online **directional** expected-trajectory-Fisher estimator with
   seed-disjoint construction and evaluation pools;
5. a hierarchical artifact schema keyed by optimizer trajectory, prompt,
   update family, update instance, and rollout seed;
6. explicit stop-first order, `ignore_eos`, EOS/stop/additional-stop ids, stop
   strings, latent timeout, proxy, support, and natural-exit semantics for every
   adapter, including proof that id 524 is not preempted by a generic stop;
7. matched-compute direct continuation-KL Monte Carlo and all baselines required
   by `P0_KILL_CRITERIA.md`; and
8. fake-model or tiny-model tests for identity KL, exact same-history forcing,
   RNG reproducibility, temperature scaling, stopping events, recursive
   finite-difference JVP, full-support Fisher, and exact hard-support loss.

Two directional expected-Fisher implementations should be compared on
calibration compute and variance:

- exact conditional categorical Fisher using the existing JVP machinery; and
- a streaming sampled trajectory-score outer product, which uses fewer
  derivative passes but may require more histories.

Neither may be chosen on held-out outcomes.

## 9. Remaining blockers before freeze

P1 may be redrafted now, but remains `NO-GO-EXECUTION` until:

1. the early deterministic recursive interface and both natural update
   constructions pass source preflight;
2. recursive closure and same-history forcing pass identity, source-equivalence,
   token/string/length-stop precedence, timeout, support, proxy, and exit tests;
3. a source-only semantic-tail scan confirms enough non-formatting visible
   continuation to separate `h`, `H_cert`, and `H_eval`;
4. the exact full-support primary law, extended-real truncated sensitivity,
   recursive estimand, and special-token semantics are frozen;
5. optimizer/objective construction and evaluation data are demonstrably disjoint;
6. a fixed microbenchmark closes the 8-GB JVP boundary and numeric GPU-hour,
   wall-time, disk, prompt, candidate, and history caps are frozen;
7. mandatory baselines, false-safe prevalence/count thresholds, support/OOD
   gates, and simultaneous comparison family are fully specified; and
8. an exact adapted-metric baseline contract is frozen; and
9. independent source, mathematics/statistics, and collision reviews pass the
   resulting preregistration.

Until then, the correct next action is source-only/fake-model adapter design and
baseline specification. It is not to run frozen C2, open held-out outcomes,
rent an H20, or implement the adaptive certificate.
