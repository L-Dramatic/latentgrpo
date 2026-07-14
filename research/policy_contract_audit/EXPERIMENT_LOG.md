# LPCA Experiment Log

## 2026-07-14: LEPO analytic audit v1 - invalidated for source claims

- Config: `configs/lepo_analytic_audit_v1.json`
- Config SHA-256:
  `d69c321b4ac26ec4eede19973112588bb6b9103673c635d4c988ce7fc2ef847f`
- Artifact: `artifacts/policy_contract_audit/lepo_analytic_audit_v1.json`
- Frozen checks: 6/6 passed on the synthetic distribution.

The run used finite raw logits as direct inputs to a full-vocabulary Concrete
sampler. It detected omitted top-k tail mass and a large difference from the
exact Concrete score. After the run, a deeper source trace established that
LEPO calls Gumbel-Softmax on `next_token_scores` *after* Transformers applies
the configured TopK and TopP processors. With the released `top_k=30`, the
nonzero support is already bounded by the 30 archived entries. Therefore:

- v1 remains a valid counterfactual stress test for an unfiltered variant;
- v1 is **not valid evidence** that released LEPO drops action mass;
- no v1 tail or reconstruction statistic may appear as a LEPO result;
- thresholds were not edited after seeing the result;
- a source-faithful v2 must pass archive-normalization controls before any
  surrogate-score result is considered.

This invalidation is retained rather than erased because the correction is part
of the audit trail and guards against repeating the same source-tracing error.

## 2026-07-14: LEPO source-faithful analytic audit v2 - pass

- Config: `configs/lepo_source_faithful_audit_v2.json`
- Config SHA-256:
  `bb7882ad25449e0a09e838875f4e64d49af7bb7c786f6e665e5d82fad0d34eb5`
- Artifact: `artifacts/policy_contract_audit/lepo_source_faithful_audit_v2.json`
- Frozen checks: 8/8 passed on 18 scenarios and 540,000 actions.

Positive contract results:

- maximum archive mass error: `8.88e-16`;
- maximum reconstruction L1 error: `0`;
- active support range: `24-29`, below the archived width of 30;
- exact Concrete score control: maximum SNR `1.145 < 5`;
- exact density-ratio control: maximum deviation `1.902 < 5` standard errors.

Detected distinctions:

- minimum proxy-mode disagreement: `54.99%`;
- minimum surrogate expected-score SNR: `11.836`;
- minimum pointwise surrogate/exact score gap: `4.497` in L2 norm;
- mean pointwise surrogate/exact score cosine: `0.314`;
- mean dynamic support churn under the frozen synthetic drift: `9.01%`.

Interpretation boundary: v2 establishes that the archive is correct and that
the acknowledged surrogate is not the exact score of the source-faithful
filtered Concrete action. It does not establish benchmark harm, because its
logits and rewards are synthetic. Real-checkpoint gradient and behavior gates
are preregistered separately.

## 2026-07-14: public-checkpoint Stage A v1 - pass

- Config: `configs/public_checkpoint_stage_a_v1.json`
- Frozen config SHA-256:
  `47536e7f9a9504ef958afa8b06280599025e6edbb0068e4cf5dfe7db97ebd629`
- Summary artifact:
  `artifacts/policy_contract_audit/public_checkpoint_stage_a_v1.json`
- Summary artifact SHA-256:
  `2ae86696f037964137bea64ae338ab1f56b11d498a8b7248ce9ae62bab53e320`
- Record artifact:
  `artifacts/policy_contract_audit/public_checkpoint_stage_a_v1_records.jsonl`
- Record artifact SHA-256:
  `4f07ef4c69ed0c3e452e6572d046782f6848f82231f0163d9a6f6742e0c5548c`
- Scope: all 500 MATH-500 prompts, three temperatures, 1,500 state-temperature
  records, and 32 actions per seed for two fixed seeds.
- Runtime: `252.79` seconds on the recorded RTX 4060 Laptop GPU environment.

All five controls passed:

- archive-mass error p99.9: `6.66e-16`;
- action-reconstruction L1 error p99.9: `0`;
- active support range: `7-23`, within the frozen bound of 30;
- exact-score SNR p99: `1.496 < 6`;
- exact-ratio deviation p99: `2.597 < 6` standard errors.

All five preregistered effects passed:

- excluded full-model mass median: `0.19295 >= 0.01`;
- proxy-mode disagreement mean: `0.51378 >= 0.10`;
- surrogate/exact score cosine median: `0.33124 <= 0.80`;
- surrogate/exact score relative error median: `0.96291 >= 0.25`;
- dynamic support churn mean: `0.03881 >= 0.01`.

The executed-mixture/proxy-token embedding relative L2 median was `1.08056`;
this metric was descriptive and did not enter the gate. The frozen decision
rule therefore authorizes Stage B. This result establishes prevalence and
operational scale at a public checkpoint, not reward harm, training failure,
or superiority of an exact-density replacement.

## 2026-07-14: Stage B1 engineering preflight attempt 1 - invalidated

The first one-state preflight stopped before writing a trajectory record when
a later TopK/TopP-filtered state had singleton support. The generic Concrete
sampler correctly rejects fewer than two categories, but released LEPO's
Gumbel-Softmax call degenerates to the deterministic one-hot action on a
singleton support. The replay now implements that source-faithful boundary and
records its frequency. No effect result was produced, and no frozen selection,
threshold, horizon, temperature, or aggregation rule changed.

## 2026-07-14: Stage B1 engineering preflight attempt 2 - controls pass

- Config: `configs/stage_b_sequential_v1.json`
- Frozen config SHA-256:
  `63ea4518ae2ac5e2ff52eecd385eda68cba125f950612205a8eee802c512a25f`
- Artifact:
  `artifacts/policy_contract_audit/stage_b_sequential_v1_preflight.json`
- Artifact SHA-256:
  `27d49240815d97320a5dd7dceb49151d6f6e8fdbc65d6ee79b33462c4236d236`
- Record artifact:
  `artifacts/policy_contract_audit/stage_b_sequential_v1_preflight_records.jsonl`
- Record artifact SHA-256:
  `32b846ba18ed829101bba4e706e9b01aad69f18f6567a214fd71ae09416dba17`
- Scope: one frozen hash-selected prompt, two temperatures, two seeds, and four
  32-step paired trajectories.

All engineering controls passed. Action normalization error was `4.44e-16`,
the projected/identity branch JS maximum was `1.26e-17`, all trajectories were
finite and complete, and projected/identity reference NLLs were identical.

The preflight also found singleton filtered support on `60.16%` of latent
steps. Sequence effects varied non-monotonically by horizon and are not used to
change any gate. No public trained LEPO checkpoint was found in the official
repository, paper links, web search, or Hugging Face model index on the audit
date. The official repository does provide a latent-32 evaluation config that
points to the public `Qwen/Qwen2.5-3B` base model. B1 v1 is therefore retained
as an author-supported public-base stress test, not evidence about a trained
LEPO policy. B2 evaluates only the first latent action and is not exposed to
the observed 32-step support collapse.

## 2026-07-14: Stage B2 preregistration correction before effects

Before any B2 reward or gradient result was produced, the advantage
normalization contract was corrected from population standard deviation to the
Bessel-corrected sample standard deviation used by the pinned LEPO trainer's
default `torch.std` call. No selection rule, effect threshold, candidate step,
reward, or decision rule changed. The corrected and subsequently frozen
canonical config SHA-256 is
`1605fe744bdca5dad7d4dc1a99b295f80a682c11841bf2c3010ea3999d218f3f`.

## 2026-07-14: Stage B2 engineering preflight - controls pass

- Config: `configs/stage_b_gradient_v1.json`
- Frozen canonical config SHA-256:
  `1605fe744bdca5dad7d4dc1a99b295f80a682c11841bf2c3010ea3999d218f3f`
- Artifact:
  `artifacts/policy_contract_audit/stage_b_gradient_v1_preflight.json`
- Artifact SHA-256:
  `881aef4a69a87d252ad4d5a794f1d9052459294762abfb6eff5ecea8bcd55b78`
- Record artifact:
  `artifacts/policy_contract_audit/stage_b_gradient_v1_preflight_records.jsonl`
- Record artifact SHA-256:
  `c1ddf93aa61f9727bf4704b5dd4e1e68dd14cadd74bc2708d790e93e0bb81146`
- Scope: one frozen hash-selected state, three temperatures, 128 gradient
  actions and 256 independent evaluation actions per temperature.
- Runtime: `135.17` seconds after model loading.

All engineering controls passed in the preflight: action-sum error was
`4.44e-16`, finite-reward rate was `1.0`, exact-gradient split-half cosine was
`0.409`, candidate ratio deviation p99 was `1.426`, and candidate ESS fraction
p10 was `0.969`.

The preflight's effect estimates are recorded only as a gate-sensitivity check,
not as a result: exact/surrogate gradient cosine was `0.831`, relative error was
`0.594`, exact gain was `0.004316`, surrogate gain was `0.004802`, and the
exact-minus-surrogate difference was `-0.000486`. Thus exactness was not
naturally superior on the first state and the preregistered training gate has a
real possibility of rejection. Thresholds and aggregation rules remained
unchanged before the full 48-state run.

## 2026-07-14: Stage B2 interpretation clarification - protocol unchanged

An independent collision review during the full run identified
Dropout-GRPO's mean-only group-advantage theorem and its explicit warning that
division by the random within-group reward standard deviation biases the raw
expected-reward policy-gradient estimator. B2 intentionally retains that
standard-deviation normalization because it is source-faithful to the pinned
LEPO trainer.

Therefore `exact` in B2 denotes an **exact filtered-Concrete density-score
direction under the released group advantage**, not an unbiased gradient of
raw expected continuation utility. The exact and surrogate branches still use
identical actions, rewards, groups, and advantages, so the score comparison is
controlled. The independent-action, exact-importance-ratio candidate test
remains a direct local-utility comparison. No running code, record, sample,
threshold, aggregation rule, or decision gate was changed, and no partial full-
run effect estimate was inspected to make this clarification.

## 2026-07-14: Stage B2 full reward-gradient audit - training gate fails

- Config: `configs/stage_b_gradient_v1.json`
- Frozen canonical config SHA-256:
  `1605fe744bdca5dad7d4dc1a99b295f80a682c11841bf2c3010ea3999d218f3f`
- Summary artifact:
  `artifacts/policy_contract_audit/stage_b_gradient_v1.json`
- Summary artifact SHA-256:
  `47d233229239100d83a31e4b3e27d9fa21aaa30dfdc65456db4ffc7fd38e5009`
- Record artifact:
  `artifacts/policy_contract_audit/stage_b_gradient_v1_records.jsonl`
- Record artifact SHA-256:
  `37c9d921595fd143d16770d828469e3e57145b0614cd5bbe9f227b167e6a3f2b`
- Scope: 48 frozen hash-selected prompts, three temperatures, 144
  state-temperature records, 128 gradient actions and 256 independent
  evaluation actions per record.
- Runtime: `4169.85` seconds on the recorded RTX 4060 Laptop GPU environment.

All five numerical controls passed. The exact-score direction was reproducible
enough for the frozen gate (split-half cosine median `0.44646`), importance
sampling remained stable (ESS-fraction p10 `0.96215` and ratio-z p99
`3.01372`), and every reward was finite. Both semantic mismatch gates also
passed: exact/surrogate gradient cosine median was `0.79481` and relative
error median was `0.70711`. The released surrogate and exact density score are
therefore materially different reward-conditioned directions.

The operational-superiority gates all failed. At the frozen RMS step `0.03`,
the exact candidate gain was `0.0063270`, while the released surrogate gain
was `0.0079559`. Their paired difference was `-0.0016289`, with the frozen
record-level bootstrap lower bound `-0.0018677`. Both candidates had positive
gain on every record, so sign-disagreement and positive-rate advantage were
both zero. The preregistered decision consequently sets
`authorize_matched_training=false`. The runner's nonzero exit denotes this
scientific gate failure, not an engineering failure.

## 2026-07-14: Stage B2 prompt-cluster robustness - confirms stop

- Artifact:
  `artifacts/policy_contract_audit/stage_b_gradient_v1_prompt_cluster_robustness.json`
- Artifact SHA-256:
  `46f9e5a93d5d96a0f2e99a9a56a8b0d33872f6b34b700cd470d549177e37cc78`
- Bootstrap: 10,000 prompt-level resamples; all three temperatures for a
  prompt remain in the same cluster.

The exact-minus-surrogate gain remained negative at every temperature:
`-0.001428` at `0.3`, `-0.001859` at `0.5`, and `-0.001600` at `1.0`. The
prompt-clustered 95% interval for the overall difference was
`[-0.001942, -0.001349]`, entirely below zero. This analysis is non-gating but
rules out pseudo-replication across temperatures as an explanation for the
stop decision.

No matched exact-density-substitution training will be run, and no cloud GPU
will be rented for that branch. This result rejects the proposed replacement
as a performance method under the frozen local-utility protocol; it does not
erase the demonstrated policy-contract mismatch. The next permitted work is
the source-faithful audit of independently released trained SofT-GRPO and
Latent-GRPO checkpoints, where the question is mechanism validity rather than
assuming that exact likelihood must improve reward.

## 2026-07-14: trained-checkpoint native-load smoke v1 - pass

- Config: `configs/trained_checkpoint_smoke_v1.json`
- Canonical config SHA-256:
  `e411ebf5654fa26b6e0d2ced46998050ebc7dcb2909e0a33538c01a65377bf5d`
- Artifact:
  `artifacts/policy_contract_audit/trained_checkpoint_smoke_v1.json`
- Artifact SHA-256:
  `516530c977adc755edc07e8af07732440726c32e0a9a0a8d07bbda9f072b5ac2`

The pinned SofT-GRPO Qwen 1.5B and Latent-GRPO Llama 1B checkpoints both pass
all seven integrity and no-effect forward gates: weight bytes, weight SHA-256,
config/input vocabulary agreement, output/input vocabulary agreement,
prompt-id range, finite logits, and exact repeatability across two evaluation-
mode forwards. Peak allocated GPU memory was 3,415 MiB and 2,866 MiB,
respectively, on the RTX 4060 Laptop GPU.

The Latent-GRPO tokenizer still contains one added id (`128256`) outside the
official checkpoint's valid embedding rows (`0..128255`). The frozen smoke
prompt used ids only through `128009`, so the forward is valid. No tokenizer
entry was removed and no embedding was resized. This pass authorizes only the
next source-sampler equivalence control; it provides no method-effect evidence.
