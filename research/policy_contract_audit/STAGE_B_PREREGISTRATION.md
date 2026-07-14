# Stage B Preregistration

Stage B was frozen after public-checkpoint Stage A passed and before any
sequence-effect or reward-conditioned gradient result was inspected.

## Frozen configurations

- Sequential replay: `configs/stage_b_sequential_v1.json`
- Canonical sequential config SHA-256:
  `63ea4518ae2ac5e2ff52eecd385eda68cba125f950612205a8eee802c512a25f`
- Reward-gradient audit: `configs/stage_b_gradient_v1.json`
- Canonical gradient config SHA-256:
  `1605fe744bdca5dad7d4dc1a99b295f80a682c11841bf2c3010ea3999d218f3f`

Canonical hashes use UTF-8 JSON serialized with sorted keys and compact
separators. Both protocols pin the same model, dataset, prompt source, and LEPO
source commit used in Stage A.

## B1: sequential paired replay

B1 asks whether LEPO's continuous execution history and its proxy-token
history are operationally interchangeable. For each source rollout, three
branches share the prompt and source proxy sequence:

1. the source branch executes each sampled filtered-Concrete mixture `zE`;
2. the projected branch executes the embedding of the source proxy id;
3. an identity branch duplicates the projected branch as a numerical control.

The projected replay is deliberately teacher-forced on the source proxy ids.
It isolates the accumulated consequence of the execution map for one stored
rollout; it is not presented as an alternative autonomous policy. Distribution
divergence is recorded after latent horizons 1, 4, 8, 16, and 32. The first 16
tokens of the dataset's reference solution are then scored under both final
states with identical teacher forcing.

The 96 prompts are selected without labels or outputs by sorting SHA-256 hashes
of the fixed salt and dataset `unique_id`. Two temperatures and two seeds yield
384 paired trajectories. B1 passes only if every numerical control and at
least three of five frozen sequence-effect gates pass.

## B2: reward-conditioned gradient audit

B2 tests a stronger question than Stage A's pointwise score comparison. It
defines a frozen dense utility for each sampled action: the mean teacher-forced
log probability of the first 16 reference-solution tokens after the sampled
mixture is executed. This is a local continuation utility, not MATH accuracy
and not a claim about final-answer reward.

For each of 48 hash-selected prompts and three temperatures, 128 actions form
16 groups of eight. Rewards are standardized with the Bessel-corrected sample
standard deviation used by the pinned trainer to mirror its zero-mean
group-relative advantage structure. The exact filtered-Concrete
score and released soft-label surrogate then produce two reward-conditioned
logit directions. A disjoint set of 256 actions evaluates fixed-support
candidate updates at frozen RMS steps using exact Concrete importance ratios.

Here and in the artifacts, `exact` means that the direction uses the exact
filtered-Concrete density score. It does not mean that the direction is an
unbiased gradient of raw expected continuation utility: the source-faithful
within-group sample-standard-deviation normalizer is random and couples all
rewards in a group. Both directions share that normalization, so B2 isolates
the score choice; the disjoint importance-ratio evaluation is the operational
utility test. This interpretation clarification changes no sample, estimator,
threshold, aggregation rule, or decision gate.

Matched training is not authorized by score mismatch alone. It requires all
controls, both gradient-semantic effects, and evidence that the exact candidate
has a better held-out local decision: either a positive paired gain with its
bootstrap lower bound above zero, or a ten-point advantage in positive-update
rate. The sign-disagreement metric is descriptive for this decision and cannot
alone authorize training.

## Interpretation and prior-art boundary

RLPT already identifies Top-K/Nucleus rollout versus full-vocabulary training
as an off-policy mismatch in ordinary discrete-token RL and reuses the stored
behavior support during optimization. B1 and B2 therefore do not claim support
alignment itself as new. They target continuous latent actions, a many-to-one
execution map, proxy histories, and a reward-conditioned surrogate/exact-score
comparison that RLPT does not formulate.

Engineering preflights may repair crashes, cache handling, record resumption,
or numerical-control failures. They may not change row selection, effects,
thresholds, horizons, temperatures, rewards, or aggregation after any
effect-bearing output is produced. Any invalidated run remains in the
append-only experiment log.

## Public-checkpoint limitation recorded after engineering preflight

No author-released trained LEPO checkpoint was discoverable on 2026-07-14.
The official repository's latent-32 evaluation example points to the public
Qwen2.5-3B base model, which is the checkpoint already frozen above. A one-row
engineering preflight found frequent singleton support at later latent steps.
The frozen B1 run is therefore interpreted only as a public-base stress test;
it cannot support claims about a trained LEPO policy. B2 remains a first-action
audit at an in-distribution prompt state and keeps its original protocol.
