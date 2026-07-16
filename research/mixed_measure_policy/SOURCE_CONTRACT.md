# Source Contract

## Pinned implementation

Official checkout commit:
`c0994fbefb9de023912878534c7ae213b44b1966`.

Sampler:
`_external/official_latent_grpo/sglang_latent_reasoning_pkg/python/sglang/srt/layers/sampler.py`.

Objective:
`_external/official_latent_grpo/verl-0.4.x/verl/utils/torch_functional.py`.

## Executed random object

At a latent step, the pinned sampler performs the following operations:

1. compute full-vocabulary log probabilities;
2. construct a policy-dependent top-p candidate mask while retaining at least
   the requested Top-K count;
3. sample standard Gumbel noise and hard-clip it to `[-1.5, 3.0]`;
4. add the clipped noise to candidate log probabilities;
5. select ordered post-noise Top-K token IDs;
6. softmax selected scores into mixture weights;
7. execute the embedding mixture and expose the first selected ID as proxy.

A source-faithful action therefore cannot be represented only by K independent
continuous Gumbel values. It contains:

- a dynamic candidate set;
- an ordered discrete support;
- continuous interior score variation;
- lower and upper clipping events with positive mass;
- mixture weights that determine the executed latent embedding;
- a proxy token that affects request/control flow.

## Released optimization contract

The pinned objective gathers the current log probabilities of stored selected
IDs, subtracts them from stored selected scores, evaluates the standard
unclipped Gumbel log density componentwise, and averages across K. It does not
include the probability of the post-noise Top-K selection event, the dynamic
candidate event, or Dirac masses created by clipping. For negative advantages,
it also applies a backward-only sign rule while preserving the forward value.

This is a documented surrogate contract, not the log likelihood of the full
executed action.

## Existing local evidence

The source-faithful replay already recorded that selected upper clipping atoms
occurred in `27.89%-35.1%` of selected entries, clipping changed complete
ordered Top-K support in at least `98.65%` of samples, dynamic current top-p
support rejected `7.69%` of old clean actions on average, and exact versus
official-style clipping decisions differed on `37.59%` of clean samples. See
`research/topk_concrete/EXPERIMENT_LOG.md`.
