# Score-Squashed Gumbel Policy Optimization

This directory develops and tests SSG-PO, a sparse latent policy that applies a
strictly increasing smooth squash to complete Gumbel-perturbed token scores.
The transform preserves ordinary Gumbel Top-K support exactly, bounds the score
range used for mixture weights, and has a fixed continuous support suitable for
cross-policy likelihood ratios.

## Scientific boundary

The method does not claim that tanh squashing, Gumbel Top-K, transformed
distributions, or order-statistic likelihoods are individually new. The target
contribution is a sampler-objective contract for sparse vocabulary-mixture
latent reasoning, together with a real optimization consequence.

## Authorized evidence order

1. derive and independently validate the selected-score density;
2. freeze and run a synthetic support/concentration gate;
3. complete the direct-collision audit;
4. freeze a public-checkpoint rollout audit;
5. train only if all prior gates pass.

Hard clipping and dynamic top-p remain baselines, not hidden parts of the SSG
policy. Gate 1 is complete: the exact density, known boundaries, independent
change-of-variables reference, score identity, ratio normalization, candidate
support, and paired-support identity pass their targeted tests. This establishes
mathematical validity, not method usefulness. No checkpoint download or
training is authorized. The frozen Gate 2 subsequently failed because the
fixed squash produced substantially sharper mixtures than hard clipping in
diffuse-logit regimes. SSG-PO is stopped as a method; its theory and negative
trade-off result are retained for the Latent Policy Contract Audit.
