# LPCA Literature and Novelty Boundary

Last primary-source verification: 2026-07-14.

This document is a claim boundary, not a general latent-reasoning survey. It
records which nearby papers already own each broad idea and what evidence LPCA
would still need for a defensible contribution.

## Direct collisions

### RLPT: discrete support alignment

[Reinforcement Learning with Promising Tokens](https://arxiv.org/abs/2602.03195)
explicitly formulates Top-K/Nucleus rollout as a truncated behavior policy,
identifies full-vocabulary optimization as an off-policy mismatch, stores the
behavior support, and reuses it for masked current/old token probabilities and
ratios. A newer OpenReview version makes the support-alignment claim even more
explicit.

**Claim removed from LPCA:** discovering rollout-truncation versus
full-vocabulary optimization mismatch, or proposing same-support masking, in
ordinary discrete-token LLM RL.

**Boundary left open:** RLPT's action, executed object, and recorded token are
the same discrete variable. It does not address continuous or mixed latent
actions, many-to-one embedding execution, proxy-token histories, Concrete
density scores, or custom latent surrogate gradients.

### NF-CoT: exact continuous-thought likelihood

[Latent Reasoning with Normalizing Flows](https://arxiv.org/abs/2606.06447)
builds a flow-based continuous-thought architecture with tractable likelihood,
autoregressive KV-cache sampling, and direct policy-gradient refinement.

**Claim removed from LPCA:** being the first method to give continuous latent
thoughts exact likelihoods or to apply policy gradients to an exactly scored
latent architecture.

**Boundary left open:** NF-CoT is an architectural positive control. It does
not provide a source-pinned cross-method audit of what existing latent-policy
systems sample, execute, store, score, clip, and expose to stopping/reward
logic. No official code or model release was discoverable from its project
page on the verification date, so implementation-level claims remain pending.

### Dropout-GRPO: replayable augmented action

[Dropout-GRPO](https://arxiv.org/abs/2606.10184) treats one rollout-shared
Bernoulli dropout mask as part of an augmented action `(mask, answer)`. It
stores the mask through a replayable RNG seed, reconstructs the same latent
recurrence during the update, and proves the mean-centered group surrogate is
the policy gradient of a mask-marginalized Bayesian model-average objective at
the old policy, up to the finite-group factor.

**Claim removed from LPCA:** being the first latent-reasoning work to specify a
replayable stochastic action, align rollout and update randomness, or prove a
well-defined latent-policy gradient.

**Boundary left open:** this is a strong contract-compliant positive control,
not a cross-family contract audit. Its stochastic node is an exogenous,
parameter-independent discrete mask and its scored answer remains a standard
token sequence. It does not audit filtered Concrete mixtures, many-to-one
execution, proxy histories, changing supports, or acknowledged soft-label
surrogates. No official code repository was found on the verification date, so
its source-level reproduction remains conditional on a release.

Dropout-GRPO's unbiasedness theorem uses a mean-only group advantage and
explicitly warns that dividing by the random within-group reward standard
deviation introduces bias. LPCA B2 retains LEPO's standard-deviation
normalization for source fidelity; its `exact` branch therefore means an
exact-density score direction, not an unbiased expected-reward gradient.

## Audited target methods

### Four Axioms: representation-level audit

[Formalizing Latent Thoughts: Four Axioms of Thought Representation in
LLMs](https://arxiv.org/abs/2606.27378) already provides an axiomatic intrinsic
audit of latent representations. It formalizes Causality, Minimality,
Separability, and Stability and evaluates candidate representations across
five open-weight models and 23 reasoning tasks, including soft thinking with
Gumbel noise.

**Claim removed from LPCA:** being the first formal, axiomatic, intrinsic, or
cross-model audit of latent thoughts in general.

**Boundary left open:** the Four Axioms paper asks whether a representation is
a functionally adequate thought state. It does not define or test the
probability contract of a stochastic latent RL policy: which random action is
sampled, which measure scores it, which object is executed and stored, whether
old/current supports admit a ratio, how proxy histories affect control flow,
or whether a surrogate changes reward-conditioned policy updates. LPCA must
make this policy-semantic object, rather than generic representation quality,
the center of its title, theory, and benchmark.

The distinction is explicit in its protocol: stochastic extraction methods
such as Gumbel soft thinking use one fixed global seed so the representation
map remains deterministic. That is appropriate for its axioms but deliberately
removes the rollout distribution, likelihood, and off-policy reuse questions
that LPCA studies.

### LEPO

[LEPO](https://arxiv.org/abs/2604.17892v4) samples Gumbel-Softmax latent
vectors, executes expectation embeddings, and explicitly names Eq. 9 a soft
surrogate objective. The paper states that the one-hot case recovers the
discrete objective, but the released default filters the sampler support while
the surrogate uses the raw full-model softmax. The pinned code also advances a
clean-logit argmax proxy stream for stopping and decoding while the transformer
consumes continuous embeddings.

**LPCA obligation:** do not criticize LEPO merely for using a disclosed
surrogate. Measure whether its reward-conditioned direction and dual histories
change decisions at operational scale. Stage A establishes prevalence; Stage
B is the first consequence gate.

### Latent-GRPO

[Latent-GRPO](https://arxiv.org/abs/2604.27998) explicitly describes its
one-sided Gumbel-margin quantity as a surrogate latent likelihood with a custom
backward rule, not an exact probability density. It also diagnoses latent
mixture non-closure and selectively updates a first latent token.

**LPCA obligation:** separate acknowledged surrogate semantics, clipping
atoms, post-noise support selection, omitted selection events, aggregation,
and the advantage-dependent backward rule. None is evidence of harm until a
source-faithful decision or training consequence is shown.

### SofT-GRPO

[SofT-GRPO](https://arxiv.org/abs/2511.06411) already defines a Gumbel
auxiliary action, reconstructs soft-token execution, and uses a latent
likelihood-ratio surrogate. The generic Gumbel-Softmax/Concrete mathematics is
therefore not an LPCA novelty claim.

**LPCA obligation:** audit whether the scored auxiliary variable, the executed
mixture, component aggregation, clipping, and behaviorally null directions
form a coherent contract and whether any distinction changes PPO decisions.

## Adjacent ratio and engine-mismatch work

### CTPO and NFPO

[CTPO](https://arxiv.org/abs/2605.07331) derives cumulative prefix importance
ratios for token-level LLM policy gradients and contrasts their bias and
variance with token- and sequence-level ratios.
[NFPO](https://arxiv.org/abs/2605.20865) studies multi-step forward likelihood
ratios as a bridge between local PPO surrogates and longer-range correction.

**Claim removed from LPCA:** a generic first diagnosis that local token ratios
omit prefix state-distribution correction, or a generic first multi-step ratio
repair.

**Boundary left open:** these papers assume a well-defined discrete token
policy. LPCA first asks whether a latent method has specified the random action,
measure, support, executed transition, and stored history on which any prefix
ratio would be built.

### MIPI/MIPU

[The Mirage of Optimizing Training Policies](https://arxiv.org/abs/2606.29526)
studies probability mismatch between separate rollout and training engines and
optimizes inference-side improvement.

**Boundary left open:** engine numerical mismatch is distinct from a
within-implementation semantic mismatch between continuous execution,
proxy-token control flow, and the quantity used as a latent likelihood.

### LatentSeek and GoRL

[LatentSeek](https://arxiv.org/abs/2505.13308) optimizes instance-specific
pre-head latent representations at test time using the exact likelihood of
decoded tokens. The stochastic policy action remains the decoded token, while
the latent vector is a directly optimized conditioning variable; it is not the
same action/density problem as a sampled continuous latent mixture.

[GoRL](https://arxiv.org/abs/2512.02581) is a useful continuous-control
precedent for separating a tractable latent policy from a deterministic or
conditional action generator. It removes any broad claim that such
optimization/generation decoupling is new, while providing an external
positive-control pattern for LPCA's execution-map contract.

## Reward and alternative-paradigm context

[Likelihood-Based Reward Designs for General LLM Reasoning](https://arxiv.org/abs/2602.03979)
provides independent motivation for reference-continuation log likelihood as a
dense reasoning reward. LPCA Stage B2 uses it only as a frozen local utility;
it does not equate that utility with MATH accuracy or final-answer reward.

[Discrete Latent Reasoning](https://arxiv.org/abs/2606.29712) argues for learned
discrete latent tokens as an alternative to unstable continuous latents. It is
not a policy-contract audit, but it raises the bar for why continuous latent RL
should be retained rather than replaced by a canonical discrete action space.

## Defensible LPCA thesis after collision checks

The strongest remaining thesis is:

> A latent-reasoning policy is not specified by naming a noise source or a
> surrogate loss. The sampled random variable, its induced measure, the object
> executed by the model, the stored rollout state, the control-flow history,
> and the old/current likelihood construction must define one explicit
> contract. Released methods instantiate materially different contracts, and
> source-pinned audits can determine when those distinctions alter gradient,
> clipping, support, stopping, or continuation decisions.

This is primarily a theory, analysis, and benchmark contribution. It becomes
a method paper only if a preregistered consequence gate identifies a specific
repair that then wins a matched-compute training comparison. Exactness alone,
or a failed identity check without decision consequences, is insufficient.

## Minimum evidence for a competitive main paper

1. A formal contract taxonomy with exact positive and negative controls.
2. Source-pinned executable audits of at least LEPO, SofT-GRPO, and
   Latent-GRPO, plus formal positive controls from NF-CoT and Dropout-GRPO;
   both become executable targets if official code is released.
3. Real-checkpoint, reward-conditioned consequences rather than pointwise
   score mismatch alone.
4. Honest positive findings, including LEPO's complete default archive and
   every disclosed-surrogate boundary.
5. Cross-model or cross-method replication and explicit public-checkpoint
   limitations.
6. A repair only when a frozen gate demonstrates local and training-level
   utility under matched samples, tokens, wall-clock, and compute.

The paper must also compare explicitly against the Four Axioms framework and
show that policy-contract metrics answer questions its representation metrics
cannot. Merely adding a new checklist to a latent-thought audit is not enough.
