# Latent Policy Contract Audit: Source-Pinned Method Matrix

Audit date: 2026-07-14. Source commits and exact locations are recorded in
`SOURCE_MANIFEST.json`. A row marked `surrogate` is not automatically a defect:
the audit asks whether the surrogate is named, normalized for its stated role,
and empirically faithful enough for the downstream decision that uses it.

## Contract definitions

1. **Action semantics**: What random variable is called the policy action?
2. **Execution map**: Which function of that variable is consumed by the next
   model step, stopping logic, and reward code?
3. **Normalization**: Is the evaluated object a normalized probability law?
   For an exact old/new density, `E_old[p_new / p_old] = 1`; for a regular
   score, `E_policy[grad log p] = 0`.
4. **Support**: Do old and current laws share support wherever a ratio is used?
5. **Sampler-likelihood consistency**: Does the evaluated likelihood describe
   the same random object and history that generated the rollout?
6. **Aggregation**: At what level are component, step, and trajectory terms
   summed, averaged, clipped, and normalized?

## Matrix

| Method | Stochastic action | Executed object | Stored rollout state | Training quantity | Current audit status |
|---|---|---|---|---|---|
| SofT-GRPO | Scores on a preselected, renormalized top-k vocabulary support plus Gumbel noise clamped to `[-1.5, 3]`, then softmaxed and sorted | Softmax-weighted embedding mixture; first sorted support id is also the discrete proxy | Sorted top-k ids and perturbed scores, plus token stream | Current probabilities renormalized on stored ids; inferred Gumbels are clamped again, components below a current-log-probability threshold are masked, and remaining component log terms are averaged; discrete token log-probability follows the soft phase | Execution is reconstructable, but the clamp creates boundary atoms and the current-dependent component mask plus mean aggregation define a surrogate rather than the exact mixed law. The auxiliary perturbed-score action also retains random directions erased by simplex execution. Each distinction requires a separate test. |
| LEPO | Gumbel-Softmax vector on logits already filtered by configured TopK and TopP processors | Full expectation embedding `z E`; filtered-logit argmax proxy drives the id stream when latent sampling is disabled | Full executed embeddings, proxy ids, and the top-k entries of `z` | Advantage-weighted soft cross entropy against the unfiltered current-model softmax for the latent prefix; GRPO-style token objective for suffix | Archive contract passes. The acknowledged surrogate has score `z-softmax(raw logits)`, not the exact filtered-Concrete score, and the continuous execution history differs from the proxy-id history. Real-checkpoint reward-conditioned score-direction and history-effect tests are required. |
| RLPT | Discrete token sampled from a behavior-policy Top-K support | The sampled token itself | Token trajectory plus the rollout support mask | Current and old token log-probabilities renormalized on the stored rollout support | Direct prior art for support-aligned discrete LLM RL. It explicitly identifies ordinary Top-K/Nucleus rollout versus full-vocabulary optimization as off-policy mismatch, so LPCA cannot claim that diagnosis or same-support repair as novel. It does not define or audit continuous latent actions, many-to-one execution maps, Concrete densities, proxy histories, or latent surrogate gradients. |
| Latent-GRPO | Clipped and shifted Gumbel-perturbed scores followed by post-noise top-k selection | Softmax-weighted embedding mixture over selected support | Selected ids and perturbed scores, proxy token stream | Mean selected marginal-Gumbel surrogate with an advantage-dependent straight-through backward rule | Paper acknowledges the final objective as a surrogate. Dynamic support, clipping atoms, omitted selection event, and backward-only gradient modification are separate contract dimensions and must not be conflated. |
| Exact Concrete reference | Full positive simplex vector | Full expectation embedding | Full simplex action | Exact Concrete density | Normalized common-support control. Computationally expensive at full vocabulary but analytically tractable. |
| Exact Top-K Concrete reference | Ordered support plus normalized top-k weights | Corresponding top-k embedding mixture | Ordered support and weights | Exact mixed discrete-continuous density | Normalized control for post-noise support selection. Earlier frozen gates showed support selection dominates its KL in the tested regimes, so it is a diagnostic reference rather than the current method proposal. |
| NF-CoT | Flow-generated continuous thought | Continuous thought inside the shared causal stream | Paper describes invertible flow state and exact likelihood terms | Change-of-variables flow likelihood and policy-gradient objective | Strong positive control at the formal level. No official code/model link was exposed on the project page at audit time, so implementation claims are prohibited until a release appears. |
| Dropout-GRPO | Rollout-shared Bernoulli mask plus the sampled answer sequence | Mask-perturbed latent recurrence followed by the sampled answer | Replayable mask RNG seed and answer tokens | Token likelihood conditioned on the identical replayed mask; mean-centered group surrogate | Strong formal positive control for explicit augmented-action semantics and rollout/update replay. The mask law is exogenous and parameter-independent, so this does not solve filtered-Concrete or proxy-history contracts. No official code was found at audit time. |

## LEPO paper-code boundary

The v4 paper's Eq. 9 sums over the entire vocabulary and presents the sampled
Gumbel-Softmax vector as the soft label. A first-pass audit incorrectly treated
the model's raw logits as the sampler input and therefore appeared to find lost
tail mass. That result is invalid for the released default. In Transformers'
sampling path, `next_token_scores` has already passed through TopK and TopP
processors before LEPO calls Gumbel-Softmax. The same configured `top_k=30` is
then used for archiving, so the archive contains the complete nonzero action
almost surely. The implemented top-k sum is therefore equal to the paper's
full-vocabulary sum for the released configuration.

The remaining source-grounded questions are different:

1. **Surrogate semantics**: the reward-weighted soft cross entropy is not the
   log density of the sampled continuous Gumbel-Softmax action. The paper names
   it a surrogate, so LPCA must measure bias and decision consequences rather
   than treating non-exactness alone as a defect.
2. **Dual histories**: the transformer consumes the continuous embedding,
   while stopping, decoding, and any history-sensitive logits processors see a
   proxy id stream based on the filtered clean-logit argmax.
3. **Paper/source versioning**: the pinned repository's last commit predates
   arXiv v4, so code conclusions are scoped to that commit.
4. **Checkpoint availability**: no public trained checkpoint was found on the
   audit date. The repository includes a latent-32 evaluation config pointing
   to the public Qwen2.5-3B base model, so public sequential results are scoped
   to that base-model stress test until trained weights become available.

The paper already labels the latent objective a **soft surrogate objective**.
LPCA therefore must not frame surrogate use itself as concealment. The stronger
scientific question is whether the surrogate satisfies the identities and
decision stability expected by the optimizer that consumes it.

## RLPT novelty boundary

RLPT is a direct collision with the broad claim that rollout truncation and
full-vocabulary optimization are newly recognized as inconsistent. Its paper
defines the actual discrete behavior policy on a state-dependent Top-K support,
stores the behavior mask, and reuses that same support for current/old masked
log-probabilities and ratios during optimization. LPCA therefore treats this
as established discrete-policy prior art and as a positive support-alignment
control.

The remaining LPCA thesis is narrower and structurally different. Latent
reasoning may sample a continuous or mixed action, execute a many-to-one
embedding map, maintain a separate proxy-token history, and optimize a named
surrogate that is not a log density of the executed action. RLPT does not
specify these contracts or test their sequence-level and reward-conditioned
consequences. Any LPCA novelty claim must be attached to that latent-policy
contract, the cross-method audit, and demonstrated consequences, not to token
truncation by itself.

## Claims currently allowed

- Different released latent-RL systems assign policy semantics to different
  random objects even when all are described informally as soft or latent
  tokens.
- A likelihood can be exact for an auxiliary variable yet allocate ratio and
  clipping variation to directions erased by the execution map.
- LEPO's default top-k archive covers its filtered Gumbel-Softmax support; this
  is a positive contract result and an explicit correction to the invalid v1
  raw-logit audit.
- Latent-GRPO's released objective is an acknowledged surrogate and includes a
  backward-only, advantage-dependent modification.
- NF-CoT is a formal exact-likelihood positive control, not yet a source-level
  control.
- Dropout-GRPO is a formal replayable-action positive control: it explicitly
  includes the shared mask in the augmented policy and replays rollout
  randomness during the update.
- RLPT already establishes same-support masking for ordinary discrete-token
  rollout and optimization; LPCA's claim begins only beyond that setting.

## Claims not yet allowed

- That any observed contract violation harms benchmark accuracy.
- That exact likelihood necessarily produces better policy optimization.
- That a surrogate is invalid merely because it is not an exact density.
- That source behavior at the pinned commits describes later unreleased code.
- That LEPO's acknowledged soft surrogate causes harmful optimization bias.
- That support-aligned Top-K optimization for discrete-token RL is new.
- That replayable latent stochasticity or a well-defined latent-policy gradient
  is new in general.

Those claims require the preregistered public-checkpoint audit and, only after
its effect-size gates pass, matched training diagnostics.
