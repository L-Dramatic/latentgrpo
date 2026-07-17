# Independent Audit of the ChatGPT Pro Report

Audit date: 2026-07-18

## What the report did well

The report is materially grounded rather than speculative. It read the frozen
repository commit, reconstructed the official Latent-GRPO sampler and replay
contract, preserved prior KILL decisions, searched a broad recent literature,
and refused to authorize immediate full training. Its gap taxonomy and staged
resource discipline are useful project assets.

## Primary-method collision

The OMPI-R objective has two defining operations:

1. form a multi-path response log likelihood with log-mean-exp; and
2. reward-tilt and cross-entropy-fit the finite candidate distribution.

Both operations already have direct method ancestors:

- [Target Policy Optimization](https://arxiv.org/abs/2604.06159) defines
  `q_i proportional to p_old_i exp(u_i/eta)`, minimizes candidate cross entropy,
  and obtains candidate-logit gradient `p-q`.
- [Beyond Verifiable Rewards / JEPO](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6bd67a424dc59481e1e5a5061ffc8dfe-Abstract-Conference.html)
  treats chain of thought as a latent variable. Its multi-sample objective puts
  the log outside the average conditional answer probability and explicitly
  interprets that average as marginalizing chain of thought.
- [Training Chain-of-Thought via Latent-Variable Inference](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e69a9560c450ca76584d9eb37e7f5ae8-Abstract-Conference.html)
  already optimizes answer marginal likelihood by averaging over latent
  rationales.
- [Latent Chain-of-Thought for Visual Reasoning](https://arxiv.org/abs/2510.23925)
  and [Latent Thought Flow](https://arxiv.org/abs/2606.16222) further crowd the
  posterior/marginal latent-reasoning framing.

OMPI-R is a coherent application of these ideas to private continuous paths,
but the exact identity gate shows that its advertised responsibility update is
the composed gradient, not a new optimizer. The report's refinement invariance,
responsibility identity, and finite-candidate target bound do not recover an
independent theoretical contribution.

## Missing objective term

JEPO's derivation makes the distinction especially clear: when latent paths are
sampled from a policy-dependent proposal, the gradient contains both a
proposal/REINFORCE term and a conditional-answer term. OMPI-R freezes the
proposal and keeps only the latter family of effects. This may be a practical
replay surrogate, but it weakens the report's headline claim that it optimizes
the observable policy itself.

## Gate-quality assessment

The proposed Gate 0 is mathematically clean but favors the method by design:
Rao-Blackwellization over intentionally aliased paths should beat a diagonal
estimator. It is an implementation check, not a novelty test.

Gate 1A is not identifiable as written. Its three main statistics all become
maximal under latent irrelevance. The report checks responsibility collapse to
one path but misses the opposite collapse in which every path is behaviorally
identical.

## AAAI-level assessment after subtraction

| Dimension | Report as research audit | OMPI-R as method paper |
|---|---:|---:|
| Factual grounding | 9/10 | 8/10 |
| Literature breadth | 8/10 | 5/10 independent remainder |
| Technical coherence | 8/10 | 6/10 |
| Method novelty | n/a | 3/10 |
| Evidence readiness | 7/10 gate design | 1/10 current evidence |
| AAAI main-track viability | n/a | 3/10 |
| Compute efficiency of next justified step | 9/10 | 10/10: stop before GPU |

The report is a strong red-team artifact, but its selected winner does not
survive the report's own novelty KILL condition after exact subtraction.

## What remains valuable

- The observable/private-computation distinction should remain a diagnostic.
- The all-pairs likelihood matrix can be used to measure whether latent paths
  are irrelevant, separated, or behaviorally aliased.
- OMPI-R should remain as a mandatory composed baseline if a genuinely new
  response-marginal method is proposed later.
- OMPI-P is not evaluated by this exact frozen-proposal identity, but it would
  need a separate claim and collision audit against pathwise latent-policy and
  variational objectives. It is not authorized by the current report.

