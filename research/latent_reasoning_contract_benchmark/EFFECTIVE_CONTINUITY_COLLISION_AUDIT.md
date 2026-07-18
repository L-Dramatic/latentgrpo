# Effective Continuity Collision Audit

Audit date: 2026-07-18

Decision: `KILL_AS_PRIMARY_IDEA`

## Candidate that was tested

The candidate claim was that methods described as continuous or soft may be
effectively discrete at runtime, and should be evaluated by action entropy,
effective support, soft-vs-hard transition divergence, and layerwise collapse
rather than by method names.

Our SofT-GRPO result is strong evidence for that phenomenon, but the paper claim
is not new enough.

## Direct collisions

| Neighbor | What it already establishes | Collision |
|---|---|---|
| [LLMs are Single-threaded Reasoners](https://arxiv.org/abs/2508.03440) | Soft inputs are dominated by the highest-probability token; uses JS diagnostics and proposes stochastic Gumbel soft thinking. | Direct collision on effective-discreteness diagnosis and standard remedy. |
| [The Illusion of Superposition?](https://arxiv.org/abs/2604.06374) | Across training-free, fine-tuned, and from-scratch regimes, probes entropy and soft-to-argmax interventions; reports collapse in pretrained/fine-tuned models. | Direct collision on entropy, intervention, and collapse taxonomy. |
| [Soft Tokens, Hard Truths](https://openreview.net/forum?id=9JjKTp8Jmy) | Trains continuous CoT with RL at scale and reports that soft training followed by discrete inference performs best. | Strong collision on the practical implication that runtime continuity need not carry the gain. |
| [SofT-GRPO](https://arxiv.org/abs/2511.06411) | Uses Gumbel-softmax RL and fixes `tau=0.1`; reports that raising it to `0.25` destabilizes training. | The low-softness/high-drift tradeoff is already visible in the source paper. |
| [Multiplex Thinking](https://arxiv.org/abs/2601.08808) | Samples several tokens and merges their embeddings, with an on-policy objective and public checkpoint. | A strong positive method baseline; a benchmark cannot ignore it. |

## Boundary

Adding more checkpoints to report which models collapse would be a useful
reproduction study, not a competitive AAAI contribution by itself. Renaming
entropy as effective support or adding a source-contract table does not clear
the collision.

The only remaining opening adjacent to these assets is a method that makes one
merged soft transition faithfully approximate multiple hard branches. That is
a different claim and is documented separately; it is not a rescue of
LRC-Bench.
