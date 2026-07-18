# Gate -1 Collision Matrix

Search cutoff: 2026-07-18. Primary paper, official project, and official source
pages were prioritized. Search phrases included latent reasoning benchmark,
mechanism audit, causal intervention, source-faithful, sampler/execution/
likelihood, policy contract, shortcut, supervision, and continuous CoT limits.

| Direct neighbor | Already covers | Fatal overlap | Residual boundary for LRC-Bench | Decision |
|---|---|---|---|---|
| [How Do Latent Reasoning Methods Perform Under Weak and Strong Supervision?](https://arxiv.org/abs/2602.22441) | Coconut, CODI, SIM-CoT, and CoLaR; shortcut behavior; BFS-like exploration; supervision tradeoff | Generic cross-method claim that latent computation is bypassed or not faithful search | Released-source conformance and stochastic latent-RL sampler/storage/objective semantics | Broad claim killed; narrow boundary open |
| [Observable Patterns Are Not Explanations](https://arxiv.org/abs/2606.12689) | Coconut/CODI matched controls, causal interventions, and causal geometry | Decodability or geometry as mechanism evidence; generic causal-use audit | Source-to-runtime contract and stochastic RL execution/measure agreement | Broad claim killed; narrow boundary open |
| [Formalizing Latent Thoughts: Four Axioms](https://arxiv.org/abs/2606.27378) | Representation-level Causality, Minimality, Separability, Stability across models/tasks | First axiomatic or intrinsic benchmark of latent representations | Implementation conformance and policy semantics; Four Axioms is a required baseline | Very high collision; differentiation required |
| [Do Latent-CoT Models Think Step-by-Step?](https://arxiv.org/abs/2602.00449) | Mechanistic CODI study with probes, attention, patching, partial paths, and late fusion | First sequential-faithfulness or CODI mechanism claim | Cross-release source contract and stochastic RL layer | Method-specific collision |
| [Limits of Continuous Chain-of-Thought](https://openreview.net/pdf?id=UQFTJPqJAc) | Controllable depth and multi-chain reasoning failures | First benchmark of depth or multi-path limits | Contract conformance rather than reasoning-depth capability | Benchmark-axis collision |
| [Interpreting Latent CoT Reasoning as Dynamical Systems](https://arxiv.org/abs/2607.09698) | Coconut/CODI trajectory dynamics and stability classes | Generic latent-dynamics interpretation claim | Released computation and learning contract; dynamics may be a baseline only | High adjacency |
| [Towards Inference-time Scaling for Continuous Space Reasoning](https://aclanthology.org/2026.findings-acl.1338/) | Coconut sampling, reranking, trajectory geometry, efficiency | Generic claim that latent trajectories lack useful discrimination | Training/runtime conformance, especially latent RL | Adjacent outcome study |
| [CODI](https://aclanthology.org/2025.emnlp-main.36/) and [Coconut](https://arxiv.org/abs/2412.06769) | The original deterministic mechanisms and reported task results | LRC-Bench cannot redefine their algorithms from a generic latent abstraction | Source files are normative evidence; deviations must be reported, not repaired silently | Audited targets |
| [Latent-GRPO](https://arxiv.org/abs/2604.27998) and [SofT-GRPO](https://arxiv.org/abs/2511.06411) | Released stochastic latent-RL mechanisms and acknowledged surrogates | Gumbel-softmax or generic surrogate mathematics is not novelty | Cross-release executable sampler/storage/execution/objective conformance | Audited targets |

## Collision decision

`NO-GO` for a broad latent-mechanism or shortcut benchmark.

`CONDITIONAL GO TO GATE -1` for the narrow source-faithful conformance
benchmark. No exact paper was found that jointly audits provenance, recurrence,
executed object, learning semantics, optional stochastic policy measure, and
outcome relevance across deterministic latent reasoning and stochastic latent
RL. This is a search result, not proof of novelty. A direct collision found
later triggers permanent KILL rather than a rename.
