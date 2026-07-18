# Gate -1 Method Matrix

| Method | Mechanism family | Recurrence/execution | Learning semantics | Policy measure | Checkpoint status | Current runtime evidence |
|---|---|---|---|---|---|---|
| Coconut | Deterministic hidden-state recurrence | Last hidden state replaces later latent input embeddings | Curriculum progressively replaces explicit thoughts | `N/A` | Public independent reproduction, pinned and local | Six-step checkpoint replay and prior intervention assets exist |
| CODI | Deterministic self-distilled recurrence | Last hidden state, optionally projected, is recursively executed with cache | Student CE + hidden-state distillation + teacher CE | `N/A` | Official public revision pinned, downloaded, and hash verified | Source path pinned; runtime integration pending |
| Latent-GRPO | Stochastic post-noise vocabulary mixture RL | Dynamic candidate construction, clipped Gumbels, perturbed top-k mixture embedding | GRPO with released latent surrogate and custom backward semantics | Required | Official 1B checkpoint pinned, hashed, and local | Native load and source-sampler replay passed; previous proposed repairs failed |
| SofT-GRPO | Stochastic truncated vocabulary mixture RL | Fixed top-k renormalization, Gumbel perturbation, soft mixture embedding | GRPO with reconstructed auxiliary action and component aggregation | Required | Official 1.5B subfolder pinned, hashed, and local | Native load and source-sampler replay passed; action is often near one-hot |

## Why these four

They provide two deterministic and two stochastic methods while keeping the
smallest available checkpoints near GPT-2 to 1.5B scale. The panel is not a
leaderboard. It is designed to test whether one typed audit can remain faithful
to materially different mechanisms without inventing a false universal action
space.

## Known asymmetries

- Coconut's public local checkpoint is an independent reproduction, not an
  original-author release. Provenance must remain visible in all tables.
- CODI's official checkpoint is locally hash verified but still needs native-load replay.
- Latent-GRPO and SofT-GRPO expose a stochastic action/measure layer absent from
  the deterministic methods.
- Each method uses its native task and prompt. Only standardized within-method
  effects may be pooled.
