# PCMC Literature Collision Audit

Audit date: 2026-07-18

## Direct neighbors

- [Soft Thinking](https://arxiv.org/abs/2505.15778) already executes a
  probability-weighted token-embedding mixture as a continuous concept.
- [Soft Concept Mixing](https://arxiv.org/abs/2511.16885) exposes the model to
  probability-weighted soft concepts during training, mixes them with
  contextual hidden states, and optimizes the latent process with RL.
- [Mixup Regularization: A Probabilistic Perspective](https://proceedings.mlr.press/v286/el-laham25a.html)
  formalizes probabilistic fusion for conditional densities and extends it to
  intermediate neural representations.
- [Policy Distillation](https://arxiv.org/abs/1511.06295) establishes the broad
  pattern of compressing one or multiple teacher policies into one student.
- [Self-Distilled Reasoner](https://arxiv.org/abs/2601.18734) demonstrates
  on-policy same-model distribution matching for LLM reasoning, making generic
  self-distillation an especially strong collision.

## Remaining possible contribution

The KL loss, mixture target, set adapter, and single-student compression are not
individually new. A defensible independent contribution would need all of:

- a natural, replicated causal failure of arithmetic soft execution;
- a policy-induced hard-branch mixture target that predicts this failure better
  than entropy or sharpness diagnostics;
- posterior-aware sequence closure rather than only one-step matching;
- a one-forward constrained barycenter that captures a substantial oracle gain;
- final reasoning improvement over hard sampling, sharpening, SCM exposure,
  and same-parameter distillation at matched compute.

This is a high evidentiary bar. Until those facts are observed, PCMC is a
mechanism hypothesis rather than an AAAI method.

