# Literature Collision Audit

Audit date: 2026-07-16. Only primary paper pages are used below.

| Work | What it already establishes | Consequence for MMLPO |
|---|---|---|
| [Clipped Action Policy Gradient (ICML 2018)](https://arxiv.org/abs/1802.07564) | A clipped continuous action law is a mixture of an interior density and fixed endpoint Dirac masses; its policy-gradient estimator is unbiased and can be used with PPO/TRPO. | Direct collision with any generic "clipping-aware mixed measure" claim. The remaining distinction must be policy-dependent atom locations plus Top-K/proxy structure. |
| [Hybrid Policy Optimization (2026)](https://arxiv.org/abs/2605.14297) | An unbiased mixed score-function/pathwise gradient for hybrid discrete-continuous actions and piecewise-smooth systems. | Direct collision with a generic hybrid-gradient claim. HPO requires an exogenous differentiable simulator and explicitly excludes fully black-box simulators, so it does not directly solve terminal-reward LLM rollout. |
| [LEPO (2026)](https://arxiv.org/abs/2604.17892) | Gumbel-Softmax latent rollouts and a unified objective for latent distributions and discrete answer tokens. | Direct collision with broad claims about jointly optimizing continuous latent and discrete actions. LEPO uses a soft-label surrogate rather than the full mixed action likelihood. |
| [SofT-GRPO (2025)](https://arxiv.org/abs/2511.06411) | Gumbel-reparameterized soft-thinking policy optimization. | Blocks a generic Gumbel reparameterization contribution. |
| [Latent-GRPO (2026)](https://arxiv.org/abs/2604.27998) | The target sampler and surrogate, including latent-density and sampling-mechanism concerns. | MMLPO must improve this exact contract rather than restate its motivation. |
| [Reparameterization Gradient for Non-differentiable Models (NeurIPS 2018)](https://proceedings.neurips.cc/paper_files/paper/2018/hash/b096577e264d1ebd6b41041f392eec23-Abstract.html) | Unbiased reparameterization with explicit boundary/manifold correction for piecewise differentiable models. | Boundary correction is not new; applying it to vocabulary-scale autoregressive Top-K boundaries must overcome severe enumeration and reward-differentiability costs. |
| [Policy Gradient using Weak Derivatives (2020)](https://arxiv.org/abs/2004.04843) | An alternative unbiased policy-gradient theorem based on measure-valued derivatives. | Absence of an ordinary likelihood ratio does not imply absence of all gradients. A new method needs a specialized low-cost derivative decomposition. |
| [An Empirical Analysis of Measure-Valued Derivatives for Policy Gradients (2021)](https://arxiv.org/abs/2107.09359) | Measure-valued derivatives can work with non-differentiable value approximators. | A generic weak-derivative fallback is prior art and still requires policy-specific positive/negative measure constructions. |

## Collision verdict

The phenomenon-level statement is narrow but defensible: clipping the noise
before a policy-dependent score transform creates moving action atoms, unlike
CAPG's fixed executed-action endpoints. The method-level neighborhood is
crowded. Every generic repair is already represented by CAPG, mixed gradients,
boundary-corrected reparameterization, or weak derivatives.

To survive, MMLPO needed a new decomposition exploiting the exact sparse
Top-K latent structure while retaining black-box outcome rewards and near-GRPO
cost. No such decomposition survived the mathematical red team.
