# Behavioral Geometry Experiment Log

This log is append-oriented. Thresholds are frozen before each reported run,
and failed gates are not rewritten as passes.

## Superseding protocol note

The v1 smoke and population pilot continued generation after the model emitted
EOS. Those post-EOS logits do not belong to the terminated continuation
distribution and created an artificial tail. Their raw artifacts are preserved,
but neither run is valid behavioral evidence. All later runs treat EOS as an
absorbing termination and zero-pad only for aggregation.

## 2026-07-14: Public GPT-2 Coconut joint-KL integration smoke v1 (invalidated)

- Config: `research/behavioral_geometry/configs/public_gpt2_coconut_joint_kl_smoke_v1.json`
- Artifact: `artifacts/behavioral_geometry/public_gpt2_coconut_joint_kl_smoke_v1.json`
- Checkpoint: `connordilgren/gpt2-gsm8k-coconut`, `checkpoint_33`
- Status: **Invalidated after protocol audit**
- Evidence level: retained debugging record only

The reference policy sampled each continuation once. The candidate policy was
then teacher-forced on exactly those token histories, and its categorical KL
terms were summed according to the autoregressive chain rule. A reported
temperature of `0.8` was applied to the logits before KL computation, matching
the actual sampling distribution.

- An identical latent produced bitwise-identical logits, exact forced tokens,
  and total joint KL `0.0`.
- A deterministic norm-`0.1` perturbation produced mean six-step joint KL
  `1.3935e-7`, above the frozen sensitivity floor `1e-10`.
- A numerically stable float64 KL calculation measured first-step KL
  `2.8239e-21`; effectively all measured difference appeared in the tail.
- The candidate comparison used four rollout calls, 52 model forward calls,
  and 12 generated tokens.

Both sampled and forced continuations used one consistent full-recompute forward
path after recurrent latent construction. However, the fixed six-step horizon
continued beyond EOS, so the reported tail-only difference cannot support a
claim about valid model behavior. The raw artifact is mirrored as
`public_gpt2_coconut_joint_kl_smoke_v1_invalid_post_eos.json`.

## 2026-07-14: Public GPT-2 Coconut joint-KL integration smoke v2

- Config: `research/behavioral_geometry/configs/public_gpt2_coconut_joint_kl_smoke_v2.json`
- Artifact: `artifacts/behavioral_geometry/public_gpt2_coconut_joint_kl_smoke_v2.json`
- Status: **Pass**
- Evidence level: corrected real-checkpoint measurement contract only

EOS is an absorbing termination in v2. The identical-latent control again had
exact logits, exact forced tokens, and zero KL. For a norm-`0.5` perturbation,
the effective continuation had three valid steps, mean joint KL was
`4.2813e-9`, and first-step KL was `7.1907e-20`. This validates the corrected
instrument but remains a single-prompt integration check.

## 2026-07-14: Public GPT-2 Coconut BCG population pilot v1 (invalidated)

- Artifact: `artifacts/behavioral_geometry/public_gpt2_coconut_bcg_pilot_v1.json`
- Preserved copy: `artifacts/behavioral_geometry/public_gpt2_coconut_bcg_pilot_v1_invalid_post_eos.json`
- Status: **Invalid**

The run evaluated 72 candidates but included repeated post-EOS answer tokens.
It is excluded from every scientific comparison and gate decision.

## 2026-07-14: Public GPT-2 Coconut BCG population pilot v2

- Config: `research/behavioral_geometry/configs/public_gpt2_coconut_bcg_pilot_v2.json`
- Artifact: `artifacts/behavioral_geometry/public_gpt2_coconut_bcg_pilot_v2.json`
- Split: 12 GSM8K training examples, 72 candidates
- Status: **Exploratory calibration; not a gate pass**

After EOS correction, next-token KL still ranked full continuation KL poorly:
pooled Spearman `0.0887`, top-20% risk recall `0.20`, and hidden top-risk
fraction `0.4667`. Euclidean distance was stronger than next-token KL
(Spearman `0.6737`, recall `0.60`). The run used 168 rollout calls, 1,708 model
forward calls, and 230.6 CPU seconds.

## 2026-07-14: Post-hoc prefix and template audit of pilot v2

- Artifact: `artifacts/behavioral_geometry/public_gpt2_coconut_bcg_pilot_v2_posthoc_prefix_audit.json`
- Status: **Adverse confound found**
- Evidence level: post-hoc exploratory analysis

All 24 sampled reference continuations begin with the fixed delimiter `###`.
Consequently, H1 mostly measures sensitivity at a formatting token. Cumulative
H2 KL already predicts the full terminated continuation well (pooled Spearman
`0.8830`, top-risk recall `0.80`), and H3 is stronger (`0.9306`, recall `0.80`).
This result blocks a held-out H1-vs-full gate on the current checkpoint. Future
gates must include template-aware H2/H3 or semantic-boundary baselines and must
replicate on a model/task where the continuation contains a nontrivial suffix.
