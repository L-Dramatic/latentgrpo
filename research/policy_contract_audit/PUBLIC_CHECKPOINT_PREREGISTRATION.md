# Public-Checkpoint Audit Preregistration

Frozen configuration: `configs/public_checkpoint_stage_a_v1.json`.
Canonical SHA-256:
`47536e7f9a9504ef958afa8b06280599025e6edbb0068e4cf5dfe7db97ebd629`.

## Objective

Determine whether the source-faithful LEPO contract distinctions survive real
Qwen2.5-3B logits on the complete MATH-500 evaluation set. This stage measures
policy geometry and gradient semantics, not benchmark accuracy.

## Fixed sources

- Model: `Qwen/Qwen2.5-3B` at
  `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`.
- Dataset: `HuggingFaceH4/MATH-500` at
  `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`.
- Prompt construction: released LEPO zero-shot MATH-500 prompt at commit
  `0ca191f7304bc435204568dcc8ccc6f9ba1d196d`.
- All 500 test rows are included; there is no result-dependent subsampling.

## State construction

For each prompt, record the model logits at the first latent position after the
source-faithful chat-formatted prompt. Apply TopK=30 and TopP=0.95 in the same
order as Transformers, then draw 32 Gumbel-Softmax actions for each temperature
in `{0.3, 0.5, 1.0}` and each of two fixed seeds.

## Controls

1. Archived probabilities must sum to one.
2. Scattering archived ids and probabilities must reconstruct the action.
3. The active support must not exceed 30.
4. Exact Concrete score and density-ratio identities are checked on a fixed
   support at every state. The logit perturbation is Gaussian with frozen RMS
   `0.05` and seed `714303`; it is used only for the fixed-support ratio control
   and the separately reported dynamic-support churn diagnostic.

Failure of a control invalidates the run and triggers code repair without
interpreting effect metrics.

## Effect metrics

1. Full-model probability mass excluded by TopK/TopP.
2. Frequency with which the Gumbel action mode differs from the proxy id.
3. Cosine and relative error between the released soft-label score and the
   exact fixed-support Concrete score.
4. Candidate-support churn under frozen, norm-calibrated logit perturbations.
5. Embedding-space separation between the executed mixture and proxy token,
   reported descriptively but not used as a Stage-A gate.

## Gate

All controls and at least four of five configured effect gates must pass before
LPCA spends compute on sequential latent rollouts or matched training. A failed
effect gate is retained as negative evidence. Thresholds and aggregation rules
cannot be changed after the first complete run.

## Interpretation limits

- Stage A cannot establish reward or accuracy harm.
- Exact fixed-support likelihood is a diagnostic control, not yet a proposed
  replacement objective.
- Proxy-mode disagreement is not itself a bug; its operational pathways are
  stopping, decoding, reward text, and history-sensitive processors.
- A pass authorizes Stage B; it does not validate the full paper thesis.

## Recorded outcome

The first complete run used the frozen configuration hash above and passed all
five controls and all five effect gates. The immutable summary is
`artifacts/policy_contract_audit/public_checkpoint_stage_a_v1.json` with
SHA-256
`2ae86696f037964137bea64ae338ab1f56b11d498a8b7248ce9ae62bab53e320`.
The 1,500 source records are in
`artifacts/policy_contract_audit/public_checkpoint_stage_a_v1_records.jsonl`
with SHA-256
`4f07ef4c69ed0c3e452e6572d046782f6848f82231f0163d9a6f6742e0c5548c`.
No threshold, aggregation rule, source revision, or inclusion rule was changed
after the run. Stage B is authorized; matched training remains unauthorized
until Stage B's separately frozen gate passes.
