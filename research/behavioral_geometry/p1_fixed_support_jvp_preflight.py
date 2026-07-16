"""Fixed-support conditional-JVP repair preflight for P1.

The previous real-checkpoint JVP gate correctly failed because finite
perturbations changed the source sampler's discrete top-k support. This program
does not overwrite that result. It defines and tests a narrower, explicit
conditional derivative: the source no-noise soft-mixture recurrence conditional
on the top-k support recorded from one unperturbed natural trace.

The condition is auditable:
* at the reference trace, the conditional recurrence must reproduce the
  unmodified pinned sampler replay exactly;
* all finite differences use that same recorded support, never a selected
  favourable dynamic branch;
* the dynamic-support diagnostic is still recorded separately, and a pass here
  never claims a finite perturbation follows the released sampler unchanged.

Only a local checkpoint and a fixed synthetic marker are used. No benchmark,
calibration, held-out, reward, or training data is read.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

from research.behavioral_geometry.p1_official_sampler_replay import (
    replay_pinned_sampler_on_fake_request,
)
from research.behavioral_geometry.p1_source_sampler_contract import SourceLatentSamplerConfig


MODEL_DIR = Path("/mnt/e/LantentGRPO/_models/Latent-GRPO-Llama-1B")
SOURCE_END_ID = 524
MODEL_VOCAB_SIZE = 128256
PROMPT_TEXT = "0 0 0 <think>"
N_ACTIONS = 5
TOPK = 10
# This grid and the acceptance rule below are defined before this script is
# executed. It is a new conditional-branch engineering test, not a rewrite of
# the failed dynamic-support JVP at epsilon=0.01.
EPSILON_GRID = (0.20, 0.10, 0.05, 0.02, 0.01)
COARSE_ACCEPTANCE_EPSILONS = (0.10, 0.05)
MAX_RELATIVE_ERROR = 0.10
TARGET_SIZE = 257


@dataclass
class ConditionalJVPResult:
    status: str
    runtime: dict[str, Any]
    trace: dict[str, Any]
    conditional_identity: dict[str, Any]
    jvp: dict[str, Any]
    gpu: dict[str, Any]
    failures: list[str]


def _config() -> SourceLatentSamplerConfig:
    return SourceLatentSamplerConfig(
        max_topk=TOPK,
        top_p=1.0,
        temperature=1.0,
        gumbel_softmax_temperature=1.0,
        noise_scale=1.0,
        add_noise_gumbel_softmax=False,
        use_one_sided_gumbel_noise=True,
        latent_mode=True,
        latent_end_token_id=SOURCE_END_ID,
    )


def _step(model, past_key_values, embedding: torch.Tensor, position: int):
    return model(
        inputs_embeds=embedding.reshape(1, 1, -1),
        attention_mask=torch.ones((1, position + 1), dtype=torch.long, device=embedding.device),
        position_ids=torch.tensor([[position]], dtype=torch.long, device=embedding.device),
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=True,
    )


def _mixture(
    embedding_weight: torch.Tensor, logits: torch.Tensor, support_ids: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact no-noise source formula conditional on a recorded top-k support."""

    support = torch.tensor(support_ids, dtype=torch.long, device=logits.device)
    selected_logits = logits.float().gather(0, support)
    probs = torch.softmax(selected_logits, dim=0)
    embedding = torch.sum(
        probs.to(embedding_weight.dtype).unsqueeze(-1) * embedding_weight[support], dim=0
    )
    return embedding, probs


def _prefill(model, prompt_ids: tuple[int, ...]):
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
    with torch.inference_mode():
        return model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=True,
            return_dict=True,
        )


def _reference_trace(model, base_output, embedding_weight, prompt_length: int):
    """Record exactly N_ACTIONS source supports from a natural no-noise trace."""

    past = base_output.past_key_values
    logits = base_output.logits[0, -1]
    supports: list[list[int]] = []
    proxies: list[int] = []
    source_probs: list[torch.Tensor] = []
    endpoint = None
    with torch.inference_mode():
        for action_index in range(N_ACTIONS):
            proposal = replay_pinned_sampler_on_fake_request(logits, _config())
            support = [int(item) for item in proposal.topk_indices.detach().cpu().tolist()]
            supports.append(support)
            proxies.append(proposal.next_token_id)
            source_probs.append(proposal.topk_probs.detach())
            if proposal.next_token_id == SOURCE_END_ID:
                raise RuntimeError("synthetic reference trace exited before five actions")
            embedding = torch.sum(
                proposal.topk_probs.to(embedding_weight.dtype).unsqueeze(-1)
                * embedding_weight[proposal.topk_indices],
                dim=0,
            )
            endpoint = _step(model, past, embedding, prompt_length + action_index)
            past = endpoint.past_key_values
            logits = endpoint.logits[0, -1]
    assert endpoint is not None
    return endpoint, supports, proxies, source_probs


def _conditional_closure(
    model,
    base_output,
    embedding_weight: torch.Tensor,
    *,
    prompt_length: int,
    supports: list[list[int]],
    initial_embedding: torch.Tensor,
):
    """Consume exactly the recorded support sequence; no top-k is re-selected."""

    if len(supports) != N_ACTIONS:
        raise ValueError("fixed support trace has the wrong length")
    past = base_output.past_key_values
    embedding = initial_embedding
    output = None
    for action_index in range(N_ACTIONS):
        output = _step(model, past, embedding, prompt_length + action_index)
        past = output.past_key_values
        if action_index + 1 < N_ACTIONS:
            embedding, _ = _mixture(
                embedding_weight, output.logits[0, -1], supports[action_index + 1]
            )
    assert output is not None
    return output


def _dynamic_support_trace(
    model,
    base_output,
    embedding_weight: torch.Tensor,
    *,
    prompt_length: int,
    initial_embedding: torch.Tensor,
):
    """Diagnostic only: rerun the source branch under a finite perturbation."""

    past = base_output.past_key_values
    embedding = initial_embedding
    supports: list[list[int]] = []
    logits = base_output.logits[0, -1]
    output = None
    with torch.inference_mode():
        for action_index in range(N_ACTIONS):
            proposal = replay_pinned_sampler_on_fake_request(logits, _config())
            supports.append([int(item) for item in proposal.topk_indices.detach().cpu().tolist()])
            if action_index == 0:
                current_embedding = embedding
            else:
                if proposal.next_token_id == SOURCE_END_ID:
                    raise RuntimeError("dynamic diagnostic exited before five actions")
                current_embedding = torch.sum(
                    proposal.topk_probs.to(embedding_weight.dtype).unsqueeze(-1)
                    * embedding_weight[proposal.topk_indices],
                    dim=0,
                )
            output = _step(model, past, current_embedding, prompt_length + action_index)
            past = output.past_key_values
            logits = output.logits[0, -1]
    assert output is not None
    return supports


def _projection(logits: torch.Tensor) -> torch.Tensor:
    """A fixed 257-logit projection avoids a one-logit BF16 quantisation gate."""

    ids = (torch.arange(TARGET_SIZE, device=logits.device) * 131 + 17) % MODEL_VOCAB_SIZE
    weights = torch.cos(torch.arange(TARGET_SIZE, device=logits.device, dtype=torch.float32) * 0.37)
    weights = weights / weights.norm()
    return torch.dot(logits[ids].float(), weights)


def run_preflight() -> ConditionalJVPResult:
    failures: list[str] = []
    runtime = {
        "torch": torch.__version__,
        "transformers": transformers_version,
        "model_dir": str(MODEL_DIR),
        "cuda_available": torch.cuda.is_available(),
    }
    if not torch.cuda.is_available():
        return ConditionalJVPResult("BLOCKED_NO_CUDA", runtime, {}, {}, {}, {}, ["CUDA unavailable"])

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=True)
    prompt_ids = tuple([int(tokenizer.bos_token_id)] + tokenizer(PROMPT_TEXT, add_special_tokens=False)["input_ids"])
    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).to("cuda").eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        embedding_weight = model.get_input_embeddings().weight

        reference_base = _prefill(model, prompt_ids)
        dynamic_endpoint, supports, proxies, source_probs = _reference_trace(
            model, reference_base, embedding_weight, len(prompt_ids)
        )
        reconstructed_probs = []
        for support, source_prob in zip(supports, source_probs):
            # The logits are reconstructed at the same reference states below.
            reconstructed_probs.append(source_prob.float().detach().cpu().tolist())

        conditional_base = _prefill(model, prompt_ids)
        initial_embedding, initial_probs = _mixture(
            embedding_weight, conditional_base.logits[0, -1], supports[0]
        )
        conditional_endpoint = _conditional_closure(
            model,
            conditional_base,
            embedding_weight,
            prompt_length=len(prompt_ids),
            supports=supports,
            initial_embedding=initial_embedding,
        )
        endpoint_error = float(
            (dynamic_endpoint.logits.float() - conditional_endpoint.logits.float()).abs().max().item()
        )
        source_initial_error = float(
            (source_probs[0].float() - initial_probs.float()).abs().max().item()
        )
        if endpoint_error > 1e-5:
            failures.append("fixed-support recurrence does not reproduce source replay at reference")
        if source_initial_error > 1e-6:
            failures.append("conditional initial mixture differs from source no-noise sampler")

        grad_base = _prefill(model, prompt_ids)
        base_embedding, _ = _mixture(
            embedding_weight, grad_base.logits[0, -1], supports[0]
        )
        base_embedding = base_embedding.detach().clone().requires_grad_(True)
        grad_endpoint = _conditional_closure(
            model,
            grad_base,
            embedding_weight,
            prompt_length=len(prompt_ids),
            supports=supports,
            initial_embedding=base_embedding,
        )
        scalar = _projection(grad_endpoint.logits[0, -1])
        gradient = torch.autograd.grad(scalar, base_embedding, retain_graph=False)[0]
        direction = torch.tensor([1.0, -1.0, 0.5], dtype=base_embedding.dtype, device="cuda")
        direction = direction.repeat((base_embedding.numel() + 2) // 3)[: base_embedding.numel()]
        direction = direction / direction.float().norm().to(direction.dtype)
        analytic = float((gradient.float() * direction.float()).sum().item())

        rows = []
        for epsilon in EPSILON_GRID:
            plus_embedding = base_embedding.detach() + epsilon * direction
            minus_embedding = base_embedding.detach() - epsilon * direction
            plus_base = _prefill(model, prompt_ids)
            plus_endpoint = _conditional_closure(
                model,
                plus_base,
                embedding_weight,
                prompt_length=len(prompt_ids),
                supports=supports,
                initial_embedding=plus_embedding,
            )
            minus_base = _prefill(model, prompt_ids)
            minus_endpoint = _conditional_closure(
                model,
                minus_base,
                embedding_weight,
                prompt_length=len(prompt_ids),
                supports=supports,
                initial_embedding=minus_embedding,
            )
            finite_difference = float(
                ((_projection(plus_endpoint.logits[0, -1]) - _projection(minus_endpoint.logits[0, -1]))
                / (2 * epsilon)).item()
            )
            central_delta = (plus_embedding.float() - minus_embedding.float()) / 2
            effective_scale = float((central_delta * direction.float()).sum().item())
            relative_error = abs(analytic - finite_difference) / max(abs(analytic), 1e-3)

            plus_dynamic_base = _prefill(model, prompt_ids)
            plus_dynamic_supports = _dynamic_support_trace(
                model,
                plus_dynamic_base,
                embedding_weight,
                prompt_length=len(prompt_ids),
                initial_embedding=plus_embedding,
            )
            minus_dynamic_base = _prefill(model, prompt_ids)
            minus_dynamic_supports = _dynamic_support_trace(
                model,
                minus_dynamic_base,
                embedding_weight,
                prompt_length=len(prompt_ids),
                initial_embedding=minus_embedding,
            )
            rows.append(
                {
                    "epsilon": epsilon,
                    "finite_difference": finite_difference,
                    "abs_error": abs(analytic - finite_difference),
                    "relative_error": relative_error,
                    "effective_scale_over_requested": effective_scale / epsilon,
                    "plus_dynamic_support_matches_reference": plus_dynamic_supports == supports,
                    "minus_dynamic_support_matches_reference": minus_dynamic_supports == supports,
                }
            )

        coarse_rows = [row for row in rows if row["epsilon"] in COARSE_ACCEPTANCE_EPSILONS]
        coarse_pass = all(row["relative_error"] <= MAX_RELATIVE_ERROR for row in coarse_rows)
        if not coarse_pass:
            failures.append("conditional JVP does not satisfy the fixed coarse-grid error rule")

        trace = {
            "prompt_text": PROMPT_TEXT,
            "prompt_ids": list(prompt_ids),
            "n_actions": N_ACTIONS,
            "proxies": proxies,
            "supports": supports,
        }
        conditional_identity = {
            "reference_endpoint_max_abs_error": endpoint_error,
            "source_initial_mixture_max_abs_error": source_initial_error,
            "conditional_branch_definition": (
                "At each action, gather the reference source top-k ids from current logits, "
                "softmax those gathered logits, then form the embedding mixture. No top-k ids "
                "are reselected inside the conditional branch."
            ),
        }
        jvp = {
            "analytic": analytic,
            "target": "fixed 257-logit cosine projection",
            "epsilon_grid": list(EPSILON_GRID),
            "coarse_acceptance_epsilons": list(COARSE_ACCEPTANCE_EPSILONS),
            "max_relative_error": MAX_RELATIVE_ERROR,
            "rows": rows,
            "conditional_pass": coarse_pass,
            "interpretation": (
                "A conditional pass validates only the derivative within the recorded support "
                "branch. Dynamic-support columns remain a separate discontinuity diagnostic."
            ),
        }
        gpu = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "free_bytes_after_checks": int(torch.cuda.mem_get_info()[0]),
        }
        return ConditionalJVPResult(
            "PASS" if not failures else "FAIL",
            runtime,
            trace,
            conditional_identity,
            jvp,
            gpu,
            failures,
        )
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        return ConditionalJVPResult(
            "OOM", runtime, {}, {}, {}, {}, [f"conditional JVP OOM: {exc}"[:400]]
        )
    except Exception as exc:
        return ConditionalJVPResult(
            "ERROR", runtime, {}, {}, {}, {}, [f"{type(exc).__name__}: {exc}"[:500]]
        )
    finally:
        if model is not None:
            del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print(json.dumps(asdict(run_preflight()), ensure_ascii=False, indent=2, sort_keys=True))

