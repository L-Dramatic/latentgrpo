"""Narrow real-checkpoint identity preflight for P1.

This program intentionally uses only a synthetic marker prompt and the local
pinned checkpoint.  It performs no dataset access, calibration, risk scoring,
training, or parameter update.  Its outputs are a compatibility signal only;
they cannot establish the P1 phenomenon.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from research.behavioral_geometry.p1_official_sampler_replay import (
    replay_pinned_sampler_on_fake_request,
)
from research.behavioral_geometry.p1_source_sampler_contract import SourceLatentSamplerConfig


MODEL_DIR = Path(r"E:\LantentGRPO\_models\Latent-GRPO-Llama-1B")
SOURCE_END_ID = 524
FORBIDDEN_TOKENIZER_ONLY_ID = 128256
# This suite is fixed before inspection and contains no task/question, answer,
# calibration, or held-out text.  If a JVP-compatible trace exists, the first
# listed prompt with at least five natural latent actions is used; every probe
# trace is retained in the result so this engineering selection is auditable.
SYNTHETIC_PROMPT_TEXTS = (
    ("marker_only", "<think>"),
    ("zeros", "0 0 0 <think>"),
    ("alpha", "alpha <think>"),
    ("alpha_beta_gamma", "alpha beta gamma <think>"),
    ("synthetic_marker", "synthetic marker <think>"),
)
# This is a diagnostic grid, not a post-hoc acceptance search.  The primary
# check remains epsilon=0.01 and retains its fixed pass/fail rule below.  The
# complete ordered grid is retained to determine whether a bfloat16 input
# quantisation floor, rather than the recursive state machine, explains a
# failed finite-difference comparison.
FINITE_DIFFERENCE_EPSILONS = (0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001)


class _InsufficientLatentDepth(RuntimeError):
    """Internal control flow for a valid but too-short natural closure."""


@dataclass
class PreflightResult:
    status: str
    runtime: dict[str, Any]
    tokenizer: dict[str, Any]
    head: dict[str, Any]
    replay: dict[str, Any]
    cache: dict[str, Any]
    jvp: dict[str, Any]
    gpu: dict[str, Any]
    failures: list[str]


def _to_list(tensor: torch.Tensor) -> list:
    return tensor.detach().float().cpu().tolist()


def _cache_ptrs(past_key_values: Any) -> list[int]:
    if hasattr(past_key_values, "key_cache"):
        tensors = list(past_key_values.key_cache) + list(past_key_values.value_cache)
    elif isinstance(past_key_values, (tuple, list)):
        tensors = [tensor for layer in past_key_values for tensor in layer]
    else:
        return []
    return [int(tensor.data_ptr()) for tensor in tensors if isinstance(tensor, torch.Tensor)]


def _safe_embedding(weight: torch.Tensor, token_ids: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    if torch.any(token_ids < 0) or torch.any(token_ids >= weight.shape[0]):
        raise ValueError("source sampler emitted an out-of-vocabulary token id")
    return torch.sum(probs.to(weight.dtype).unsqueeze(-1) * weight[token_ids], dim=0)


def _sampler_config(*, noise: bool) -> SourceLatentSamplerConfig:
    return SourceLatentSamplerConfig(
        max_topk=10,
        top_p=1.0,
        temperature=1.0,
        gumbel_softmax_temperature=1.0,
        noise_scale=1.0,
        add_noise_gumbel_softmax=noise,
        use_one_sided_gumbel_noise=True,
        latent_mode=True,
        latent_end_token_id=SOURCE_END_ID,
    )


def _prefill(model, device: torch.device, prompt_ids: tuple[int, ...]):
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )


def _step(model, past_key_values: Any, embedding: torch.Tensor, position: int):
    device = embedding.device
    return model(
        inputs_embeds=embedding.reshape(1, 1, -1),
        attention_mask=torch.ones((1, position + 1), dtype=torch.long, device=device),
        position_ids=torch.tensor([[position]], dtype=torch.long, device=device),
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=True,
    )


def _deterministic_closure(
    model,
    base_output,
    embedding_weight: torch.Tensor,
    *,
    prompt_length: int,
    grad: bool,
    initial_embedding: torch.Tensor | None = None,
):
    """Consume one initial and four recomputed source-style latent mixtures."""

    position = prompt_length
    logits = base_output.logits[0, -1]
    if initial_embedding is None:
        proposal = replay_pinned_sampler_on_fake_request(logits, _sampler_config(noise=False))
        embedding = _safe_embedding(embedding_weight, proposal.topk_indices, proposal.topk_probs)
        proxies = [proposal.next_token_id]
    else:
        embedding = initial_embedding
        proposal = replay_pinned_sampler_on_fake_request(logits, _sampler_config(noise=False))
        proxies = [proposal.next_token_id]
    past = base_output.past_key_values
    supports: list[list[int]] = [[int(value) for value in proposal.topk_indices.detach().cpu().tolist()]]
    latest = None
    context = torch.enable_grad() if grad else torch.inference_mode()
    with context:
        for step_index in range(5):
            latest = _step(model, past, embedding, position + step_index)
            past = latest.past_key_values
            # Source semantics: the proposed mixture is overwritten by hard
            # E_524, that embedding is consumed by this step, and the next
            # model output is visible-token one.  It must never be sampled as a
            # further latent action.
            if proxies[-1] == SOURCE_END_ID:
                return latest, proxies, supports, _cache_ptrs(past), embedding, True
            if step_index == 4:
                break
            next_logits = latest.logits[0, -1]
            next_proposal = replay_pinned_sampler_on_fake_request(
                next_logits, _sampler_config(noise=False)
            )
            supports.append([int(value) for value in next_proposal.topk_indices.detach().cpu().tolist()])
            proxies.append(next_proposal.next_token_id)
            if next_proposal.next_token_id == SOURCE_END_ID:
                embedding = embedding_weight[SOURCE_END_ID]
            else:
                embedding = _safe_embedding(
                    embedding_weight, next_proposal.topk_indices, next_proposal.topk_probs
                )
    assert latest is not None
    return latest, proxies, supports, _cache_ptrs(past), embedding, False


def run_preflight() -> PreflightResult:
    from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

    failures: list[str] = []
    runtime = {
        "torch": torch.__version__,
        "transformers": transformers_version,
        "cuda_available": torch.cuda.is_available(),
        "model_dir": str(MODEL_DIR),
    }
    if not torch.cuda.is_available():
        return PreflightResult(
            "BLOCKED_NO_CUDA", runtime, {}, {}, {}, {}, {}, {}, ["CUDA is unavailable"]
        )
    device = torch.device("cuda:0")
    cuda_index = 0
    torch.cuda.set_device(cuda_index)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    runtime["gpu_name"] = torch.cuda.get_device_properties(device).name
    runtime["gpu_total_bytes"] = int(torch.cuda.get_device_properties(device).total_memory)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=True)
    marker_ids = tokenizer("</think>", add_special_tokens=False)["input_ids"]
    prompt_suite = [
        (name, tuple([tokenizer.bos_token_id] + tokenizer(text, add_special_tokens=False)["input_ids"]))
        for name, text in SYNTHETIC_PROMPT_TEXTS
    ]
    tokenizer_info = {
        "len_tokenizer": len(tokenizer),
        "model_vocab_size_expected": 128256,
        "marker_ids": marker_ids,
        "source_end_id": SOURCE_END_ID,
        "source_end_decode": tokenizer.decode([SOURCE_END_ID]),
        "eos_token_id": tokenizer.eos_token_id,
        "all_special_ids": tokenizer.all_special_ids,
        "forbidden_tokenizer_only_id": FORBIDDEN_TOKENIZER_ONLY_ID,
        "forbidden_id_decodes": tokenizer.decode([FORBIDDEN_TOKENIZER_ONLY_ID]),
        "synthetic_prompt_suite": [
            {"name": name, "token_ids": list(ids)} for name, ids in prompt_suite
        ],
    }
    if not marker_ids or marker_ids[0] != SOURCE_END_ID:
        failures.append("tokenizer </think> prefix does not begin with source id 524")
    if SOURCE_END_ID == tokenizer.eos_token_id or SOURCE_END_ID in tokenizer.all_special_ids:
        failures.append("source id 524 is a tokenizer EOS/special id")
    if any(token_id < 0 or token_id >= 128256 for _, ids in prompt_suite for token_id in ids):
        failures.append("synthetic prompt contains a tokenizer-only id")

    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        embedding_weight = model.get_input_embeddings().weight
        output_weight = model.get_output_embeddings().weight
        primary_name, primary_ids = prompt_suite[0]
        prefill_a = _prefill(model, device, primary_ids)
        prefill_b = _prefill(model, device, primary_ids)
        prefill_difference = float((prefill_a.logits - prefill_b.logits).abs().max().item())
        if prefill_difference != 0.0:
            failures.append("repeated synthetic checkpoint prefill is not bit-identical")

        noisy_seed = 20260716
        torch.manual_seed(noisy_seed)
        torch.cuda.manual_seed_all(noisy_seed)
        noisy_a = replay_pinned_sampler_on_fake_request(
            prefill_a.logits[0, -1], _sampler_config(noise=True)
        )
        torch.manual_seed(noisy_seed)
        torch.cuda.manual_seed_all(noisy_seed)
        noisy_b = replay_pinned_sampler_on_fake_request(
            prefill_a.logits[0, -1], _sampler_config(noise=True)
        )
        sampler_identical = (
            noisy_a.next_token_id == noisy_b.next_token_id
            and torch.equal(noisy_a.topk_indices, noisy_b.topk_indices)
            and torch.equal(noisy_a.topk_probs, noisy_b.topk_probs)
            and torch.equal(noisy_a.topk_gumbels, noisy_b.topk_gumbels)
        )
        if not sampler_identical:
            failures.append("fixed-seed source sampler replay is not identical")

        selected_ids = torch.unique(
            torch.cat(
                (
                    noisy_a.topk_indices.detach(),
                    torch.tensor([SOURCE_END_ID, tokenizer.eos_token_id], device=device),
                )
            )
        )
        hidden = prefill_a.hidden_states[-1][0, -1]
        manual_selected_logits = torch.matmul(output_weight[selected_ids], hidden)
        model_selected_logits = prefill_a.logits[0, -1, selected_ids]
        head_selected_error = float(
            (manual_selected_logits.float() - model_selected_logits.float()).abs().max().item()
        )
        head = {
            "input_shape": list(embedding_weight.shape),
            "output_shape": list(output_weight.shape),
            "tied_data_ptr": int(embedding_weight.data_ptr()) == int(output_weight.data_ptr()),
            "lm_head_has_bias": getattr(model.get_output_embeddings(), "bias", None) is not None,
            "selected_logit_max_abs_error": head_selected_error,
            "config_logit_scale_keys": [
                key for key in model.config.to_dict() if any(part in key.lower() for part in ("softcap", "logit_scale", "logits_scale"))
            ],
        }
        if tuple(embedding_weight.shape) != (128256, 2048) or tuple(output_weight.shape) != (128256, 2048):
            failures.append("real model embedding/head shape differs from pinned config")
        if not head["tied_data_ptr"]:
            failures.append("real model does not expose tied input/output embeddings")
        if head["lm_head_has_bias"]:
            failures.append("real model lm_head has an unmodeled bias")
        if head_selected_error > 1e-2:
            failures.append("checkpoint logits do not match selected direct head rows")

        ref_output, ref_proxies, ref_supports, ref_ptrs, _, ref_exited = _deterministic_closure(
            model, prefill_a, embedding_weight, prompt_length=len(primary_ids), grad=False
        )
        cand_output, cand_proxies, cand_supports, cand_ptrs, _, cand_exited = _deterministic_closure(
            model, prefill_b, embedding_weight, prompt_length=len(primary_ids), grad=False
        )
        closure_difference = float((ref_output.logits - cand_output.logits).abs().max().item())
        if closure_difference != 0.0 or ref_proxies != cand_proxies or ref_supports != cand_supports:
            failures.append("independent same-action recursive closures do not replay identically")
        if set(ref_ptrs) & set(cand_ptrs):
            failures.append("reference and candidate closures share cache storage")

        # Minimal four-later-action directional derivative.  Parameters remain
        # frozen, so only activation/cached-state memory is charged.  A failure
        # is recorded rather than retried with weaker semantics.
        jvp: dict[str, Any]
        try:
            depth_probes = []
            selected_depth_probe = None
            for probe_name, probe_ids in prompt_suite:
                probe_prefill = _prefill(model, device, probe_ids)
                _, probe_proxies, _, _, _, probe_exited = _deterministic_closure(
                    model,
                    probe_prefill,
                    embedding_weight,
                    prompt_length=len(probe_ids),
                    grad=False,
                )
                depth_probes.append(
                    {
                        "name": probe_name,
                        "latent_actions": len(probe_proxies),
                        "proxies": probe_proxies,
                        "exited_to_visible": probe_exited,
                    }
                )
                if selected_depth_probe is None and len(probe_proxies) >= 5:
                    selected_depth_probe = (probe_name, probe_ids)
            if selected_depth_probe is None:
                jvp = {"status": "INSUFFICIENT_LATENT_DEPTH", "probes": depth_probes}
                failures.append(
                    "no fixed synthetic marker has the required five-action natural latent closure"
                )
                raise _InsufficientLatentDepth
            selected_name, selected_ids = selected_depth_probe
            base_prefill = _prefill(model, device, selected_ids)
            base_proposal = replay_pinned_sampler_on_fake_request(
                base_prefill.logits[0, -1], _sampler_config(noise=False)
            )
            base_embedding = _safe_embedding(
                embedding_weight, base_proposal.topk_indices, base_proposal.topk_probs
            ).detach().clone().requires_grad_(True)
            grad_output, grad_proxies, grad_supports, _, _, grad_exited = _deterministic_closure(
                model,
                base_prefill,
                embedding_weight,
                prompt_length=len(selected_ids),
                grad=True,
                initial_embedding=base_embedding,
            )
            if len(grad_proxies) < 5:
                jvp = {
                    "status": "INSUFFICIENT_LATENT_DEPTH",
                    "selected_prompt": selected_name,
                    "probes": depth_probes,
                    "latent_actions": len(grad_proxies),
                    "exited_to_visible": grad_exited,
                    "proxies": grad_proxies,
                }
                failures.append(
                    "fixed synthetic marker exits before the required five-action JVP closure"
                )
                raise _InsufficientLatentDepth
            scalar = grad_output.logits[0, -1, 0].float()
            gradient = torch.autograd.grad(scalar, base_embedding, retain_graph=False)[0]
            direction = torch.tensor([1.0, -1.0, 0.5], device=device, dtype=base_embedding.dtype)
            direction = direction.repeat((base_embedding.numel() + 2) // 3)[: base_embedding.numel()]
            direction = direction / direction.float().norm().to(direction.dtype)
            analytic = float((gradient.float() * direction.float()).sum().item())
            finite_difference_grid = []
            primary: dict[str, Any] | None = None
            for epsilon in FINITE_DIFFERENCE_EPSILONS:
                plus_embedding = base_embedding.detach() + epsilon * direction
                minus_embedding = base_embedding.detach() - epsilon * direction
                plus_prefill = _prefill(model, device, selected_ids)
                plus, plus_proxies, plus_supports, _, _, plus_exited = _deterministic_closure(
                    model,
                    plus_prefill,
                    embedding_weight,
                    prompt_length=len(selected_ids),
                    grad=False,
                    initial_embedding=plus_embedding,
                )
                minus_prefill = _prefill(model, device, selected_ids)
                minus, minus_proxies, minus_supports, _, _, minus_exited = _deterministic_closure(
                    model,
                    minus_prefill,
                    embedding_weight,
                    prompt_length=len(selected_ids),
                    grad=False,
                    initial_embedding=minus_embedding,
                )
                finite_difference = float(
                    ((plus.logits[0, -1, 0].float() - minus.logits[0, -1, 0].float()) / (2 * epsilon)).item()
                )
                # Inputs to this bfloat16 checkpoint have already been
                # rounded when we arrive here.  Report the resulting central
                # direction and orthogonal residue, but never substitute this
                # diagnostic estimate for the fixed primary acceptance test.
                central_delta = (plus_embedding.float() - minus_embedding.float()) / 2
                effective_scale = float((central_delta * direction.float()).sum().item())
                orthogonal_residual = float(
                    (central_delta - effective_scale * direction.float()).norm().item()
                )
                finite_difference_effective = (
                    float(
                        ((plus.logits[0, -1, 0].float() - minus.logits[0, -1, 0].float())
                        / (2 * effective_scale)).item()
                    )
                    if abs(effective_scale) > 1e-12
                    else None
                )
                proxy_trace_stable = grad_proxies == plus_proxies == minus_proxies
                support_trace_stable = grad_supports == plus_supports == minus_supports
                exit_trace_stable = grad_exited == plus_exited == minus_exited
                trace_stable = proxy_trace_stable and support_trace_stable and exit_trace_stable
                row = {
                    "requested_epsilon": epsilon,
                    "finite_difference_requested": finite_difference,
                    "finite_difference_effective": finite_difference_effective,
                    "effective_central_scale": effective_scale,
                    "effective_scale_over_requested": effective_scale / epsilon,
                    "central_orthogonal_residual_norm": orthogonal_residual,
                    "trace_stable": trace_stable,
                    "proxy_trace_stable": proxy_trace_stable,
                    "support_trace_stable": support_trace_stable,
                    "exit_trace_stable": exit_trace_stable,
                    "plus_proxies": plus_proxies,
                    "minus_proxies": minus_proxies,
                    "plus_exited_to_visible": plus_exited,
                    "minus_exited_to_visible": minus_exited,
                    "abs_error_requested": abs(analytic - finite_difference),
                    "abs_error_effective": (
                        abs(analytic - finite_difference_effective)
                        if finite_difference_effective is not None
                        else None
                    ),
                }
                finite_difference_grid.append(row)
                if epsilon == 0.01:
                    primary = row
            assert primary is not None
            finite_difference = primary["finite_difference_requested"]
            trace_stable = primary["trace_stable"]
            jvp_error = primary["abs_error_requested"]
            jvp = {
                "status": "PASS" if trace_stable and jvp_error <= 0.1 else "FAIL",
                "selected_prompt": selected_name,
                "probes": depth_probes,
                "analytic": analytic,
                "finite_difference": finite_difference,
                "abs_error": jvp_error,
                "trace_stable": trace_stable,
                "primary_epsilon": 0.01,
                "finite_difference_grid": finite_difference_grid,
            }
            if jvp["status"] != "PASS":
                failures.append("four-later-action checkpoint directional derivative failed")
        except _InsufficientLatentDepth:
            pass
        except torch.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            jvp = {"status": "OOM", "detail": str(exc)[:300]}
            failures.append("8-GB checkpoint directional derivative ran out of memory")
        except Exception as exc:  # contract failures must remain visible
            jvp = {"status": "ERROR", "detail": f"{type(exc).__name__}: {exc}"[:500]}
            failures.append("checkpoint directional derivative could not execute")

        replay = {
            "prefill_repeat_max_abs_difference": prefill_difference,
            "fixed_seed_sampler_identical": sampler_identical,
            "noisy_proxy": noisy_a.next_token_id,
            "noisy_topk_ids": [int(value) for value in noisy_a.topk_indices.detach().cpu().tolist()],
            "noisy_topk_probs": _to_list(noisy_a.topk_probs),
            "noisy_scores": _to_list(noisy_a.topk_gumbels),
        }
        cache = {
            "independent_replay_max_abs_difference": closure_difference,
            "reference_proxies": ref_proxies,
            "candidate_proxies": cand_proxies,
            "cache_storage_disjoint": not bool(set(ref_ptrs) & set(cand_ptrs)),
            "reference_exited_to_visible": ref_exited,
            "candidate_exited_to_visible": cand_exited,
        }
        gpu = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "free_bytes_after_checks": int(torch.cuda.mem_get_info()[0]),
        }
        return PreflightResult(
            "PASS" if not failures else "FAIL", runtime, tokenizer_info, head, replay, cache, jvp, gpu, failures
        )
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        return PreflightResult(
            "OOM", runtime, tokenizer_info, {}, {}, {}, {}, {}, [f"model preflight OOM: {exc}"[:500]]
        )
    except Exception as exc:
        return PreflightResult(
            "ERROR", runtime, tokenizer_info, {}, {}, {}, {}, {}, [f"{type(exc).__name__}: {exc}"[:500]]
        )
    finally:
        if model is not None:
            del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    print(json.dumps(asdict(run_preflight()), ensure_ascii=False, indent=2, sort_keys=True))
