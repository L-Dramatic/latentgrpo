"""Source-equivalent one-step soft actions for the frozen PCMC checkpoints."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .checkpoint_protocol import CheckpointProfile


@dataclass(frozen=True)
class SourceAction:
    token_ids: Tensor
    weights: Tensor
    proxy_token_id: int
    sampler_metadata: dict[str, Any]


@dataclass(frozen=True)
class ConditionedAction:
    status: str
    token_ids: Tensor
    weights: Tensor
    structural_end_mass: float
    reason: str | None


def _top_k_renormalize(probabilities: Tensor, top_k: int) -> Tensor:
    if probabilities.ndim != 1 or top_k < 1 or top_k > probabilities.numel():
        raise ValueError("invalid top-k renormalization input")
    values, indices = torch.topk(probabilities, k=top_k)
    result = torch.zeros_like(probabilities)
    result.scatter_(0, indices, values)
    return result / result.sum()


def _top_p_renormalize(probabilities: Tensor, top_p: float) -> Tensor:
    if probabilities.ndim != 1 or not 0.0 < top_p <= 1.0:
        raise ValueError("invalid top-p renormalization input")
    sorted_probabilities, sorted_indices = torch.sort(
        probabilities, descending=True
    )
    cumulative = torch.cumsum(sorted_probabilities, dim=0)
    keep = (cumulative - sorted_probabilities) < top_p
    kept = sorted_probabilities * keep
    result = torch.zeros_like(probabilities)
    result.scatter_(0, sorted_indices, kept)
    total = result.sum()
    if not bool(torch.isfinite(total)) or float(total) <= 0.0:
        raise ValueError("top-p renormalization removed every token")
    return result / total


def _latent_grpo_action(logits: Tensor, profile: CheckpointProfile) -> SourceAction:
    from research.behavioral_geometry.p1_official_sampler_replay import (
        replay_pinned_sampler_on_fake_request,
    )
    from research.behavioral_geometry.p1_source_sampler_contract import (
        SourceLatentSamplerConfig,
    )

    sampler = profile.sampler
    replay = replay_pinned_sampler_on_fake_request(
        logits,
        SourceLatentSamplerConfig(
            max_topk=sampler.max_topk,
            top_p=sampler.top_p,
            temperature=sampler.temperature,
            gumbel_softmax_temperature=sampler.gumbel_softmax_temperature,
            noise_scale=sampler.noise_scale,
            add_noise_gumbel_softmax=True,
            use_one_sided_gumbel_noise=True,
            latent_mode=True,
            latent_end_token_id=int(profile.structural_end_token_id),
        ),
    )
    return SourceAction(
        token_ids=replay.topk_indices.detach().clone(),
        weights=replay.topk_probs.detach().float().clone(),
        proxy_token_id=replay.next_token_id,
        sampler_metadata={
            "adapter": profile.adapter,
            "raw_top1_token_id": int(replay.topk_original_indices[0].item()),
            "used_noisy_branch": bool(
                int(replay.topk_original_indices[0].item())
                != profile.structural_end_token_id
            ),
        },
    )


def _soft_grpo_action(logits: Tensor, profile: CheckpointProfile) -> SourceAction:
    sampler = profile.sampler
    probabilities = torch.softmax(logits.float() / sampler.temperature, dim=-1)
    probabilities = _top_k_renormalize(probabilities, sampler.top_k)
    probabilities = _top_p_renormalize(probabilities, sampler.top_p)
    topk_probabilities, topk_indices = torch.topk(
        probabilities, k=sampler.max_topk
    )
    topk_probabilities = topk_probabilities / topk_probabilities.sum()
    topk_logits = torch.log(topk_probabilities + 1e-6)
    gumbels = -torch.empty_like(topk_logits).exponential_().log()
    gumbels = gumbels.clamp(sampler.gumbel_clip_min, sampler.gumbel_clip_max)
    perturbed_logits = topk_logits + sampler.noise_scale * gumbels
    weights = torch.softmax(
        perturbed_logits / sampler.gumbel_softmax_temperature, dim=-1
    )
    weights, order = torch.sort(weights, descending=True)
    token_ids = topk_indices[order]
    return SourceAction(
        token_ids=token_ids.detach().clone(),
        weights=weights.detach().float().clone(),
        proxy_token_id=int(token_ids[0].item()),
        sampler_metadata={
            "adapter": profile.adapter,
            "pre_noise_topk_token_ids": [int(value) for value in topk_indices.tolist()],
            "pre_noise_topk_probabilities": [
                float(value) for value in topk_probabilities.tolist()
            ],
        },
    )


def sample_source_action(logits: Tensor, profile: CheckpointProfile) -> SourceAction:
    if logits.ndim != 1 or not logits.is_floating_point():
        raise ValueError("source action requires one floating-point logit vector")
    if profile.adapter == "official_latent_grpo_llama":
        return _latent_grpo_action(logits, profile)
    if profile.adapter == "soft_grpo_qwen":
        return _soft_grpo_action(logits, profile)
    raise ValueError(f"unsupported checkpoint adapter: {profile.adapter}")


def condition_on_content(
    action: SourceAction,
    profile: CheckpointProfile,
    *,
    minimum_content_support: int,
) -> ConditionedAction:
    token_ids = action.token_ids.long()
    weights = action.weights.float()
    if token_ids.ndim != 1 or weights.ndim != 1 or token_ids.shape != weights.shape:
        raise ValueError("source action ids and weights must be matching vectors")
    if not bool(torch.isfinite(weights).all()) or bool(torch.any(weights < 0.0)):
        raise ValueError("source action has invalid weights")
    total = weights.sum()
    if not bool(torch.isfinite(total)) or abs(float(total) - 1.0) > 1e-5:
        raise ValueError("source action weights are not normalized")
    structural_id = profile.structural_end_token_id
    structural_mask = token_ids == int(structural_id)
    structural_mass = float(weights[structural_mask].sum().item())
    content_mask = ~structural_mask
    content_ids = token_ids[content_mask]
    content_weights = weights[content_mask]
    positive = content_weights > 0.0
    content_ids = content_ids[positive]
    content_weights = content_weights[positive]
    if structural_mass > profile.maximum_structural_end_mass + 1e-12:
        return ConditionedAction(
            status="INELIGIBLE_CONTENT",
            token_ids=content_ids,
            weights=content_weights,
            structural_end_mass=structural_mass,
            reason="structural end mass exceeds frozen cap",
        )
    if content_ids.numel() < minimum_content_support:
        return ConditionedAction(
            status="INELIGIBLE_CONTENT",
            token_ids=content_ids,
            weights=content_weights,
            structural_end_mass=structural_mass,
            reason="content support is below frozen minimum",
        )
    content_total = content_weights.sum()
    if not bool(torch.isfinite(content_total)) or float(content_total) <= 0.0:
        raise ValueError("content action has no positive probability mass")
    return ConditionedAction(
        status="COMPLETE",
        token_ids=content_ids,
        weights=content_weights / content_total,
        structural_end_mass=structural_mass,
        reason=None,
    )


def weighted_embedding(
    embedding_table: Tensor, token_ids: Tensor, weights: Tensor
) -> Tensor:
    if token_ids.ndim != 1 or weights.ndim != 1 or token_ids.shape != weights.shape:
        raise ValueError("weighted embedding requires matching rank-one inputs")
    if bool(torch.any(token_ids < 0)) or bool(
        torch.any(token_ids >= embedding_table.shape[0])
    ):
        raise ValueError("weighted embedding received an invalid model token id")
    return torch.sum(
        weights.float().unsqueeze(-1) * embedding_table[token_ids], dim=0
    ).to(embedding_table.dtype)


def action_fingerprint(action: SourceAction) -> str:
    digest = hashlib.sha256()
    digest.update(int(action.proxy_token_id).to_bytes(8, "little", signed=True))
    digest.update(action.token_ids.detach().long().contiguous().cpu().numpy().tobytes())
    digest.update(action.weights.detach().float().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()
