"""CPU reimplementation of the pinned source's latent sampler contract.

This is deliberately a *math-level* contract, not an import of the serving
stack and not a checkpoint adapter.  It reproduces the sampler branch in the
pinned official source so that its top-p support, Gumbel perturbation, special
latent-end fallback, proxy, and soft-embedding rules can be tested on a small
CPU tensor before they are coupled to a real model.

The authoritative behavior is ``sglang/srt/layers/sampler.py`` in the pinned
official source.  In particular, its Gumbel selection uses raw-logit
probabilities, forces at least ``max_topk`` candidates through top-p, and
falls back to raw top-k when the *raw* top-1 token is the latent-end token.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SourceLatentSamplerConfig:
    """Scalar fake-request view of the source latent sampler parameters."""

    max_topk: int
    top_p: float = 1.0
    temperature: float = 1.0
    gumbel_softmax_temperature: float = 1.0
    noise_scale: float = 1.0
    add_noise_gumbel_softmax: bool = True
    use_one_sided_gumbel_noise: bool = True
    latent_mode: bool = True
    latent_end_token_id: int = 524

    def __post_init__(self) -> None:
        if self.max_topk < 1:
            raise ValueError("max_topk must be positive")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.gumbel_softmax_temperature <= 0.0:
            raise ValueError("gumbel_softmax_temperature must be positive")
        if self.noise_scale < 0.0:
            raise ValueError("noise_scale must be non-negative")
        if self.latent_end_token_id < 0:
            raise ValueError("latent_end_token_id must be non-negative")


@dataclass(frozen=True)
class SourceLatentSamplerResult:
    """Audit record for one source-style soft latent action."""

    proxy: int
    proposed_embedding: Tensor
    mixture_token_ids: Tensor
    mixture_probs: Tensor
    raw_topk_token_ids: Tensor
    raw_topk_probs: Tensor
    original_temperature_topk_probs: Tensor
    top_p_mask: Tensor
    gumbel_noise: Tensor | None
    selection_scores: Tensor | None
    used_noisy_branch: bool


def _validate_inputs(logits: Tensor, embedding_table: Tensor, config: SourceLatentSamplerConfig) -> None:
    if logits.ndim != 1:
        raise ValueError("fake source sampler accepts one rank-1 logit vector")
    if not logits.is_floating_point():
        raise TypeError("logits must be floating point")
    if embedding_table.ndim != 2 or not embedding_table.is_floating_point():
        raise TypeError("embedding_table must be a rank-2 floating tensor")
    if embedding_table.shape[0] != logits.numel():
        raise ValueError("embedding table vocabulary must match logits")
    if config.max_topk > logits.numel():
        raise ValueError("max_topk cannot exceed vocabulary size")
    if config.latent_end_token_id >= logits.numel():
        raise ValueError("latent_end_token_id is outside the fake vocabulary")


def _weighted_embedding(embedding_table: Tensor, token_ids: Tensor, probs: Tensor) -> Tensor:
    return probs.to(dtype=embedding_table.dtype) @ embedding_table[token_ids]


def source_style_latent_action(
    logits: Tensor,
    embedding_table: Tensor,
    config: SourceLatentSamplerConfig,
    *,
    generator: torch.Generator | None = None,
) -> SourceLatentSamplerResult:
    """Reproduce the pinned source sampler's latent-action selection branch.

    The result's ``proxy`` is the token appended to request output state.  Its
    ``proposed_embedding`` is the top-k mixture subsequently consumed unless a
    generic stop or the latent-end structural rule preempts it.  Those latter
    ordering rules are exercised separately by :mod:`p1_fake_preflight`.

    This function intentionally preserves two source-specific details that are
    easy to accidentally "clean up": the Gumbel selection law is based on
    raw logits (not ``temperature``), and the special 524 fallback is triggered
    by the raw top-1 token, rather than the perturbed proxy.
    """

    _validate_inputs(logits, embedding_table, config)
    logits_f32 = logits.float()
    raw_topk_logits, raw_topk_token_ids = torch.topk(logits_f32, k=config.max_topk)
    raw_topk_probs = torch.softmax(raw_topk_logits, dim=-1)
    temperature_probs = torch.softmax(logits_f32 / config.temperature, dim=-1)
    original_temperature_topk_probs = temperature_probs[raw_topk_token_ids]

    if not config.add_noise_gumbel_softmax:
        return SourceLatentSamplerResult(
            proxy=int(raw_topk_token_ids[0].item()),
            proposed_embedding=_weighted_embedding(
                embedding_table, raw_topk_token_ids, raw_topk_probs
            ),
            mixture_token_ids=raw_topk_token_ids,
            mixture_probs=raw_topk_probs,
            raw_topk_token_ids=raw_topk_token_ids,
            raw_topk_probs=raw_topk_probs,
            original_temperature_topk_probs=original_temperature_topk_probs,
            top_p_mask=torch.ones_like(logits_f32, dtype=torch.bool),
            gumbel_noise=None,
            selection_scores=None,
            used_noisy_branch=False,
        )

    full_log_probs = torch.log_softmax(logits_f32, dim=-1)
    sorted_probs, sorted_indices = torch.sort(torch.softmax(logits_f32, dim=-1), descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_mask = (cumulative_probs - sorted_probs) < config.top_p
    # Exact source rule: Top-P < Top-K still admits Top-K.
    sorted_mask[: config.max_topk] = True
    top_p_mask = torch.zeros_like(logits_f32, dtype=torch.bool)
    top_p_mask.scatter_(0, sorted_indices, sorted_mask)
    sampling_log_probs = full_log_probs.masked_fill(~top_p_mask, float("-inf"))

    gumbel_noise = -torch.empty_like(sampling_log_probs).exponential_(generator=generator).log()
    gumbel_noise = gumbel_noise.clamp(-1.5, 3.0)
    if config.use_one_sided_gumbel_noise:
        gumbel_noise = gumbel_noise - (-1.5)
    gumbel_noise = config.noise_scale * gumbel_noise
    selection_scores = sampling_log_probs + gumbel_noise
    noisy_scores, noisy_token_ids = torch.topk(selection_scores, k=config.max_topk)
    noisy_probs = torch.softmax(noisy_scores / config.gumbel_softmax_temperature, dim=-1)

    # Exact source fallback condition: latent mode AND a non-524 *raw* top-1
    # token are both required before the perturbed action may be used.
    use_noisy_branch = (
        config.latent_mode and int(raw_topk_token_ids[0].item()) != config.latent_end_token_id
    )
    mixture_token_ids = noisy_token_ids if use_noisy_branch else raw_topk_token_ids
    mixture_probs = noisy_probs if use_noisy_branch else raw_topk_probs
    return SourceLatentSamplerResult(
        proxy=int(mixture_token_ids[0].item()),
        proposed_embedding=_weighted_embedding(embedding_table, mixture_token_ids, mixture_probs),
        mixture_token_ids=mixture_token_ids,
        mixture_probs=mixture_probs,
        raw_topk_token_ids=raw_topk_token_ids,
        raw_topk_probs=raw_topk_probs,
        original_temperature_topk_probs=original_temperature_topk_probs,
        top_p_mask=top_p_mask,
        gumbel_noise=gumbel_noise,
        selection_scores=selection_scores,
        used_noisy_branch=use_noisy_branch,
    )
