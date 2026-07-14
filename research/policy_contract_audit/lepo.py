from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class LepoLatentStep:
    """Source-faithful objects produced by one LEPO latent rollout step."""

    full_action: Tensor
    archived_topk_probabilities: Tensor
    archived_topk_indices: Tensor
    proxy_token_id: Tensor
    executed_embedding: Tensor

    @property
    def archived_mass(self) -> Tensor:
        return self.archived_topk_probabilities.sum(dim=-1)


def _validate_logits_and_embeddings(logits: Tensor, embeddings: Tensor) -> None:
    if not isinstance(logits, Tensor) or not logits.is_floating_point():
        raise TypeError("logits must be a floating-point tensor")
    if not isinstance(embeddings, Tensor) or not embeddings.is_floating_point():
        raise TypeError("embeddings must be a floating-point tensor")
    if logits.ndim < 1 or embeddings.ndim != 2:
        raise ValueError("logits and embeddings have invalid ranks")
    if logits.shape[-1] != embeddings.shape[0]:
        raise ValueError("logits and embeddings must share the vocabulary size")
    if logits.dtype != embeddings.dtype or logits.device != embeddings.device:
        raise ValueError("logits and embeddings must share dtype and device")
    if not torch.isfinite(logits).all() or not torch.isfinite(embeddings).all():
        raise ValueError("logits and embeddings must be finite")


def apply_lepo_sampling_filters(
    logits: Tensor,
    *,
    top_k: int,
    top_p: float,
) -> Tensor:
    """Apply the TopK-then-TopP order used by Transformers generation."""

    if not isinstance(logits, Tensor) or not logits.is_floating_point():
        raise TypeError("logits must be a floating-point tensor")
    if logits.ndim < 1 or not torch.isfinite(logits).all():
        raise ValueError("logits must be finite and have a vocabulary dimension")
    if not 1 <= int(top_k) <= logits.shape[-1]:
        raise ValueError("top_k must fit within the vocabulary")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")

    topk_threshold = torch.topk(logits, k=int(top_k), dim=-1).values[..., -1:]
    filtered = logits.masked_fill(logits < topk_threshold, -torch.inf)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, dim=-1, descending=False)
        cumulative_probabilities = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        sorted_remove = cumulative_probabilities <= (1.0 - top_p)
        sorted_remove[..., -1:] = False
        remove = torch.zeros_like(sorted_remove).scatter(
            -1, sorted_indices, sorted_remove
        )
        filtered = filtered.masked_fill(remove, -torch.inf)
    return filtered


def replay_lepo_latent_step(
    logits: Tensor,
    embeddings: Tensor,
    gumbels: Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float = 0.95,
) -> LepoLatentStep:
    """Replay the released LEPO Gumbel-Softmax and top-k archive behavior.

    The released implementation first applies generation filters, executes the
    resulting Gumbel-Softmax vector, saves its top-k entries, and advances an
    auxiliary token-id stream with the clean filtered-logit argmax when
    ``do_latent_sample`` is false. Under the released default, the TopK filter
    bounds the nonzero support by the same ``top_k`` used for archiving.
    """

    _validate_logits_and_embeddings(logits, embeddings)
    if gumbels.shape != logits.shape or gumbels.dtype != logits.dtype:
        raise ValueError("gumbels must match logits")
    if gumbels.device != logits.device or not torch.isfinite(gumbels).all():
        raise ValueError("gumbels must be finite and on the logits device")
    if not 0 < int(top_k) <= logits.shape[-1]:
        raise ValueError("top_k must fit within the vocabulary")
    if not math_is_positive_finite(temperature):
        raise ValueError("temperature must be finite and positive")

    filtered_logits = apply_lepo_sampling_filters(
        logits, top_k=top_k, top_p=top_p
    )
    action = torch.softmax((filtered_logits + gumbels) / temperature, dim=-1)
    topk_probabilities, topk_indices = torch.topk(action, k=int(top_k), dim=-1)
    proxy_token_id = torch.argmax(filtered_logits, dim=-1)
    executed_embedding = action @ embeddings
    return LepoLatentStep(
        full_action=action,
        archived_topk_probabilities=topk_probabilities,
        archived_topk_indices=topk_indices,
        proxy_token_id=proxy_token_id,
        executed_embedding=executed_embedding,
    )


def math_is_positive_finite(value: float) -> bool:
    value_tensor = torch.as_tensor(value)
    return bool(torch.isfinite(value_tensor) and value_tensor > 0)


def lepo_soft_target_score(
    archived_topk_probabilities: Tensor,
    archived_topk_indices: Tensor,
    current_logits: Tensor,
) -> Tensor:
    """Gradient of the released latent soft-label objective w.r.t. logits.

    This is the gradient of ``sum_k z_k log softmax(logits)_k`` over the
    archived top-k entries. It is a surrogate score, not a continuous-action
    log-density score.
    """

    if archived_topk_probabilities.shape != archived_topk_indices.shape:
        raise ValueError("archived probabilities and indices must match")
    if archived_topk_indices.dtype != torch.long:
        raise TypeError("archived_topk_indices must use torch.long")
    if archived_topk_probabilities.ndim < 1:
        raise ValueError("archived targets need an action dimension")
    if archived_topk_probabilities.shape[:-1] != current_logits.shape[:-1]:
        raise ValueError("archive and logits batch dimensions must match")
    if (archived_topk_probabilities < 0).any():
        raise ValueError("archived probabilities must be nonnegative")
    if (archived_topk_indices < 0).any() or (
        archived_topk_indices >= current_logits.shape[-1]
    ).any():
        raise ValueError("archived indices are outside the vocabulary")

    sparse_target = torch.zeros_like(current_logits).scatter_add(
        -1, archived_topk_indices, archived_topk_probabilities
    )
    mass = archived_topk_probabilities.sum(dim=-1, keepdim=True)
    return sparse_target - mass * torch.softmax(current_logits, dim=-1)
