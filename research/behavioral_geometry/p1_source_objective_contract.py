"""Pure CPU contract for the pinned source Gumbel-likelihood surrogate.

The code mirrors only ``logprobs_from_logits_topk_gumbel`` in the pinned VERL
source.  It is a preflight oracle for the source-likelihood term, including
the advantage-dependent straight-through gradient route.  It intentionally
does not stand in for the full PPO objective, optimizer update, or checkpoint
gradient required by P1 Family B.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def source_gumbel_likelihood_contract(
    logits: Tensor,
    rollout_topk_ids: Tensor,
    rollout_topk_gumbels: Tensor,
    labels: Tensor,
    *,
    temperature: float,
    advantages: Tensor | None = None,
) -> Tensor:
    """Mirror the source formula while retaining autograd for CPU contract tests."""

    if logits.ndim < 2:
        raise ValueError("logits must have a vocabulary dimension")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    batch_shape = logits.shape[:-1]
    vocab_size = logits.shape[-1]
    k_num = rollout_topk_ids.shape[-1]
    if rollout_topk_ids.shape != rollout_topk_gumbels.shape:
        raise ValueError("rollout ids and Gumbel scores must share shape")
    if rollout_topk_ids.shape[:-1] != batch_shape or labels.shape != batch_shape:
        raise ValueError("labels and rollout records must align with logits")

    flat_logits = logits.reshape(-1, vocab_size)
    flat_labels = labels.reshape(-1)
    flat_ids = rollout_topk_ids.reshape(-1, k_num)
    flat_gumbels = rollout_topk_gumbels.reshape(-1, k_num)
    safe_ids = flat_ids.masked_fill(flat_ids == -100, 0)
    full_log_probs = F.log_softmax(flat_logits.float(), dim=-1)
    selected_log_probs = full_log_probs.gather(-1, safe_ids)
    raw_diff = flat_gumbels.float() - selected_log_probs
    standard = -raw_diff - (-raw_diff).exp()

    if advantages is not None:
        flat_advantages = advantages.reshape(-1, 1)
        if flat_advantages.shape[0] != raw_diff.shape[0]:
            raise ValueError("advantages must align with the flattened rollout positions")
        expanded_advantages = flat_advantages.expand_as(raw_diff)
        need_flip = (expanded_advantages <= 0) & (raw_diff < 0)
        flipped_diff = -raw_diff
        flipped = -flipped_diff - (-flipped_diff).exp()
        flip_float = need_flip.float()
        gradient_proxy = (1.0 - flip_float) * standard + flip_float * flipped
        gumbel_value = standard.detach() + (gradient_proxy - gradient_proxy.detach())
    else:
        gumbel_value = standard

    gumbel_value = gumbel_value.sum(dim=-1).div(k_num).to(dtype=logits.dtype)
    is_standard_token = (flat_ids[:, 1:] == -100).all(dim=-1)
    batch_temperatures = torch.ones(
        (flat_logits.shape[0], 1), dtype=flat_logits.dtype, device=flat_logits.device
    )
    batch_temperatures[is_standard_token] = temperature
    answer_value = F.log_softmax(flat_logits / batch_temperatures, dim=-1).gather(
        -1, flat_labels.unsqueeze(-1)
    ).squeeze(-1)
    return torch.where(is_standard_token, answer_value, gumbel_value).view(*batch_shape)
