"""Independent CPU formula for the pinned PPO policy-loss component."""

from __future__ import annotations

import torch
from torch import Tensor


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    return (values * mask).sum() / (mask.sum() + 1e-8)


def source_policy_loss_contract(
    old_log_prob: Tensor,
    log_prob: Tensor,
    advantages: Tensor,
    response_mask: Tensor,
    *,
    cliprange: float,
    cliprange_low: float,
    cliprange_high: float,
    clip_ratio_c: float = 3.0,
    neg_adv_weight: float = 1.0,
    loss_agg_mode: str = "token-mean",
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Match pinned ``compute_policy_loss`` for the P1 source-objective gate."""

    if old_log_prob.shape != log_prob.shape or log_prob.shape != advantages.shape:
        raise ValueError("old/current log-probs and advantages must share shape")
    if response_mask.shape != log_prob.shape:
        raise ValueError("response mask must match policy tensors")
    if clip_ratio_c <= 1.0:
        raise ValueError("clip_ratio_c must exceed one")
    if loss_agg_mode != "token-mean":
        raise ValueError("P1 binds the released token-mean aggregation")

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = _masked_mean(-negative_approx_kl, response_mask)
    losses1 = -advantages * ratio
    losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clipped1 = torch.maximum(losses1, losses2)
    clip_fraction = _masked_mean(torch.gt(losses2, losses1).float(), response_mask)
    losses3 = -advantages * clip_ratio_c
    clipped2 = torch.min(losses3, clipped1)
    lower_clip_fraction = _masked_mean(
        torch.gt(clipped1, losses3) * (advantages < 0).float(), response_mask
    )
    per_token_loss = torch.where(advantages < 0, clipped2 * neg_adv_weight, clipped1)
    return (
        _masked_mean(per_token_loss, response_mask),
        clip_fraction,
        ppo_kl,
        lower_clip_fraction,
    )
