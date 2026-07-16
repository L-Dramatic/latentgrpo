"""Independent contract for the released GRPO first-mask winner routine."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch import Tensor


def source_include_advantage_contract(
    token_level_rewards: Tensor,
    response_mask: Tensor,
    index: np.ndarray,
    *,
    old_log_probs: Tensor | None,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
) -> tuple[Tensor, Tensor, dict]:
    """Mirror the pinned include-overlong advantage construction exactly."""

    if token_level_rewards.shape != response_mask.shape:
        raise ValueError("rewards and response mask must share shape")
    if old_log_probs is not None and old_log_probs.shape != response_mask.shape:
        raise ValueError("old log-probs and response mask must share shape")
    if len(index) != token_level_rewards.shape[0]:
        raise ValueError("one group index per sample is required")

    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean: dict[object, Tensor] = {}
    id2std: dict[object, Tensor] = {}
    with torch.no_grad():
        for row in range(scores.shape[0]):
            id2score[index[row]].append(scores[row])
        for group_id, group_scores in id2score.items():
            if len(group_scores) == 1:
                id2mean[group_id] = torch.tensor(0.0)
                id2std[group_id] = torch.tensor(1.0)
            elif len(group_scores) > 1:
                # Preserve the source's tensor(list-of-scalar-tensors) and its
                # default unbiased standard deviation rather than modernizing it.
                group_tensor = torch.tensor(group_scores)
                id2mean[group_id] = torch.mean(group_tensor)
                id2std[group_id] = torch.std(torch.tensor([group_scores]))
            else:
                raise ValueError(f"no score in prompt index: {group_id}")
        for row in range(scores.shape[0]):
            group_id = index[row]
            if norm_adv_by_std_in_grpo:
                scores[row] = (scores[row] - id2mean[group_id]) / (id2std[group_id] + epsilon)
            else:
                scores[row] = scores[row] - id2mean[group_id]
        scores = scores.unsqueeze(-1) * response_mask

        start_indices = response_mask.int().argmax(dim=1)
        group_rows = defaultdict(list)
        for row in range(scores.shape[0]):
            group_rows[index[row]].append(row)
        if old_log_probs is None:
            raise ValueError("contract requires old log-probs to avoid the source random fallback")
        lengths = response_mask.sum(dim=-1).float()
        mean_log_probs = (old_log_probs * response_mask).sum(dim=-1) / (lengths + 1e-8)
        for group_id, rows in group_rows.items():
            positive_rows = [
                row for row in rows if scores[row, start_indices[row]] > 0
            ]
            if positive_rows:
                # numpy.argmax selects the first maximum, matching the source.
                winner = positive_rows[int(np.argmax([mean_log_probs[row].item() for row in positive_rows]))]
                for row in rows:
                    if row != winner:
                        scores[row, start_indices[row]] = 0.0
    return scores, scores, id2std


def source_actor_zero_max_length_advantages(
    advantages: Tensor,
    attention_mask: Tensor,
    *,
    response_length: int,
    exclude_overlong_samples_from_advantage: bool,
) -> Tensor:
    """Mirror the released actor's in-place max-length zeroing branch."""

    if advantages.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("advantages and attention mask must be rank 2")
    if advantages.shape[0] != attention_mask.shape[0] or response_length < 1:
        raise ValueError("batch dimensions and response length must be valid")
    if not exclude_overlong_samples_from_advantage:
        current_response_length = attention_mask[:, -response_length:].sum(dim=-1)
        is_clipped = current_response_length == response_length
        advantages[is_clipped] = 0
    return advantages
