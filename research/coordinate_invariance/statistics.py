from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class Estimate:
    value: float
    ci_low: float
    ci_high: float
    count: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "value": self.value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "count": self.count,
        }


def bootstrap_mean(
    values: Sequence[float] | torch.Tensor,
    *,
    bootstrap_samples: int,
    seed: int,
    confidence: float = 0.95,
) -> Estimate:
    tensor = torch.as_tensor(values, dtype=torch.float64).flatten()
    if tensor.numel() < 1:
        raise ValueError("bootstrap requires at least one value")
    if not torch.isfinite(tensor).all():
        raise ValueError("bootstrap values must be finite")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(
        tensor.numel(),
        (bootstrap_samples, tensor.numel()),
        generator=generator,
    )
    sampled_means = tensor[indices].mean(dim=1)
    tail = (1.0 - confidence) / 2.0
    quantiles = torch.quantile(
        sampled_means, torch.tensor([tail, 1.0 - tail], dtype=torch.float64)
    )
    return Estimate(
        value=float(tensor.mean()),
        ci_low=float(quantiles[0]),
        ci_high=float(quantiles[1]),
        count=int(tensor.numel()),
    )

