"""Direct CPU replay of the pinned PPO policy-loss function.

Only the import-time ``verl.utils.torch_functional.masked_mean`` dependency is
stubbed, using its exact released formula.  The unmodified
``compute_policy_loss`` body from the pinned source then runs on small CPU
tensors.  This is a formula gate, not a training run.
"""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import torch
from torch import Tensor

from .p1_official_sampler_replay import PINNED_OFFICIAL_SOURCE_COMMIT


OFFICIAL_PPO_CORE_PATH = (
    Path(__file__).resolve().parents[2]
    / "_external"
    / "official_latent_grpo"
    / "verl-0.4.x"
    / "verl"
    / "trainer"
    / "ppo"
    / "core_algos.py"
)


def _package_module(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _stubbed_ppo_dependencies() -> dict[str, ModuleType]:
    modules: dict[str, ModuleType] = {
        "verl": _package_module("verl"),
        "verl.utils": _package_module("verl.utils"),
        "verl.trainer": _package_module("verl.trainer"),
        "verl.trainer.ppo": _package_module("verl.trainer.ppo"),
    }
    torch_functional = ModuleType("verl.utils.torch_functional")

    def masked_mean(values: Tensor, mask: Tensor, axis=None) -> Tensor:
        # Exact released ``verl.utils.torch_functional.masked_mean`` formula.
        return (values * mask).sum(axis=axis) / (mask.sum(axis=axis) + 1e-8)

    torch_functional.masked_mean = masked_mean
    modules[torch_functional.__name__] = torch_functional
    return modules


@lru_cache(maxsize=1)
def _load_pinned_ppo_core() -> ModuleType:
    if not OFFICIAL_PPO_CORE_PATH.is_file():
        raise FileNotFoundError(f"pinned PPO source missing: {OFFICIAL_PPO_CORE_PATH}")
    replacements = _stubbed_ppo_dependencies()
    previous = {name: sys.modules.get(name) for name in replacements}
    try:
        sys.modules.update(replacements)
        spec = importlib.util.spec_from_file_location("p1_pinned_official_ppo_core_cpu_replay", OFFICIAL_PPO_CORE_PATH)
        if spec is None or spec.loader is None:
            raise ImportError("could not construct a module spec for pinned core_algos.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


def replay_pinned_policy_loss(
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
    """Execute the pinned source PPO policy-loss body on CPU tensors."""

    return _load_pinned_ppo_core().compute_policy_loss(
        old_log_prob,
        log_prob,
        advantages,
        response_mask,
        cliprange=cliprange,
        cliprange_low=cliprange_low,
        cliprange_high=cliprange_high,
        clip_ratio_c=clip_ratio_c,
        neg_adv_weight=neg_adv_weight,
        loss_agg_mode=loss_agg_mode,
    )


def replay_pinned_include_advantage(
    token_level_rewards: Tensor,
    response_mask: Tensor,
    index,
    *,
    old_log_probs: Tensor | None,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
) -> tuple[Tensor, Tensor, dict]:
    """Run the exact released include-overlong first-mask winner routine."""

    return _load_pinned_ppo_core().compute_latent_grpo_outcome_advantage_firstmask_best_include_advantage(
        token_level_rewards,
        response_mask,
        index,
        old_log_probs=old_log_probs,
        epsilon=epsilon,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    )
