"""Direct CPU replay of the pinned source Gumbel-likelihood function.

The released VERL module cannot be conventionally imported in this Windows
preflight environment because its package initializer requires Ray.  This
harness runs the unmodified ``verl/utils/torch_functional.py`` file with only
unrelated import-time package symbols stubbed.  It forces the source's
Flash-Attention *formula branch* on CPU and replaces only the unreachable
hard-token cross-entropy kernel with an algebraically equivalent PyTorch
log-softmax gather.

This verifies the source Gumbel surrogate formula and its straight-through
backward convention.  It does not claim to reproduce the Flash-Attention CUDA
kernel, PPO aggregation/clipping, rewards, or an actual model gradient.
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


OFFICIAL_TORCH_FUNCTIONAL_PATH = (
    Path(__file__).resolve().parents[2]
    / "_external"
    / "official_latent_grpo"
    / "verl-0.4.x"
    / "verl"
    / "utils"
    / "torch_functional.py"
)


def _package_module(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _stubbed_objective_dependencies() -> dict[str, ModuleType]:
    """Supply only the imports needed to define the target source function."""

    modules: dict[str, ModuleType] = {
        "verl": _package_module("verl"),
        "verl.utils": _package_module("verl.utils"),
    }
    tensordict = ModuleType("tensordict")
    tensordict.TensorDict = object
    modules[tensordict.__name__] = tensordict

    transformers = ModuleType("transformers")
    transformers.PreTrainedTokenizer = object
    modules[transformers.__name__] = transformers

    device = ModuleType("verl.utils.device")
    device.get_torch_device = lambda: None
    device.get_device_name = lambda: "cpu"
    modules[device.__name__] = device
    return modules


def _cpu_answer_logprob(logits: Tensor, labels: Tensor, inplace_backward: bool = True) -> Tensor:
    """Algebraic CPU replacement for the only Flash kernel call in hard rows."""

    del inplace_backward
    return torch.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


@lru_cache(maxsize=1)
def _load_pinned_torch_functional_module() -> ModuleType:
    if not OFFICIAL_TORCH_FUNCTIONAL_PATH.is_file():
        raise FileNotFoundError(f"pinned torch_functional source missing: {OFFICIAL_TORCH_FUNCTIONAL_PATH}")
    replacements = _stubbed_objective_dependencies()
    previous = {name: sys.modules.get(name) for name in replacements}
    try:
        sys.modules.update(replacements)
        spec = importlib.util.spec_from_file_location(
            "p1_pinned_official_torch_functional_cpu_replay", OFFICIAL_TORCH_FUNCTIONAL_PATH
        )
        if spec is None or spec.loader is None:
            raise ImportError("could not construct a module spec for pinned torch_functional.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # The desired source branch is otherwise gated solely by a Flash-Attn
        # availability boolean.  Its math below uses regular torch operators.
        module.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = True
        module.logprobs_from_logits_flash_attn = _cpu_answer_logprob
        return module
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


def replay_pinned_gumbel_likelihood(
    logits: Tensor,
    rollout_topk_ids: Tensor,
    rollout_topk_gumbels: Tensor,
    labels: Tensor,
    *,
    top_p: float,
    temperature: float,
    advantages: Tensor | None = None,
) -> Tensor:
    """Execute the pinned source's Gumbel likelihood formula on CPU tensors."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    module = _load_pinned_torch_functional_module()
    return module.logprobs_from_logits_topk_gumbel(
        logits,
        rollout_topk_ids,
        rollout_topk_gumbels,
        labels,
        top_p,
        temperature,
        inplace_backward=False,
        advantages=advantages,
    )
