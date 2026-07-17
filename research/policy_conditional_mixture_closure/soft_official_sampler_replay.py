"""Replay the unmodified pinned SofT sampler on a one-item fake CPU batch."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
from torch import Tensor, nn


PINNED_SOFT_SOURCE_COMMIT = "8d3c61380b15c3400818da5ce41c62c293a1bfb4"
OFFICIAL_SOFT_SAMPLER_PATH = (
    Path(__file__).resolve().parents[2]
    / "_external"
    / "soft_grpo"
    / "Soft-Thinking+noise+loss-main"
    / "sglang_soft_thinking_pkg"
    / "python"
    / "sglang"
    / "srt"
    / "layers"
    / "sampler.py"
)


@dataclass(frozen=True)
class SoftOfficialReplay:
    next_token_id: int
    token_ids: Tensor
    weights: Tensor
    perturbed_logits: Tensor


def _package(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _top_k_renorm(probabilities: Tensor, top_ks: Tensor) -> Tensor:
    result = torch.zeros_like(probabilities)
    for row in range(probabilities.shape[0]):
        top_k = int(top_ks[row].item())
        values, indices = torch.topk(probabilities[row], k=top_k)
        result[row].scatter_(0, indices, values)
    return result / result.sum(dim=-1, keepdim=True)


def _top_p_renorm(probabilities: Tensor, top_ps: Tensor) -> Tensor:
    result = torch.zeros_like(probabilities)
    for row in range(probabilities.shape[0]):
        values, indices = torch.sort(probabilities[row], descending=True)
        keep = (torch.cumsum(values, dim=0) - values) < float(top_ps[row])
        result[row].scatter_(0, indices, values * keep)
    return result / result.sum(dim=-1, keepdim=True)


def _stubbed_modules() -> dict[str, ModuleType]:
    modules = {
        "sglang": _package("sglang"),
        "sglang.srt": _package("sglang.srt"),
        "sglang.srt.layers": _package("sglang.srt.layers"),
        "sglang.srt.managers": _package("sglang.srt.managers"),
        "sglang.srt.sampling": _package("sglang.srt.sampling"),
    }
    distributed = ModuleType("sglang.srt.distributed")
    distributed.get_tensor_model_parallel_group = lambda: SimpleNamespace(
        device_group=None
    )
    modules[distributed.__name__] = distributed

    dp_attention = ModuleType("sglang.srt.layers.dp_attention")
    dp_attention.get_attention_tp_group = lambda: SimpleNamespace(device_group=None)
    modules[dp_attention.__name__] = dp_attention

    logits_processor = ModuleType("sglang.srt.layers.logits_processor")
    logits_processor.LogitsProcessorOutput = SimpleNamespace
    modules[logits_processor.__name__] = logits_processor

    schedule_batch = ModuleType("sglang.srt.managers.schedule_batch")
    schedule_batch.global_server_args_dict = {
        "enable_nan_detection": False,
        "enable_dp_attention": False,
        "sampling_backend": "flashinfer",
    }
    modules[schedule_batch.__name__] = schedule_batch

    sampling_batch_info = ModuleType("sglang.srt.sampling.sampling_batch_info")
    sampling_batch_info.SamplingBatchInfo = SimpleNamespace
    modules[sampling_batch_info.__name__] = sampling_batch_info

    utils = ModuleType("sglang.srt.utils")
    utils.crash_on_warnings = lambda: False
    utils.get_bool_env_var = lambda _name, default="false": default.lower() in {
        "true",
        "1",
    }
    utils.is_cuda = lambda: True
    modules[utils.__name__] = utils

    kernel = ModuleType("sgl_kernel")
    kernel.top_k_renorm_prob = _top_k_renorm
    kernel.top_p_renorm_prob = _top_p_renorm
    kernel.min_p_sampling_from_probs = lambda *_args, **_kwargs: None
    kernel.top_k_top_p_sampling_from_probs = lambda *_args, **_kwargs: None
    modules[kernel.__name__] = kernel
    return modules


@lru_cache(maxsize=1)
def _sampler_class() -> type[nn.Module]:
    if not OFFICIAL_SOFT_SAMPLER_PATH.is_file():
        raise FileNotFoundError(OFFICIAL_SOFT_SAMPLER_PATH)
    replacements = _stubbed_modules()
    previous = {name: sys.modules.get(name) for name in replacements}
    try:
        sys.modules.update(replacements)
        spec = importlib.util.spec_from_file_location(
            "pcmc_pinned_soft_sampler_replay", OFFICIAL_SOFT_SAMPLER_PATH
        )
        if spec is None or spec.loader is None:
            raise ImportError("could not load the pinned SofT sampler")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Sampler
    finally:
        for name, old_value in previous.items():
            if old_value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_value


def replay_pinned_soft_sampler(
    logits: Tensor,
    *,
    top_p: float,
    top_k: int,
    max_topk: int,
    temperature: float,
    gumbel_softmax_temperature: float,
    noise_scale: float,
) -> SoftOfficialReplay:
    if logits.ndim != 1 or not logits.is_floating_point():
        raise ValueError("replay requires one floating-point logit vector")
    sampler_type = _sampler_class()
    sampler = sampler_type.__new__(sampler_type)
    nn.Module.__init__(sampler)
    sampler.use_nan_detection = False  # type: ignore[attr-defined]
    sampler.tp_sync_group = None  # type: ignore[attr-defined]
    output = SimpleNamespace(next_token_logits=logits.float().unsqueeze(0).clone())
    device = logits.device
    info = SimpleNamespace(
        has_custom_logit_processor=False,
        is_all_greedy=False,
        grammars=(),
        device=logits.device,
        temperatures=torch.tensor([temperature], dtype=torch.float32, device=device),
        soft_thinking_modes=torch.tensor([True], device=device),
        top_ps=torch.tensor([top_p], dtype=torch.float32, device=device),
        top_ks=torch.tensor([top_k], dtype=torch.int64, device=device),
        after_thinking_top_ps=torch.tensor(
            [top_p], dtype=torch.float32, device=device
        ),
        after_thinking_top_ks=torch.tensor(
            [top_k], dtype=torch.int64, device=device
        ),
        min_ps=torch.tensor([0.0], dtype=torch.float32, device=device),
        after_thinking_min_ps=torch.tensor(
            [0.0], dtype=torch.float32, device=device
        ),
        dirichlet_alphas=torch.tensor([1.0], dtype=torch.float32, device=device),
        need_min_p_sampling=False,
        need_after_thinking_min_p_sampling=False,
        max_topk=max_topk,
        noise_factor=torch.tensor([noise_scale], dtype=torch.float32, device=device),
        gumbel_softmax_temperatures=torch.tensor(
            [gumbel_softmax_temperature], dtype=torch.float32, device=device
        ),
        noise_gumbel=True,
        noise_on_logits=True,
    )
    next_ids = sampler.forward(  # type: ignore[attr-defined]
        output,
        info,
        return_logprob=False,
        top_logprobs_nums=[],
        token_ids_logprobs=[],
        enable_soft_thinking=True,
        add_noise_dirichlet=False,
        add_noise_gumbel_softmax=True,
    )
    return SoftOfficialReplay(
        next_token_id=int(next_ids[0].item()),
        token_ids=output.topk_indices[0].clone(),
        weights=output.topk_probs[0].clone(),
        perturbed_logits=output.topk_gumbels[0].clone(),
    )
