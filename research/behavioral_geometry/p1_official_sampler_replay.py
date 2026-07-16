"""Execute the pinned official sampler on a fake CPU request.

Windows cannot import the complete SGLang serving stack in this workspace (the
stack imports Unix's :mod:`resource` module before the sampler is reached).
This harness deliberately stubs *only* the sampler's import-time framework
dependencies, then executes the unmodified pinned ``sampler.py`` file and its
real ``Sampler.forward`` body on a fake request.  It never loads a checkpoint,
tokenizer, CUDA kernel, or model.

The harness is a narrow replay oracle for the sampler branch, not a substitute
for the eventual source-equivalent model/recursive adapter.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import torch
from torch import Tensor, nn

from .p1_source_sampler_contract import SourceLatentSamplerConfig


PINNED_OFFICIAL_SOURCE_COMMIT = "c0994fb781a2d180662bb522d8ff3e8638dcf56d"
OFFICIAL_SAMPLER_PATH = (
    Path(__file__).resolve().parents[2]
    / "_external"
    / "official_latent_grpo"
    / "sglang_latent_reasoning_pkg"
    / "python"
    / "sglang"
    / "srt"
    / "layers"
    / "sampler.py"
)


@dataclass
class _FakeLogitsOutput:
    next_token_logits: Tensor


@dataclass
class _FakeSamplingInfo:
    add_noise_gumbel_softmax: Tensor
    top_ps: Tensor
    max_topk: int
    use_one_sided_gumbel_noise: Tensor
    noise_scales: Tensor
    gumbel_softmax_temperatures: Tensor
    latent_modes: Tensor
    temperatures: Tensor
    has_custom_logit_processor: bool = False
    is_all_greedy: bool = True
    grammars: tuple[object, ...] = ()


@dataclass(frozen=True)
class OfficialSamplerReplay:
    """Fields produced by the unmodified source ``Sampler.forward`` call."""

    next_token_id: int
    topk_indices: Tensor
    topk_probs: Tensor
    topk_gumbels: Tensor
    topk_original_indices: Tensor
    topk_original_probs: Tensor


def _package_module(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _stubbed_source_dependencies() -> dict[str, ModuleType]:
    """Create exactly the imported framework symbols used by ``sampler.py``."""

    modules: dict[str, ModuleType] = {
        "sglang": _package_module("sglang"),
        "sglang.srt": _package_module("sglang.srt"),
        "sglang.srt.layers": _package_module("sglang.srt.layers"),
        "sglang.srt.managers": _package_module("sglang.srt.managers"),
        "sglang.srt.sampling": _package_module("sglang.srt.sampling"),
    }

    distributed = ModuleType("sglang.srt.distributed")
    distributed.get_tensor_model_parallel_group = lambda: SimpleNamespace(device_group=None)
    modules[distributed.__name__] = distributed

    dp_attention = ModuleType("sglang.srt.layers.dp_attention")
    dp_attention.get_attention_tp_group = lambda: SimpleNamespace(device_group=None)
    modules[dp_attention.__name__] = dp_attention

    logits_processor = ModuleType("sglang.srt.layers.logits_processor")
    logits_processor.LogitsProcessorOutput = _FakeLogitsOutput
    modules[logits_processor.__name__] = logits_processor

    schedule_batch = ModuleType("sglang.srt.managers.schedule_batch")
    schedule_batch.global_server_args_dict = {
        "enable_nan_detection": False,
        "enable_dp_attention": False,
        "sampling_backend": "pytorch",
    }
    modules[schedule_batch.__name__] = schedule_batch

    sampling_batch_info = ModuleType("sglang.srt.sampling.sampling_batch_info")
    sampling_batch_info.SamplingBatchInfo = _FakeSamplingInfo
    modules[sampling_batch_info.__name__] = sampling_batch_info

    utils = ModuleType("sglang.srt.utils")
    utils.crash_on_warnings = lambda: False
    utils.get_bool_env_var = lambda _name, default="false": default.lower() in {"true", "1"}
    utils.is_cuda = lambda: False
    modules[utils.__name__] = utils
    return modules


@lru_cache(maxsize=1)
def _load_pinned_sampler_class() -> type[nn.Module]:
    """Load the original file after temporary, import-only framework stubbing."""

    if not OFFICIAL_SAMPLER_PATH.is_file():
        raise FileNotFoundError(f"pinned official sampler missing: {OFFICIAL_SAMPLER_PATH}")
    replacement_modules = _stubbed_source_dependencies()
    previous = {name: sys.modules.get(name) for name in replacement_modules}
    try:
        sys.modules.update(replacement_modules)
        spec = importlib.util.spec_from_file_location(
            "p1_pinned_official_sampler_cpu_replay", OFFICIAL_SAMPLER_PATH
        )
        if spec is None or spec.loader is None:
            raise ImportError("could not construct a module spec for pinned sampler")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Sampler
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


def _fake_sampler_instance(sampler_class: type[nn.Module]) -> nn.Module:
    """Avoid the serving-stack constructor; only ``forward`` is under test."""

    sampler = sampler_class.__new__(sampler_class)
    nn.Module.__init__(sampler)
    sampler.use_nan_detection = False  # type: ignore[attr-defined]
    sampler.tp_sync_group = None  # type: ignore[attr-defined]
    return sampler


def replay_pinned_sampler_on_fake_request(
    logits: Tensor, config: SourceLatentSamplerConfig
) -> OfficialSamplerReplay:
    """Run unmodified ``Sampler.forward`` on a one-item all-greedy fake batch."""

    if logits.ndim != 1 or not logits.is_floating_point():
        raise ValueError("logits must be one floating-point vector")
    if config.max_topk > logits.numel():
        raise ValueError("max_topk cannot exceed vocabulary size")
    if config.latent_end_token_id >= logits.numel():
        raise ValueError("latent_end_token_id is outside fake vocabulary")

    source_logits = logits.float().unsqueeze(0).clone()
    output = _FakeLogitsOutput(next_token_logits=source_logits)
    sampling_info = _FakeSamplingInfo(
        add_noise_gumbel_softmax=torch.tensor(
            [config.add_noise_gumbel_softmax], device=source_logits.device
        ),
        top_ps=torch.tensor(
            [config.top_p], dtype=source_logits.dtype, device=source_logits.device
        ),
        max_topk=config.max_topk,
        use_one_sided_gumbel_noise=torch.tensor(
            [config.use_one_sided_gumbel_noise], device=source_logits.device
        ),
        noise_scales=torch.tensor(
            [config.noise_scale], dtype=source_logits.dtype, device=source_logits.device
        ),
        gumbel_softmax_temperatures=torch.tensor(
            [config.gumbel_softmax_temperature],
            dtype=source_logits.dtype,
            device=source_logits.device,
        ),
        latent_modes=torch.tensor([config.latent_mode], device=source_logits.device),
        temperatures=torch.tensor(
            [config.temperature], dtype=source_logits.dtype, device=source_logits.device
        ),
    )
    sampler = _fake_sampler_instance(_load_pinned_sampler_class())
    next_ids = sampler.forward(  # type: ignore[attr-defined]
        output,
        sampling_info,
        return_logprob=False,
        top_logprobs_nums=[],
        token_ids_logprobs=[],
        enable_latent=True,
    )
    return OfficialSamplerReplay(
        next_token_id=int(next_ids[0].item()),
        # ``clone`` gives the caller ownership of the records while retaining
        # the autograd graph.  The latter is needed by the fake recursive JVP
        # gate; callers requiring a detached audit artifact must detach it.
        topk_indices=output.topk_indices[0].clone(),
        topk_probs=output.topk_probs[0].clone(),
        topk_gumbels=output.topk_gumbels[0].clone(),
        topk_original_indices=output.topk_original_indices[0].clone(),
        topk_original_probs=output.topk_original_probs[0].clone(),
    )
