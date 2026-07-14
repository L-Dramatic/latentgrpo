from __future__ import annotations

import hashlib
import importlib.util
import copy
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
from torch import Tensor


PINNED_SWITCH_COMMIT = "d8d97cdc6276fcfa6e48f6a6b19ce472c7b87fcd"
PINNED_SWITCH_SOURCE_SHA256 = (
    "3bdd5e66076bbcab1c3e2ee600c16fad749839cda21616b3d094211ba4fa1b27"
)
SWITCH_BASE_MODEL_ID = "Qwen/Qwen3-8B"
SWITCH_BASE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
SWITCH_BASE_WEIGHT_SHA256 = {
    "model-00001-of-00005.safetensors": "31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f",
    "model-00002-of-00005.safetensors": "5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282",
    "model-00003-of-00005.safetensors": "c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836",
    "model-00004-of-00005.safetensors": "b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a",
    "model-00005-of-00005.safetensors": "20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff",
}
SWITCH_ADAPTER_ID = "LARK-Lab/SWITCH-Phase3-GRPO-LoRA-Qwen3-8B"
SWITCH_ADAPTER_REVISION = "246fee75d774c02a110ea8608ac841a916dd5d35"
SWITCH_ADAPTER_WEIGHT_SHA256 = (
    "0cdeafb628cdadd4c0fe21507ec7d61c98ace506d9bdad87775515de032a5e2c"
)
MAX_LATENT_PER_BLOCK = 256

SwitchLatentOperation = Callable[[Tensor, int, int], Tensor]


def _detach_nested_cache(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_detach_nested_cache(item) for item in value)
    if isinstance(value, list):
        return [_detach_nested_cache(item) for item in value]
    if isinstance(value, dict):
        return {key: _detach_nested_cache(item) for key, item in value.items()}
    return value


def _clone_nested_cache(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone_nested_cache(item) for item in value)
    if isinstance(value, list):
        return [_clone_nested_cache(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_nested_cache(item) for key, item in value.items()}
    return copy.deepcopy(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_commit(source_directory: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_directory), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def verify_pinned_switch_source(source_directory: str | Path) -> dict[str, str]:
    source_directory = Path(source_directory).resolve()
    source_file = source_directory / "src" / "model" / "coconut_swi_model.py"
    if not source_file.is_file():
        raise FileNotFoundError(f"missing pinned SWITCH source: {source_file}")
    commit = _source_commit(source_directory)
    if commit != PINNED_SWITCH_COMMIT:
        raise ValueError(
            f"SWITCH source commit is {commit}, expected {PINNED_SWITCH_COMMIT}"
        )
    source_sha256 = _sha256_file(source_file)
    if source_sha256 != PINNED_SWITCH_SOURCE_SHA256:
        raise ValueError(
            "SWITCH source SHA-256 mismatch: "
            f"got {source_sha256}, expected {PINNED_SWITCH_SOURCE_SHA256}"
        )
    return {"commit": commit, "source_sha256": source_sha256}


@contextmanager
def _temporary_sys_path(path: Path) -> Iterator[None]:
    value = str(path)
    sys.path.insert(0, value)
    try:
        yield
    finally:
        if sys.path and sys.path[0] == value:
            sys.path.pop(0)
        else:
            sys.path.remove(value)


def load_pinned_switch_types(source_directory: str | Path) -> tuple[type, type]:
    source_directory = Path(source_directory).resolve()
    verify_pinned_switch_source(source_directory)
    source_file = source_directory / "src" / "model" / "coconut_swi_model.py"
    specification = importlib.util.spec_from_file_location(
        "_latent_geometry_pinned_switch", source_file
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"could not load SWITCH source from {source_file}")
    module = importlib.util.module_from_spec(specification)
    with _temporary_sys_path(source_directory):
        specification.loader.exec_module(module)
    return module.CoconutSwiModel, module.SwiTokenConfig


@dataclass(frozen=True)
class SwitchLatentStep:
    block_index: int
    step_index: int
    absolute_position: int
    proposed_native: Tensor
    consumed_native: Tensor
    exit_logits: Tensor


@dataclass(frozen=True)
class SwitchAuditRun:
    output_ids: Tensor
    latent_info: tuple[dict[str, Any], ...]
    latent_steps: tuple[SwitchLatentStep, ...]
    visible_decision_logits: tuple[Tensor, ...]
    model_forward_calls: int


@dataclass(frozen=True)
class SwitchModelBundle:
    model: Any
    tokenizer: Any
    token_config: Any
    base_snapshot: Path
    adapter_snapshot: Path
    base_revision: str
    adapter_revision: str
    source_commit: str
    device: torch.device


def load_public_switch(
    *,
    source_directory: str | Path,
    cache_directory: str | Path,
    device: str | torch.device,
    attention_implementation: str = "eager",
) -> SwitchModelBundle:
    """Download, integrity-check, and load the paper-final SWITCH checkpoint."""

    try:
        from huggingface_hub import snapshot_download
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "SWITCH C2 dependencies are unavailable; install requirements-switch-c2.txt"
        ) from error

    source_directory = Path(source_directory).resolve()
    cache_directory = Path(cache_directory).resolve()
    source_record = verify_pinned_switch_source(source_directory)
    switch_type, config_type = load_pinned_switch_types(source_directory)
    base_snapshot = Path(
        snapshot_download(
            SWITCH_BASE_MODEL_ID,
            revision=SWITCH_BASE_REVISION,
            cache_dir=str(cache_directory),
        )
    ).resolve()
    adapter_snapshot = Path(
        snapshot_download(
            SWITCH_ADAPTER_ID,
            revision=SWITCH_ADAPTER_REVISION,
            cache_dir=str(cache_directory),
        )
    ).resolve()
    for filename, expected in SWITCH_BASE_WEIGHT_SHA256.items():
        actual = _sha256_file(base_snapshot / filename)
        if actual != expected:
            raise ValueError(
                f"base weight {filename} SHA-256 is {actual}, expected {expected}"
            )
    adapter_hash = _sha256_file(adapter_snapshot / "adapter_model.safetensors")
    if adapter_hash != SWITCH_ADAPTER_WEIGHT_SHA256:
        raise ValueError(
            f"adapter weight SHA-256 is {adapter_hash}, "
            f"expected {SWITCH_ADAPTER_WEIGHT_SHA256}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_snapshot,
        local_files_only=True,
        extra_special_tokens={},
    )
    expected_tokens = {"<swi>": 151669, "</swi>": 151670, "<latent>": 151671}
    if len(tokenizer) != 151672:
        raise ValueError(f"SWITCH tokenizer length is {len(tokenizer)}, expected 151672")
    for token, expected in expected_tokens.items():
        actual = int(tokenizer.convert_tokens_to_ids(token))
        if actual != expected:
            raise ValueError(f"token {token} has id {actual}, expected {expected}")
    if int(tokenizer.eos_token_id) != 151645:
        raise ValueError("SWITCH tokenizer EOS id does not match the release")

    resolved_device = torch.device(device)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_snapshot,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": str(resolved_device)},
        attn_implementation=attention_implementation,
        trust_remote_code=True,
    )
    base_model.resize_token_embeddings(len(tokenizer))
    peft_model = PeftModel.from_pretrained(
        base_model,
        adapter_snapshot,
        local_files_only=True,
        is_trainable=False,
    )
    peft_model.eval()
    for parameter in peft_model.parameters():
        parameter.requires_grad_(False)
    token_config = config_type(
        swi_start_id=expected_tokens["<swi>"],
        swi_end_id=expected_tokens["</swi>"],
        latent_id=expected_tokens["<latent>"],
        eos_token_id=int(tokenizer.eos_token_id),
    )
    switch_model = switch_type(peft_model, token_config)
    switch_model.eval()
    return SwitchModelBundle(
        model=switch_model,
        tokenizer=tokenizer,
        token_config=token_config,
        base_snapshot=base_snapshot,
        adapter_snapshot=adapter_snapshot,
        base_revision=SWITCH_BASE_REVISION,
        adapter_revision=SWITCH_ADAPTER_REVISION,
        source_commit=source_record["commit"],
        device=resolved_device,
    )


@dataclass(frozen=True)
class SwitchReplayPlan:
    prompt_input_ids: Tensor
    prompt_attention_mask: Tensor
    visible_prefix_ids: Tensor
    latent_steps: int
    visible_target_ids: Tensor
    visible_decision_start_index: int


def build_switch_prompt(tokenizer: Any, question: str, *, suffix: str) -> str:
    """Reproduce the prompt path in the pinned SWITCH latent evaluator."""

    messages = [{"role": "user", "content": str(question) + str(suffix)}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def build_first_block_replay_plan(
    run: SwitchAuditRun,
    *,
    prompt_length: int,
    swi_start_id: int,
    swi_end_id: int,
    eos_id: int,
    visible_horizon: int,
) -> SwitchReplayPlan:
    if not isinstance(visible_horizon, int) or visible_horizon < 1:
        raise ValueError("visible_horizon must be a positive integer")
    if not run.latent_info:
        raise ValueError("run has no latent block")
    generated = run.output_ids[0, prompt_length:].detach().cpu()
    end_index = int(run.latent_info[0]["position"])
    start_index = end_index - 1
    if start_index < 0:
        raise ValueError("first latent block has an invalid visible boundary")
    if int(generated[start_index]) != int(swi_start_id):
        raise ValueError("first latent block does not start at the recorded boundary")
    if int(generated[end_index]) != int(swi_end_id):
        raise ValueError("first latent block does not end at the recorded boundary")
    target = generated[end_index + 1 : end_index + 1 + visible_horizon]
    if target.numel() != visible_horizon:
        raise ValueError("run ended before the requested post-block horizon")
    forbidden = {int(swi_start_id), int(swi_end_id), int(eos_id)}
    if any(int(token) in forbidden for token in target):
        raise ValueError("post-block horizon contains a switch boundary or EOS")
    latent_steps = int(run.latent_info[0]["n_latent_steps"])
    if latent_steps < 1:
        raise ValueError("first latent block has no latent steps")
    prompt = run.output_ids[0, :prompt_length].detach().cpu()
    return SwitchReplayPlan(
        prompt_input_ids=prompt,
        prompt_attention_mask=torch.ones_like(prompt),
        visible_prefix_ids=generated[:start_index].clone(),
        latent_steps=latent_steps,
        visible_target_ids=target.clone(),
        visible_decision_start_index=start_index + 1,
    )


class SwitchDifferentiableReplay:
    """Replay one factual block while varying only its first latent input."""

    def __init__(self, switch_model: Any, plan: SwitchReplayPlan) -> None:
        self.model = switch_model
        self.plan = plan
        self.prefix_evaluations = 0
        self.logit_evaluations = 0
        self.model_forward_calls = 0
        self.visible_logit_tokens = 0
        self.latent_logit_steps = 0
        self.prefix_cache_builds = 0
        self.prefix_cache_hits = 0
        self._prefix_cache: tuple[Tensor, tuple[Any, ...], int] | None = None

    def reset_accounting(self) -> None:
        self.prefix_evaluations = 0
        self.logit_evaluations = 0
        self.model_forward_calls = 0
        self.visible_logit_tokens = 0
        self.latent_logit_steps = 0
        self.prefix_cache_builds = 0
        self.prefix_cache_hits = 0

    def clear_prefix_cache(self) -> None:
        self._prefix_cache = None

    @staticmethod
    def _snapshot_cache(past: Any) -> tuple[Any, ...]:
        to_legacy = getattr(past, "to_legacy_cache", None)
        cache_type = type(past)
        from_legacy = getattr(cache_type, "from_legacy_cache", None)
        if callable(to_legacy) and callable(from_legacy):
            return (
                "legacy_factory",
                cache_type,
                _detach_nested_cache(to_legacy()),
            )
        return ("nested_clone", _detach_nested_cache(past))

    @staticmethod
    def _restore_cache(snapshot: tuple[Any, ...]) -> Any:
        if snapshot[0] == "legacy_factory":
            _, cache_type, payload = snapshot
            return cache_type.from_legacy_cache(payload)
        if snapshot[0] == "nested_clone":
            return _clone_nested_cache(snapshot[1])
        raise ValueError("unknown prefix-cache snapshot type")

    def _compute_prefix(self, device: torch.device):
        prompt = self.plan.prompt_input_ids.to(device).view(1, -1)
        mask = self.plan.prompt_attention_mask.to(device).view(1, -1)
        prompt_len = int(prompt.shape[1])
        with torch.no_grad():
            outputs = self.model.base_causallm(
                inputs_embeds=self.model.embedding(prompt),
                attention_mask=mask,
                position_ids=torch.arange(prompt_len, device=device).view(1, -1),
                use_cache=True,
                output_hidden_states=False,
            )
            self.model_forward_calls += 1
            past = outputs.past_key_values
            cur_pos = prompt_len
            for token_id in self.plan.visible_prefix_ids.tolist():
                embedding = self.model.embedding(
                    torch.tensor([[token_id]], dtype=torch.long, device=device)
                )
                outputs = self.model.base_causallm(
                    inputs_embeds=embedding,
                    attention_mask=torch.ones((1, cur_pos + 1), device=device),
                    position_ids=torch.tensor([[cur_pos]], device=device),
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=False,
                )
                self.model_forward_calls += 1
                past = outputs.past_key_values
                cur_pos += 1
            swi_embedding = self.model.embedding(
                torch.tensor([[self.model.swi_start_id]], device=device)
            )
            outputs = self.model.base_causallm(
                inputs_embeds=swi_embedding,
                attention_mask=torch.ones((1, cur_pos + 1), device=device),
                position_ids=torch.tensor([[cur_pos]], device=device),
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
            )
            self.model_forward_calls += 1
            return (
                outputs.hidden_states[-1][0, -1, :].detach(),
                outputs.past_key_values,
                cur_pos + 1,
            )

    def _prefix(self, device: torch.device):
        self.prefix_evaluations += 1
        if self._prefix_cache is None or self._prefix_cache[0].device != device:
            factual, past, cur_pos = self._compute_prefix(device)
            self._prefix_cache = (
                factual.detach(),
                self._snapshot_cache(past),
                cur_pos,
            )
            self.prefix_cache_builds += 1
        else:
            self.prefix_cache_hits += 1
        factual, snapshot, cur_pos = self._prefix_cache
        return factual, self._restore_cache(snapshot), cur_pos

    def factual_latent(self) -> Tensor:
        device = self.model.embedding.weight.device
        latent, _, _ = self._prefix(device)
        return latent

    def _logits_after_prefix(
        self,
        candidate: Tensor,
        factual: Tensor,
        past: Any,
        cur_pos: int,
        visible_horizon: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        if candidate.ndim != 1 or not candidate.is_floating_point():
            raise ValueError("candidate must be a floating-point rank-one tensor")
        if not torch.isfinite(candidate).all():
            raise ValueError("candidate must be finite")
        if candidate.numel() != factual.numel():
            raise ValueError("candidate hidden dimension does not match the model")
        device = factual.device
        self.logit_evaluations += 1
        self.latent_logit_steps += int(self.plan.latent_steps)
        latent_input = candidate.to(device=device, dtype=factual.dtype).view(1, 1, -1)

        latent_logits: list[Tensor] = []
        outputs = None
        for step_index in range(self.plan.latent_steps):
            if step_index > 0:
                latent_input = outputs.hidden_states[-1][:, -1:, :]
            outputs = self.model.base_causallm(
                inputs_embeds=latent_input,
                attention_mask=torch.ones((1, cur_pos + 1), device=device),
                position_ids=torch.tensor([[cur_pos]], device=device),
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
            )
            self.model_forward_calls += 1
            past = outputs.past_key_values
            cur_pos += 1
            latent_logits.append(outputs.logits[0, -1, :])

        end_embedding = self.model.embedding(
            torch.tensor([[self.model.swi_end_id]], device=device)
        )
        outputs = self.model.base_causallm(
            inputs_embeds=end_embedding,
            attention_mask=torch.ones((1, cur_pos + 1), device=device),
            position_ids=torch.tensor([[cur_pos]], device=device),
            past_key_values=past,
            use_cache=True,
            output_hidden_states=False,
        )
        self.model_forward_calls += 1
        past = outputs.past_key_values
        cur_pos += 1
        visible_logits = [outputs.logits[0, -1, :]]
        target_ids = self.plan.visible_target_ids.to(device)
        if visible_horizon is not None:
            if not isinstance(visible_horizon, int) or not (
                1 <= visible_horizon <= target_ids.numel()
            ):
                raise ValueError("visible_horizon lies outside the replay target")
            target_ids = target_ids[:visible_horizon]
        self.visible_logit_tokens += int(target_ids.numel())
        for token_id in target_ids[:-1].tolist():
            token_embedding = self.model.embedding(
                torch.tensor([[token_id]], dtype=torch.long, device=device)
            )
            outputs = self.model.base_causallm(
                inputs_embeds=token_embedding,
                attention_mask=torch.ones((1, cur_pos + 1), device=device),
                position_ids=torch.tensor([[cur_pos]], device=device),
                past_key_values=past,
                use_cache=True,
                output_hidden_states=False,
            )
            self.model_forward_calls += 1
            past = outputs.past_key_values
            cur_pos += 1
            visible_logits.append(outputs.logits[0, -1, :])
        return torch.stack(latent_logits), torch.stack(visible_logits)

    def logits_from_candidate(
        self, candidate: Tensor, *, visible_horizon: int | None = None
    ) -> tuple[Tensor, Tensor]:
        device = self.model.embedding.weight.device
        factual, past, cur_pos = self._prefix(device)
        return self._logits_after_prefix(
            candidate,
            factual,
            past,
            cur_pos,
            visible_horizon=visible_horizon,
        )

    def logits(
        self,
        coefficients: Tensor,
        basis: Tensor,
        *,
        visible_horizon: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        if coefficients.ndim != 1:
            raise ValueError("coefficients must be rank one")
        if basis.ndim != 2 or basis.shape[1] != coefficients.numel():
            raise ValueError("basis must have shape (hidden, coefficient dimension)")
        device = self.model.embedding.weight.device
        factual, past, cur_pos = self._prefix(device)
        if basis.shape[0] != factual.numel():
            raise ValueError("basis hidden dimension does not match the model")
        candidate = factual.to(torch.float32) + basis.to(
            device=factual.device, dtype=torch.float32
        ) @ coefficients.to(device=factual.device, dtype=torch.float32)
        return self._logits_after_prefix(
            candidate,
            factual,
            past,
            cur_pos,
            visible_horizon=visible_horizon,
        )


class SwitchAuditRunner:
    """Source-equivalent greedy SWITCH loop with an auditable latent hook."""

    def __init__(self, switch_model: Any) -> None:
        required = (
            "base_causallm",
            "embedding",
            "swi_start_id",
            "swi_end_id",
            "eos_token_id",
        )
        missing = [name for name in required if not hasattr(switch_model, name)]
        if missing:
            raise TypeError(f"SWITCH model is missing attributes: {missing}")
        self.model = switch_model

    def run(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        max_new_tokens: int = 512,
        min_latent_steps: int = 0,
        operation: SwitchLatentOperation | None = None,
        capture_trace: bool = False,
        track_gradients: bool = False,
    ) -> SwitchAuditRun:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("SWITCH audit runner requires batch size one")
        if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if not isinstance(min_latent_steps, int) or min_latent_steps < 0:
            raise ValueError("min_latent_steps must be a non-negative integer")
        if min_latent_steps > MAX_LATENT_PER_BLOCK:
            raise ValueError("min_latent_steps exceeds the official safety cap")

        device = input_ids.device
        prompt_len = int(input_ids.shape[1])
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")

        generated_tokens: list[int] = []
        latent_info: list[dict[str, Any]] = []
        latent_steps: list[SwitchLatentStep] = []
        visible_decision_logits: list[Tensor] = []
        forward_calls = 0
        gradient_context = nullcontext() if track_gradients else torch.no_grad()

        with gradient_context:
            inputs_embeds = self.model.embedding(input_ids)
            position_ids = torch.arange(prompt_len, device=device).unsqueeze(0)
            outputs = self.model.base_causallm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                output_hidden_states=False,
            )
            forward_calls += 1
            kv_cache = outputs.past_key_values
            cur_pos = prompt_len
            next_logits = outputs.logits[0, -1]
            next_token = int(torch.argmax(next_logits).item())
            tokens_budget = max_new_tokens

            while tokens_budget > 0:
                if capture_trace:
                    visible_decision_logits.append(next_logits.detach().clone())
                if next_token == int(self.model.eos_token_id):
                    generated_tokens.append(next_token)
                    break

                if next_token == int(self.model.swi_start_id):
                    generated_tokens.append(next_token)
                    tokens_budget -= 1
                    swi_embed = self.model.embedding(
                        torch.tensor([[next_token]], device=device)
                    )
                    outputs = self.model.base_causallm(
                        inputs_embeds=swi_embed,
                        attention_mask=torch.ones((1, cur_pos + 1), device=device),
                        position_ids=torch.tensor([[cur_pos]], device=device),
                        past_key_values=kv_cache,
                        use_cache=True,
                        output_hidden_states=True,
                    )
                    forward_calls += 1
                    kv_cache = outputs.past_key_values
                    cur_pos += 1

                    block_index = len(latent_info)
                    n_latent_steps = 0
                    exited = False
                    for step_index in range(MAX_LATENT_PER_BLOCK):
                        proposed = outputs.hidden_states[-1][0, -1, :]
                        consumed = (
                            operation(proposed, block_index, step_index)
                            if operation is not None
                            else proposed
                        )
                        if (
                            not isinstance(consumed, Tensor)
                            or consumed.shape != proposed.shape
                            or not torch.isfinite(consumed).all()
                        ):
                            raise ValueError(
                                "latent operation must return a finite shape-matched tensor"
                            )
                        outputs = self.model.base_causallm(
                            inputs_embeds=consumed.view(1, 1, -1),
                            attention_mask=torch.ones((1, cur_pos + 1), device=device),
                            position_ids=torch.tensor([[cur_pos]], device=device),
                            past_key_values=kv_cache,
                            use_cache=True,
                            output_hidden_states=True,
                        )
                        forward_calls += 1
                        kv_cache = outputs.past_key_values
                        cur_pos += 1
                        n_latent_steps += 1
                        exit_logits = outputs.logits[0, -1]
                        if capture_trace:
                            latent_steps.append(
                                SwitchLatentStep(
                                    block_index=block_index,
                                    step_index=step_index,
                                    absolute_position=cur_pos - 1,
                                    proposed_native=proposed.detach().clone(),
                                    consumed_native=consumed.detach().clone(),
                                    exit_logits=exit_logits.detach().clone(),
                                )
                            )
                        if n_latent_steps >= min_latent_steps and int(
                            torch.argmax(exit_logits).item()
                        ) == int(self.model.swi_end_id):
                            exited = True
                            break

                    generated_tokens.append(int(self.model.swi_end_id))
                    tokens_budget -= 1
                    latent_info.append(
                        {
                            "position": len(generated_tokens) - 1,
                            "n_latent_steps": n_latent_steps,
                            "natural_exit": exited,
                        }
                    )
                    end_embed = self.model.embedding(
                        torch.tensor([[self.model.swi_end_id]], device=device)
                    )
                    outputs = self.model.base_causallm(
                        inputs_embeds=end_embed,
                        attention_mask=torch.ones((1, cur_pos + 1), device=device),
                        position_ids=torch.tensor([[cur_pos]], device=device),
                        past_key_values=kv_cache,
                        use_cache=True,
                        output_hidden_states=False,
                    )
                    forward_calls += 1
                    kv_cache = outputs.past_key_values
                    cur_pos += 1
                    next_logits = outputs.logits[0, -1]
                    next_token = int(torch.argmax(next_logits).item())
                    continue

                generated_tokens.append(next_token)
                tokens_budget -= 1
                token_embed = self.model.embedding(
                    torch.tensor([[next_token]], device=device)
                )
                outputs = self.model.base_causallm(
                    inputs_embeds=token_embed,
                    attention_mask=torch.ones((1, cur_pos + 1), device=device),
                    position_ids=torch.tensor([[cur_pos]], device=device),
                    past_key_values=kv_cache,
                    use_cache=True,
                    output_hidden_states=False,
                )
                forward_calls += 1
                kv_cache = outputs.past_key_values
                cur_pos += 1
                next_logits = outputs.logits[0, -1]
                next_token = int(torch.argmax(next_logits).item())

        output_ids = torch.tensor(
            input_ids[0].tolist() + generated_tokens,
            dtype=input_ids.dtype,
            device=device,
        ).view(1, -1)
        return SwitchAuditRun(
            output_ids=output_ids,
            latent_info=tuple(latent_info),
            latent_steps=tuple(latent_steps),
            visible_decision_logits=tuple(visible_decision_logits),
            model_forward_calls=forward_calls,
        )
