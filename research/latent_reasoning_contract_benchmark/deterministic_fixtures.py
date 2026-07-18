"""Execute pinned Coconut and CODI recurrence code on a deterministic toy LM."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import torch
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parents[2]
COCONUT_SOURCE = ROOT / "_external" / "coconut" / "coconut.py"
CODI_SOURCE = ROOT / "_external" / "codi" / "src" / "model.py"


@dataclass(frozen=True)
class DeterministicFixtureResult:
    method_id: str
    source_equivalent: bool
    replay_deterministic: bool
    reconstruction_max_abs: float
    output_max_abs: float
    source_output_sha256: str
    executed_latent_steps: int


class _ToyCachedCausalLM(nn.Module):
    """Small cache-aware recurrence with no learned or random state."""

    def __init__(self, *, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        with torch.no_grad():
            values = torch.arange(vocab_size * hidden_size, dtype=torch.float32)
            values = ((values % 17.0) - 8.0) / 13.0
            self.embedding.weight.copy_(values.reshape(vocab_size, hidden_size))
        projection = torch.arange(hidden_size * vocab_size, dtype=torch.float32)
        self.register_buffer(
            "output_projection",
            (((projection % 11.0) - 5.0) / 9.0).reshape(hidden_size, vocab_size),
        )
        self.config = SimpleNamespace(vocab_size=vocab_size, hidden_size=hidden_size)
        self.transformer = SimpleNamespace(wte=self.embedding)
        self.seen_input_embeddings: list[Tensor] = []

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    @staticmethod
    def transition(value: Tensor, previous: Tensor) -> Tensor:
        return torch.tanh(value + 0.25 * previous)

    def forward(
        self,
        *,
        input_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        past_key_values: Any = None,
        **_: Any,
    ) -> Any:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("toy LM requires exactly one input representation")
        values = self.embedding(input_ids) if input_ids is not None else inputs_embeds
        assert values is not None
        self.seen_input_embeddings.append(values.detach().clone())
        batch, _, hidden_size = values.shape
        if past_key_values is None:
            old_hidden = values.new_zeros((batch, 0, hidden_size))
        else:
            old_hidden = past_key_values[0][0].squeeze(1)
        previous = (
            old_hidden[:, -1, :]
            if old_hidden.shape[1]
            else values.new_zeros((batch, hidden_size))
        )
        current = []
        for position in range(values.shape[1]):
            previous = self.transition(values[:, position, :], previous)
            current.append(previous)
        hidden = torch.stack(current, dim=1)
        all_hidden = torch.cat((old_hidden, hidden), dim=1)
        cache = ((all_hidden.unsqueeze(1), all_hidden.unsqueeze(1).clone()),)
        logits = hidden @ self.output_projection
        return SimpleNamespace(
            logits=logits,
            hidden_states=(hidden,),
            past_key_values=cache,
        )


class _FixedProjection(nn.Module):
    def forward(self, value: Tensor) -> Tensor:
        return 0.8 * value + 0.05 * torch.flip(value, dims=(-1,))


def _tensor_sha256(value: Tensor) -> str:
    payload = value.detach().float().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _load_module(path: Path, name: str) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pinned source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _coconut_class() -> type[nn.Module]:
    return _load_module(COCONUT_SOURCE, "lrc_pinned_coconut_fixture").Coconut


@lru_cache(maxsize=1)
def _codi_class() -> type[nn.Module]:
    # CODI imports PEFT at module load, although the fixture bypasses model
    # construction and never enters a LoRA path. Stub only those import-time
    # names so the unmodified CODI.forward body remains the executed oracle.
    peft = ModuleType("peft")
    peft.get_peft_model = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    peft.PeftModel = object  # type: ignore[attr-defined]
    peft.PeftConfig = object  # type: ignore[attr-defined]
    previous = sys.modules.get("peft")
    try:
        sys.modules["peft"] = peft
        return _load_module(CODI_SOURCE, "lrc_pinned_codi_fixture").CODI
    finally:
        if previous is None:
            sys.modules.pop("peft", None)
        else:
            sys.modules["peft"] = previous


def _independent_coconut(
    model: _ToyCachedCausalLM, input_ids: Tensor, latent_token_id: int
) -> tuple[Tensor, Tensor]:
    executed = model.embedding(input_ids).detach().clone()
    previous = executed.new_zeros((executed.shape[0], executed.shape[-1]))
    hidden = []
    for position in range(executed.shape[1]):
        is_latent = input_ids[:, position] == latent_token_id
        executed[:, position, :] = torch.where(
            is_latent.unsqueeze(-1), previous, executed[:, position, :]
        )
        previous = model.transition(executed[:, position, :], previous)
        hidden.append(previous)
    hidden_tensor = torch.stack(hidden, dim=1)
    return executed, hidden_tensor @ model.output_projection


def _run_coconut_once() -> tuple[Tensor, Tensor, int]:
    model = _ToyCachedCausalLM(vocab_size=13, hidden_size=5)
    latent_token_id = 12
    input_ids = torch.tensor([[2, 5, latent_token_id, latent_token_id, 7, 3]])
    wrapper = _coconut_class()(
        model,
        latent_token_id=latent_token_id,
        start_latent_id=10,
        end_latent_id=11,
        eos_token_id=0,
    )
    output = wrapper(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=input_ids.clone(),
        position_ids=torch.arange(input_ids.shape[1]).unsqueeze(0),
    )
    expected_inputs, expected_logits = _independent_coconut(
        _ToyCachedCausalLM(vocab_size=13, hidden_size=5),
        input_ids,
        latent_token_id,
    )
    reconstruction_error = (output.inputs_embeds - expected_inputs).abs().max()
    output_error = (output.logits - expected_logits).abs().max()
    metrics = torch.stack((reconstruction_error, output_error))
    return metrics, output.logits, 2


def run_coconut_fixture(tolerance: float = 1e-6) -> DeterministicFixtureResult:
    first_metrics, first_output, latent_steps = _run_coconut_once()
    second_metrics, second_output, _ = _run_coconut_once()
    deterministic = torch.equal(first_output, second_output) and torch.equal(
        first_metrics, second_metrics
    )
    reconstruction = float(first_metrics[0])
    output_error = float(first_metrics[1])
    return DeterministicFixtureResult(
        method_id="coconut_gpt2",
        source_equivalent=reconstruction <= tolerance and output_error <= tolerance,
        replay_deterministic=bool(deterministic),
        reconstruction_max_abs=reconstruction,
        output_max_abs=output_error,
        source_output_sha256=_tensor_sha256(first_output),
        executed_latent_steps=latent_steps,
    )


def _new_codi_wrapper(model: _ToyCachedCausalLM) -> nn.Module:
    wrapper_type = _codi_class()
    wrapper = wrapper_type.__new__(wrapper_type)
    nn.Module.__init__(wrapper)
    wrapper.codi = model
    wrapper.use_prj = True
    wrapper.prj = _FixedProjection()
    wrapper.fix_attn_mask = False
    wrapper.num_latent = 2
    wrapper.model_name = "gpt2"
    wrapper.training_args = SimpleNamespace(print_ref_model_stats=False)
    wrapper.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    wrapper.distill_loss_fct = nn.SmoothL1Loss()
    wrapper.distill_loss_div_std = False
    wrapper.distill_loss_type = "smooth_l1"
    wrapper.distill_loss_factor = 1.0
    wrapper.ref_loss_factor = 1.0
    wrapper.print_loss = False
    return wrapper


def _independent_codi(
    model: _ToyCachedCausalLM,
    encoder_input_ids: Tensor,
    decoder_input_ids: Tensor,
) -> tuple[list[Tensor], Tensor]:
    projection = _FixedProjection()
    outputs = model(
        input_ids=encoder_input_ids, use_cache=True, output_hidden_states=True
    )
    cache = outputs.past_key_values
    latent = projection(outputs.hidden_states[-1][:, -1, :].unsqueeze(1))
    executed_latents = []
    for _ in range(2):
        executed_latents.append(latent.detach().clone())
        outputs = model(
            inputs_embeds=latent,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=cache,
        )
        cache = outputs.past_key_values
        latent = projection(outputs.hidden_states[-1][:, -1, :].unsqueeze(1))
    decoder_embeddings = model.embedding(decoder_input_ids)
    decoded = model(
        inputs_embeds=decoder_embeddings,
        use_cache=True,
        output_hidden_states=True,
        past_key_values=cache,
    )
    return executed_latents, decoded.logits


def _run_codi_once() -> tuple[Tensor, Tensor, int]:
    model = _ToyCachedCausalLM(vocab_size=17, hidden_size=6)
    wrapper = _new_codi_wrapper(model)
    encoder = torch.tensor([[2, 4, 7]])
    decoder = torch.tensor([[8, 6, 3, 1]])
    reference = torch.tensor([[2, 4, 9, 8, 6, 3, 1]])
    output = wrapper(
        encoder_input_ids=encoder,
        decoder_input_ids=decoder,
        ref_input_ids=reference,
        labels=decoder.clone(),
        encoder_attention_mask=torch.ones_like(encoder),
        ref_answer_position=torch.tensor([1]),
        model_answer_position=torch.tensor([1]),
        ref_attention_mask=torch.ones_like(reference),
        ref_labels=reference.clone(),
    )
    source_latents = [value for value in model.seen_input_embeddings if value.shape[1] == 1]

    oracle_model = _ToyCachedCausalLM(vocab_size=17, hidden_size=6)
    expected_latents, expected_logits = _independent_codi(
        oracle_model, encoder, decoder
    )
    if len(source_latents) != len(expected_latents):
        raise RuntimeError("CODI source fixture executed an unexpected number of latent steps")
    reconstruction_error = max(
        float((observed - expected).abs().max())
        for observed, expected in zip(source_latents, expected_latents)
    )
    output_error = float((output["logits"] - expected_logits).abs().max())
    return (
        torch.tensor([reconstruction_error, output_error]),
        output["logits"],
        len(source_latents),
    )


def run_codi_fixture(tolerance: float = 1e-6) -> DeterministicFixtureResult:
    first_metrics, first_output, latent_steps = _run_codi_once()
    second_metrics, second_output, _ = _run_codi_once()
    deterministic = torch.equal(first_output, second_output) and torch.equal(
        first_metrics, second_metrics
    )
    reconstruction = float(first_metrics[0])
    output_error = float(first_metrics[1])
    return DeterministicFixtureResult(
        method_id="codi_gpt2",
        source_equivalent=reconstruction <= tolerance and output_error <= tolerance,
        replay_deterministic=bool(deterministic),
        reconstruction_max_abs=reconstruction,
        output_max_abs=output_error,
        source_output_sha256=_tensor_sha256(first_output),
        executed_latent_steps=latent_steps,
    )
