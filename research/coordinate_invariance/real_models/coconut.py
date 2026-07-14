from __future__ import annotations

import hashlib
import importlib.util
import subprocess
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from torch import Tensor

from ..charts import LatentChart
from ..evaluator import ContinuationOutput
from ..rng import make_generator, seeded_rng
from ..trace import LatentStepRecord, LatentTrace


PINNED_COCONUT_COMMIT = "27273cb8cca4bb763c041a63b036d0c3b7cbbb48"
DEFAULT_MODEL_ID = "openai-community/gpt2"
DEFAULT_MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
DEFAULT_CHECKPOINT_REPO = "connordilgren/gpt2-gsm8k-coconut"
DEFAULT_CHECKPOINT_FILE = "checkpoint_33"
DEFAULT_CHECKPOINT_SHA256 = (
    "f7a9b5fda7c5c2afa972aaa22ac9aef13d6f083e202fe8a09649d810e3593213"
)

ChartOperation = Callable[[Tensor, int, torch.Generator], Tensor]
RewardFunction = Callable[[Tensor], float]


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


def _load_pinned_coconut_class(source_directory: Path) -> type:
    source_file = source_directory / "coconut.py"
    if not source_file.is_file():
        raise FileNotFoundError(f"missing pinned Coconut source: {source_file}")
    commit = _source_commit(source_directory)
    if commit != PINNED_COCONUT_COMMIT:
        raise ValueError(
            f"Coconut source commit is {commit}, expected {PINNED_COCONUT_COMMIT}"
        )
    specification = importlib.util.spec_from_file_location(
        "_latent_geometry_pinned_coconut", source_file
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"could not load Coconut source from {source_file}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.Coconut


@dataclass(frozen=True)
class CoconutModelBundle:
    model: Any
    tokenizer: Any
    model_id: str
    model_revision: str
    attention_implementation: str
    checkpoint_path: Path
    checkpoint_sha256: str
    coconut_source_directory: Path
    coconut_source_commit: str
    device: torch.device


def load_public_coconut(
    *,
    checkpoint_path: str | Path,
    coconut_source_directory: str | Path,
    model_id: str = DEFAULT_MODEL_ID,
    model_revision: str = DEFAULT_MODEL_REVISION,
    expected_checkpoint_sha256: str = DEFAULT_CHECKPOINT_SHA256,
    attention_implementation: str | None = None,
    device: str | torch.device = "cpu",
    cache_directory: str | Path | None = None,
) -> CoconutModelBundle:
    """Load the released GPT-2 Coconut checkpoint with strict integrity checks."""

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Real-model dependencies are unavailable. Use requirements-real-model.txt."
        ) from error

    checkpoint = Path(checkpoint_path).resolve()
    source_directory = Path(coconut_source_directory).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing Coconut checkpoint: {checkpoint}")
    checkpoint_sha256 = _sha256_file(checkpoint)
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError(
            "Coconut checkpoint SHA-256 mismatch: "
            f"got {checkpoint_sha256}, expected {expected_checkpoint_sha256}"
        )
    coconut_class = _load_pinned_coconut_class(source_directory)
    resolved_device = torch.device(device)
    cache_dir = str(Path(cache_directory).resolve()) if cache_directory else None

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=model_revision, cache_dir=cache_dir
    )
    tokenizer.pad_token = tokenizer.eos_token
    for token in ("<|start-latent|>", "<|end-latent|>", "<|latent|>"):
        tokenizer.add_tokens(token)

    model_kwargs: dict[str, Any] = {
        "revision": model_revision,
        "cache_dir": cache_dir,
    }
    if attention_implementation is not None:
        model_kwargs["attn_implementation"] = attention_implementation
    base_model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    try:
        base_model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    except TypeError:  # Compatibility with Transformers releases before mean_resizing.
        base_model.resize_token_embeddings(len(tokenizer))

    model = coconut_class(
        base_model,
        tokenizer.convert_tokens_to_ids("<|latent|>"),
        tokenizer.convert_tokens_to_ids("<|start-latent|>"),
        tokenizer.convert_tokens_to_ids("<|end-latent|>"),
        tokenizer.eos_token_id,
    )
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise ValueError(
            "checkpoint mismatch: "
            f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
        )
    model.to(resolved_device)
    model.eval()

    return CoconutModelBundle(
        model=model,
        tokenizer=tokenizer,
        model_id=model_id,
        model_revision=model_revision,
        attention_implementation=str(
            getattr(base_model.config, "_attn_implementation", "unknown")
        ),
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        coconut_source_directory=source_directory,
        coconut_source_commit=PINNED_COCONUT_COMMIT,
        device=resolved_device,
    )


def prepare_coconut_input(
    bundle: CoconutModelBundle,
    question: str,
    *,
    num_latents: int = 6,
) -> tuple[Tensor, Tensor]:
    if not question.strip():
        raise ValueError("question must be non-empty")
    if not isinstance(num_latents, int) or num_latents < 1:
        raise ValueError("num_latents must be a positive integer")
    question_text = question if question.endswith("\n") else question + "\n"
    question_ids = bundle.tokenizer.encode(question_text, add_special_tokens=True)
    tokens = (
        question_ids
        + [bundle.model.start_latent_id]
        + [bundle.model.latent_token_id] * num_latents
        + [bundle.model.end_latent_id]
    )
    input_ids = torch.tensor(tokens, dtype=torch.long, device=bundle.device).unsqueeze(0)
    return input_ids, torch.ones_like(input_ids)


@dataclass(frozen=True)
class CoconutLatentStep:
    pass_index: int
    token_position: int
    proposed_native: Tensor
    charted: Tensor
    consumed_native: Tensor


@dataclass(frozen=True)
class CoconutRunResult:
    input_ids: Tensor
    filled_inputs_embeds: Tensor
    logits: Tensor
    latent_steps: tuple[CoconutLatentStep, ...]
    trace: LatentTrace | None
    model_forward_calls: int


class CoconutChartRunner:
    """Run Coconut latent passes with an exact chart around hidden-state feedback.

    The latent-pass schedule follows the MIT-licensed official Coconut
    implementation at ``PINNED_COCONUT_COMMIT``. The runner is audit-only and
    intentionally uses batch size one so every intervention has an unambiguous
    pass index and replay context.
    """

    def __init__(self, coconut_model: Any, chart: LatentChart) -> None:
        required = (
            "base_causallm",
            "embedding",
            "latent_token_id",
            "start_latent_id",
            "end_latent_id",
        )
        missing = [name for name in required if not hasattr(coconut_model, name)]
        if missing:
            raise TypeError(f"Coconut model is missing attributes: {missing}")
        embedding_dimension = int(coconut_model.embedding.weight.shape[-1])
        if chart.dimension != embedding_dimension:
            raise ValueError(
                f"chart dimension {chart.dimension} does not match model dimension "
                f"{embedding_dimension}"
            )
        self.model = coconut_model
        self.chart = chart

    def run(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        seed: int,
        operation: ChartOperation | None = None,
        native_overrides: Mapping[int, Tensor] | None = None,
        capture_trace: bool = False,
        sample_id: str = "sample-0",
        track_gradients: bool = False,
    ) -> CoconutRunResult:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Coconut chart runner requires input_ids with batch size one")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        if not sample_id:
            raise ValueError("sample_id must be non-empty")
        latent_positions = (input_ids[0] == self.model.latent_token_id).nonzero(
            as_tuple=False
        ).flatten()
        if latent_positions.numel() < 1:
            raise ValueError("input must contain at least one Coconut latent token")
        override_map = dict(native_overrides or {})
        invalid_override_keys = set(override_map) - set(range(latent_positions.numel()))
        if invalid_override_keys:
            raise ValueError(f"invalid latent override passes: {sorted(invalid_override_keys)}")

        inputs_embeds = self.model.embedding(input_ids)
        next_compute_range = (0, int(latent_positions.min().item()))
        position_ids = torch.arange(
            input_ids.shape[1], dtype=torch.long, device=input_ids.device
        ).reshape(1, -1)
        kv_cache = None
        logits: list[Tensor] = []
        latent_steps: list[CoconutLatentStep] = []
        trace = LatentTrace() if capture_trace else None
        generator = make_generator(seed, input_ids.device)
        forward_calls = 0
        gradient_context = nullcontext() if track_gradients else torch.no_grad()

        with seeded_rng(seed), gradient_context:
            for pass_index, token_position_tensor in enumerate(latent_positions):
                token_position = int(token_position_tensor.item())
                if kv_cache is None:
                    outputs = self.model.base_causallm(
                        inputs_embeds=inputs_embeds[
                            :, next_compute_range[0] : next_compute_range[1], :
                        ],
                        attention_mask=attention_mask[
                            :, next_compute_range[0] : next_compute_range[1]
                        ],
                        position_ids=position_ids[
                            :, next_compute_range[0] : next_compute_range[1]
                        ],
                        output_hidden_states=True,
                    )
                    hidden_states_offset = 0
                else:
                    past_key_values = tuple(
                        (
                            key[:, :, : next_compute_range[0], :],
                            value[:, :, : next_compute_range[0], :],
                        )
                        for key, value in kv_cache
                    )
                    outputs = self.model.base_causallm(
                        inputs_embeds=inputs_embeds[
                            :, next_compute_range[0] : next_compute_range[1], :
                        ],
                        attention_mask=attention_mask[:, : next_compute_range[1]],
                        position_ids=position_ids[
                            :, next_compute_range[0] : next_compute_range[1]
                        ],
                        past_key_values=past_key_values,
                        output_hidden_states=True,
                    )
                    hidden_states_offset = next_compute_range[0]
                forward_calls += 1
                logits.append(outputs.logits)
                hidden_states = outputs.hidden_states[-1]
                kv_cache = outputs.past_key_values

                proposed_native = hidden_states[
                    0, token_position - 1 - hidden_states_offset, :
                ]
                charted = self.chart.encode(proposed_native)
                if pass_index in override_map:
                    consumed_native = override_map[pass_index].to(
                        device=proposed_native.device, dtype=proposed_native.dtype
                    )
                    if consumed_native.shape != proposed_native.shape:
                        raise ValueError("native latent override has an unexpected shape")
                    if not torch.isfinite(consumed_native).all():
                        raise ValueError("native latent override contains non-finite values")
                    consumed_charted = self.chart.encode(consumed_native)
                else:
                    consumed_charted = (
                        operation(charted, pass_index, generator)
                        if operation is not None
                        else charted
                    )
                    if (
                        not isinstance(consumed_charted, Tensor)
                        or consumed_charted.shape != charted.shape
                        or not torch.isfinite(consumed_charted).all()
                    ):
                        raise ValueError(
                            "chart operation must return a finite shape-matched tensor"
                        )
                    consumed_native = self.chart.decode(consumed_charted).to(
                        dtype=proposed_native.dtype
                    )

                if trace is not None:
                    trace.append(
                        LatentStepRecord.capture(
                            sample_id=sample_id,
                            step_index=pass_index,
                            prefix_state={
                                "input_ids": input_ids,
                                "attention_mask": attention_mask,
                                "target_pass": pass_index,
                            },
                            latent=consumed_native,
                            rng_seed=seed,
                            chart_name=self.chart.name,
                            metadata={
                                "adapter": "CoconutChartRunner",
                                "latent_token_position": token_position,
                                "operation_applied": operation is not None,
                            },
                        )
                    )

                latent_steps.append(
                    CoconutLatentStep(
                        pass_index=pass_index,
                        token_position=token_position,
                        proposed_native=proposed_native,
                        charted=consumed_charted,
                        consumed_native=consumed_native,
                    )
                )
                inputs_embeds = inputs_embeds.clone()
                inputs_embeds[0, token_position, :] = consumed_native
                next_compute_range = (
                    next_compute_range[1],
                    input_ids.shape[1]
                    if pass_index + 1 >= latent_positions.numel()
                    else next_compute_range[1] + 1,
                )

            past_key_values = (
                tuple(
                    (
                        key[:, :, : next_compute_range[0], :],
                        value[:, :, : next_compute_range[0], :],
                    )
                    for key, value in kv_cache
                )
                if kv_cache is not None
                else None
            )
            outputs = self.model.base_causallm(
                inputs_embeds=inputs_embeds[
                    :, next_compute_range[0] : next_compute_range[1], :
                ],
                attention_mask=attention_mask[:, : next_compute_range[1]],
                position_ids=position_ids[
                    :, next_compute_range[0] : next_compute_range[1]
                ],
                past_key_values=past_key_values,
                output_hidden_states=True,
            )
            forward_calls += 1
            logits.append(outputs.logits)

        return CoconutRunResult(
            input_ids=input_ids,
            filled_inputs_embeds=inputs_embeds,
            logits=torch.cat(logits, dim=-2),
            latent_steps=tuple(latent_steps),
            trace=trace,
            model_forward_calls=forward_calls,
        )


class CoconutContinuationAdapter:
    """Replay a Coconut latent intervention and return fixed-horizon token logits."""

    def __init__(
        self,
        runner: CoconutChartRunner,
        *,
        temperature: float = 1.0,
        reward_function: RewardFunction | None = None,
        stop_at_eos: bool = False,
        track_gradients: bool = False,
    ) -> None:
        if temperature < 0 or not torch.isfinite(torch.tensor(temperature)):
            raise ValueError("temperature must be finite and non-negative")
        self.runner = runner
        self.temperature = float(temperature)
        self.reward_function = reward_function
        self.stop_at_eos = bool(stop_at_eos)
        self.track_gradients = bool(track_gradients)

    def __call__(
        self,
        context: Any,
        latent: Tensor,
        *,
        horizon: int,
        generator: torch.Generator,
    ) -> ContinuationOutput:
        return self._rollout(
            context,
            latent,
            horizon=horizon,
            generator=generator,
            forced_token_ids=None,
        )

    def force(
        self,
        context: Any,
        latent: Tensor,
        *,
        token_ids: Tensor,
        generator: torch.Generator,
    ) -> ContinuationOutput:
        if not isinstance(token_ids, Tensor) or token_ids.ndim != 1:
            raise ValueError("token_ids must be a rank-1 tensor")
        if token_ids.numel() < 1 or token_ids.dtype != torch.long:
            raise ValueError("token_ids must be a non-empty long tensor")
        return self._rollout(
            context,
            latent,
            horizon=int(token_ids.numel()),
            generator=generator,
            forced_token_ids=token_ids,
        )

    def _rollout(
        self,
        context: Any,
        latent: Tensor,
        *,
        horizon: int,
        generator: torch.Generator,
        forced_token_ids: Tensor | None,
    ) -> ContinuationOutput:
        if not isinstance(context, dict):
            raise TypeError("Coconut replay context must be a dictionary")
        required = {"input_ids", "attention_mask", "target_pass"}
        if set(context) != required:
            raise ValueError("Coconut replay context has an unexpected schema")
        input_ids = context["input_ids"]
        attention_mask = context["attention_mask"]
        target_pass = context["target_pass"]
        if not isinstance(target_pass, int) or target_pass < 0:
            raise ValueError("target_pass must be a non-negative integer")
        model_device = self.runner.model.embedding.weight.device
        input_ids = input_ids.to(model_device)
        attention_mask = attention_mask.to(model_device)
        candidate = latent.to(
            device=model_device, dtype=self.runner.model.embedding.weight.dtype
        )
        seed = int(generator.initial_seed())
        result = self.runner.run(
            input_ids,
            attention_mask,
            seed=seed,
            native_overrides={target_pass: candidate},
            track_gradients=self.track_gradients,
        )
        if generator.device != model_device:
            generator = make_generator(seed, model_device)

        current_embeddings = result.filled_inputs_embeds
        logits: list[Tensor] = []
        tokens: list[Tensor] = []
        model_forward_calls = result.model_forward_calls
        outputs = self.runner.model.base_causallm(
            inputs_embeds=current_embeddings,
            attention_mask=torch.ones(
                current_embeddings.shape[:2],
                dtype=torch.long,
                device=model_device,
            ),
        )
        model_forward_calls += 1
        next_logits = outputs.logits[0, -1, :]
        forced_tokens = (
            forced_token_ids.to(device=model_device, dtype=torch.long)
            if forced_token_ids is not None
            else None
        )
        terminated = False
        for generation_step in range(horizon):
            logits.append(next_logits)
            if forced_tokens is not None:
                token = forced_tokens[generation_step].view(1)
            elif self.temperature == 0.0:
                token = torch.argmax(next_logits).view(1)
            else:
                probabilities = torch.softmax(next_logits / self.temperature, dim=-1)
                token = torch.multinomial(
                    probabilities, num_samples=1, generator=generator
                )
            tokens.append(token.squeeze(0))
            if self.stop_at_eos and int(token.item()) == int(
                self.runner.model.eos_token_id
            ):
                terminated = True
                break
            token_embedding = self.runner.model.embedding(token.view(1, 1))
            current_embeddings = torch.cat(
                [current_embeddings, token_embedding], dim=1
            )
            if generation_step + 1 < horizon:
                outputs = self.runner.model.base_causallm(
                    inputs_embeds=current_embeddings,
                    attention_mask=torch.ones(
                        current_embeddings.shape[:2],
                        dtype=torch.long,
                        device=model_device,
                    ),
                )
                model_forward_calls += 1
                next_logits = outputs.logits[0, -1, :]

        token_ids = torch.stack(tokens)
        reward = (
            float(self.reward_function(token_ids.detach().cpu()))
            if self.reward_function is not None
            else None
        )
        return ContinuationOutput(
            logits=torch.stack(logits),
            reward=reward,
            token_ids=token_ids,
            prompt_tokens=int(input_ids.shape[1]),
            generated_tokens=0 if forced_tokens is not None else len(tokens),
            model_forward_calls=model_forward_calls,
            metadata={
                "target_pass": target_pass,
                "forced_history": forced_tokens is not None,
                "teacher_forced_tokens": len(tokens) if forced_tokens is not None else 0,
                "sampling_temperature": self.temperature,
                "continuation_forward_mode": "full_recompute",
                "requested_horizon": horizon,
                "valid_steps": len(tokens),
                "terminated": terminated,
                "stop_at_eos": self.stop_at_eos,
            },
        )
