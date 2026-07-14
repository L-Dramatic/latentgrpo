from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
from torch import Tensor

from .charts import LatentChart
from .evaluator import ContinuationOutput
from .rng import make_generator, seeded_rng
from .trace import LatentStepRecord, LatentTrace


ChartOperation = Callable[[Tensor, int, torch.Generator], Tensor]
RewardFunction = Callable[[Tensor], float]


_ANSWER_PROMPTS = {
    "small": "\n Answer:",
    "mistral": "\n Answer: [/INST]",
    "qwen": "<|im_end|>\n<|im_start|>assistant\nAnswer: ",
}


@dataclass(frozen=True)
class ChartedThoughtTrajectory:
    proposed_native_thoughts: tuple[Tensor, ...]
    charted_thoughts: tuple[Tensor, ...]
    consumed_native_thoughts: tuple[Tensor, ...]
    trace: LatentTrace | None


def _validate_model_interface(model: Any) -> None:
    required = ("llm_model", "llm_tokenizer", "proj", "config")
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise TypeError(f"model is missing required attributes: {missing}")
    if not callable(model.llm_model) or not callable(model.proj):
        raise TypeError("model.llm_model and model.proj must be callable")
    if not hasattr(model.llm_model, "get_input_embeddings"):
        raise TypeError("model.llm_model must expose get_input_embeddings()")


class LatentGRPOChartAdapter:
    """Insert an invertible chart around the LatentGRPO thought interface.

    The projection still proposes a native latent ``z``. The adapter encodes it
    as ``u = phi(z)``, optionally applies an operation in chart coordinates, and
    decodes it before the backbone consumes the thought. With no operation, this
    is an exact function-preserving wrapper up to measured numerical error.
    """

    def __init__(
        self,
        model: Any,
        chart: LatentChart,
        *,
        answer_prompt: str | None = None,
    ) -> None:
        _validate_model_interface(model)
        self.model = model
        self.chart = chart
        self.answer_prompt = (
            answer_prompt
            if answer_prompt is not None
            else _ANSWER_PROMPTS.get(str(model.config), "\n Answer:")
        )

    def _answer_prompt_embeddings(self, reference: Tensor) -> Tensor:
        tokenized = self.model.llm_tokenizer(
            self.answer_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=64,
            add_special_tokens=False,
        )
        token_ids = tokenized["input_ids"].to(reference.device)
        embeddings = self.model.llm_model.get_input_embeddings()(token_ids)
        return embeddings.to(dtype=reference.dtype)

    def generate(
        self,
        query_embeddings: Tensor,
        *,
        num_thoughts: int,
        seed: int,
        operation: ChartOperation | None = None,
        capture_trace: bool = False,
        sample_ids: Sequence[str] | None = None,
        track_gradients: bool = False,
    ) -> ChartedThoughtTrajectory:
        if not isinstance(query_embeddings, Tensor) or query_embeddings.ndim != 3:
            raise ValueError("query_embeddings must have shape (batch, sequence, hidden)")
        if query_embeddings.shape[-1] != self.chart.dimension:
            raise ValueError("query hidden dimension does not match the chart")
        if not query_embeddings.is_floating_point() or not torch.isfinite(
            query_embeddings
        ).all():
            raise ValueError("query_embeddings must contain finite floating-point values")
        if not isinstance(num_thoughts, int) or num_thoughts < 1:
            raise ValueError("num_thoughts must be a positive integer")

        batch_size = int(query_embeddings.shape[0])
        if sample_ids is None:
            normalized_sample_ids = tuple(f"sample-{index}" for index in range(batch_size))
        else:
            normalized_sample_ids = tuple(sample_ids)
            if len(normalized_sample_ids) != batch_size or any(
                not sample_id for sample_id in normalized_sample_ids
            ):
                raise ValueError("sample_ids must contain one non-empty id per batch item")
            if len(set(normalized_sample_ids)) != len(normalized_sample_ids):
                raise ValueError("sample_ids must be unique")

        current_embeddings = query_embeddings
        answer_prompt_embeddings = self._answer_prompt_embeddings(query_embeddings)
        proposed_native: list[Tensor] = []
        charted_thoughts: list[Tensor] = []
        consumed_native: list[Tensor] = []
        trace = LatentTrace() if capture_trace else None
        generator = make_generator(seed, query_embeddings.device)
        gradient_context = nullcontext() if track_gradients else torch.no_grad()

        with seeded_rng(seed), gradient_context:
            for step_index in range(num_thoughts):
                outputs = self.model.llm_model(
                    inputs_embeds=current_embeddings,
                    attention_mask=torch.ones(
                        current_embeddings.shape[:2],
                        dtype=torch.long,
                        device=current_embeddings.device,
                    ),
                    output_hidden_states=True,
                )
                hidden = outputs.hidden_states[-1][:, -1, :]
                native = self.model.proj(hidden)
                charted = self.chart.encode(native)
                consumed_charted = (
                    operation(charted, step_index, generator)
                    if operation is not None
                    else charted
                )
                if not isinstance(consumed_charted, Tensor) or consumed_charted.shape != charted.shape:
                    raise ValueError("chart operation must return a tensor with unchanged shape")
                if not torch.isfinite(consumed_charted).all():
                    raise ValueError("chart operation produced non-finite values")
                consumed = self.chart.decode(consumed_charted).to(dtype=native.dtype)

                proposed_native.append(native)
                charted_thoughts.append(consumed_charted)
                consumed_native.append(consumed)

                if trace is not None:
                    for batch_index, sample_id in enumerate(normalized_sample_ids):
                        prefix_state = {
                            "current_embeddings": current_embeddings[
                                batch_index : batch_index + 1
                            ],
                            "remaining_latent_steps": num_thoughts - step_index - 1,
                            "answer_prompt_embeddings": answer_prompt_embeddings,
                        }
                        trace.append(
                            LatentStepRecord.capture(
                                sample_id=sample_id,
                                step_index=step_index,
                                prefix_state=prefix_state,
                                latent=consumed[batch_index],
                                rng_seed=seed,
                                chart_name=self.chart.name,
                                metadata={
                                    "adapter": "LatentGRPOChartAdapter",
                                    "operation_applied": operation is not None,
                                    "sequence_length_before_action": int(
                                        current_embeddings.shape[1]
                                    ),
                                },
                            )
                        )

                current_embeddings = torch.cat(
                    [current_embeddings, consumed.unsqueeze(1)], dim=1
                )

        return ChartedThoughtTrajectory(
            proposed_native_thoughts=tuple(proposed_native),
            charted_thoughts=tuple(charted_thoughts),
            consumed_native_thoughts=tuple(consumed_native),
            trace=trace,
        )


class LatentGRPOContinuationAdapter:
    """Replay a captured LatentGRPO prefix and expose fixed-horizon token logits."""

    def __init__(
        self,
        model: Any,
        *,
        temperature: float = 1.0,
        reward_function: RewardFunction | None = None,
    ) -> None:
        _validate_model_interface(model)
        if temperature < 0 or not torch.isfinite(torch.tensor(temperature)):
            raise ValueError("temperature must be finite and non-negative")
        self.model = model
        self.temperature = float(temperature)
        self.reward_function = reward_function

    def __call__(
        self,
        context: Any,
        latent: Tensor,
        *,
        horizon: int,
        generator: torch.Generator,
    ) -> ContinuationOutput:
        if not isinstance(context, dict):
            raise TypeError("LatentGRPO replay context must be a dictionary")
        required = {
            "current_embeddings",
            "remaining_latent_steps",
            "answer_prompt_embeddings",
        }
        if not required.issubset(context):
            raise ValueError(f"replay context is missing keys: {sorted(required - set(context))}")
        current_embeddings = context["current_embeddings"]
        answer_prompt_embeddings = context["answer_prompt_embeddings"]
        remaining_steps = context["remaining_latent_steps"]
        if not isinstance(current_embeddings, Tensor) or current_embeddings.ndim != 3:
            raise ValueError("current_embeddings must have rank three")
        if current_embeddings.shape[0] != 1:
            raise ValueError("continuation replay currently supports batch size one")
        if not isinstance(answer_prompt_embeddings, Tensor) or answer_prompt_embeddings.ndim != 3:
            raise ValueError("answer_prompt_embeddings must have rank three")
        if not isinstance(remaining_steps, int) or remaining_steps < 0:
            raise ValueError("remaining_latent_steps must be a non-negative integer")
        if latent.shape != (current_embeddings.shape[-1],):
            raise ValueError("candidate latent dimension does not match the replay context")

        try:
            model_parameter = next(self.model.llm_model.parameters())
            model_device = model_parameter.device
            model_dtype = model_parameter.dtype
        except StopIteration:
            model_device = current_embeddings.device
            model_dtype = current_embeddings.dtype
        current_embeddings = current_embeddings.to(device=model_device, dtype=model_dtype)
        answer_prompt_embeddings = answer_prompt_embeddings.to(
            device=model_device, dtype=model_dtype
        )
        candidate = latent.to(device=model_device, dtype=model_dtype).view(1, 1, -1)
        current_embeddings = torch.cat([current_embeddings, candidate], dim=1)

        model_forward_calls = 0
        for _ in range(remaining_steps):
            outputs = self.model.llm_model(
                inputs_embeds=current_embeddings,
                attention_mask=torch.ones(
                    current_embeddings.shape[:2],
                    dtype=torch.long,
                    device=model_device,
                ),
                output_hidden_states=True,
            )
            model_forward_calls += 1
            hidden = outputs.hidden_states[-1][:, -1, :]
            next_latent = self.model.proj(hidden).unsqueeze(1)
            current_embeddings = torch.cat([current_embeddings, next_latent], dim=1)

        current_embeddings = torch.cat(
            [current_embeddings, answer_prompt_embeddings], dim=1
        )
        prompt_tokens = int(current_embeddings.shape[1])
        if generator.device != model_device:
            generator = make_generator(generator.initial_seed(), model_device)

        logits: list[Tensor] = []
        sampled_tokens: list[Tensor] = []
        embedding_layer = self.model.llm_model.get_input_embeddings()
        for _ in range(horizon):
            outputs = self.model.llm_model(
                inputs_embeds=current_embeddings,
                attention_mask=torch.ones(
                    current_embeddings.shape[:2],
                    dtype=torch.long,
                    device=model_device,
                ),
                output_hidden_states=False,
            )
            model_forward_calls += 1
            next_logits = outputs.logits[0, -1, :]
            logits.append(next_logits)
            if self.temperature == 0.0:
                token = torch.argmax(next_logits).view(1)
            else:
                probabilities = torch.softmax(next_logits / self.temperature, dim=-1)
                token = torch.multinomial(
                    probabilities, num_samples=1, generator=generator
                )
            sampled_tokens.append(token.squeeze(0))
            token_embedding = embedding_layer(token.view(1, 1)).to(dtype=model_dtype)
            current_embeddings = torch.cat(
                [current_embeddings, token_embedding], dim=1
            )

        token_ids = torch.stack(sampled_tokens)
        reward = (
            float(self.reward_function(token_ids.detach().cpu()))
            if self.reward_function is not None
            else None
        )
        return ContinuationOutput(
            logits=torch.stack(logits),
            reward=reward,
            token_ids=token_ids,
            prompt_tokens=prompt_tokens,
            generated_tokens=horizon,
            model_forward_calls=model_forward_calls,
            metadata={"remaining_latent_steps": remaining_steps},
        )
