from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor

from .accounting import ComputeLedger, ComputeSnapshot
from .metrics import DivergenceName, multi_horizon_divergence
from .rng import make_generator, seeded_rng
from .trace import LatentStepRecord


@dataclass(frozen=True)
class ContinuationOutput:
    """Observable continuation statistics for one latent state and one seed."""

    logits: Tensor
    reward: float | None = None
    token_ids: Tensor | None = None
    prompt_tokens: int = 0
    generated_tokens: int = 0
    model_forward_calls: int = 1
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.logits, Tensor) or self.logits.ndim != 2:
            raise ValueError("logits must have shape (horizon, classes)")
        if self.logits.shape[0] < 1 or self.logits.shape[1] < 2:
            raise ValueError("logits must contain a positive horizon and at least two classes")
        if not self.logits.is_floating_point() or not torch.isfinite(self.logits).all():
            raise ValueError("logits must contain finite floating-point values")
        if self.reward is not None and not math.isfinite(float(self.reward)):
            raise ValueError("reward must be finite")
        if self.token_ids is not None:
            if not isinstance(self.token_ids, Tensor) or self.token_ids.ndim != 1:
                raise ValueError("token_ids must be a rank-1 tensor")
        counts = (self.prompt_tokens, self.generated_tokens, self.model_forward_calls)
        if any(not isinstance(value, int) or value < 0 for value in counts):
            raise ValueError("token and model-call counts must be non-negative integers")


RolloutFunction = Callable[..., ContinuationOutput]


@dataclass(frozen=True)
class ContinuationComparison:
    seeds: tuple[int, ...]
    per_seed_divergence: Tensor
    mean_divergence: float
    standard_error: float
    per_seed_reward_delta: Tensor | None
    mean_reward_delta: float | None
    reference_outputs: tuple[ContinuationOutput, ...]
    candidate_outputs: tuple[ContinuationOutput, ...]
    compute: ComputeSnapshot


@dataclass(frozen=True)
class ContinuationEvaluation:
    seeds: tuple[int, ...]
    outputs: tuple[ContinuationOutput, ...]
    compute: ComputeSnapshot


class ContinuationEvaluator:
    """Compare continuations with identical RNG streams for each latent variant."""

    def __init__(
        self,
        rollout_function: RolloutFunction,
        *,
        ledger: ComputeLedger | None = None,
        use_no_grad: bool = True,
    ) -> None:
        if not callable(rollout_function):
            raise TypeError("rollout_function must be callable")
        self.rollout_function = rollout_function
        self.ledger = ledger if ledger is not None else ComputeLedger()
        self.use_no_grad = use_no_grad

    def _run_once(
        self,
        context: Any,
        latent: Tensor,
        *,
        horizon: int,
        seed: int,
    ) -> ContinuationOutput:
        if not isinstance(latent, Tensor) or not latent.is_floating_point():
            raise TypeError("latent must be a floating-point tensor")
        if latent.ndim != 1:
            raise ValueError("latent must be a rank-1 tensor")
        if horizon < 1:
            raise ValueError("horizon must be positive")
        generator = make_generator(seed, latent.device)
        grad_context = torch.no_grad() if self.use_no_grad else torch.enable_grad()
        with self.ledger.timed("continuation_rollout"):
            with seeded_rng(seed), grad_context:
                output = self.rollout_function(
                    context,
                    latent,
                    horizon=horizon,
                    generator=generator,
                )
        if not isinstance(output, ContinuationOutput):
            raise TypeError("rollout_function must return ContinuationOutput")
        if output.logits.shape[0] != horizon:
            raise ValueError(
                f"rollout returned horizon {output.logits.shape[0]}, expected {horizon}"
            )
        self.ledger.record_rollout(
            prompt_tokens=output.prompt_tokens,
            generated_tokens=output.generated_tokens,
            model_forward_calls=output.model_forward_calls,
        )
        return output

    def compare(
        self,
        context: Any,
        reference_latent: Tensor,
        candidate_latent: Tensor,
        *,
        horizon: int,
        seeds: Sequence[int],
        divergence: DivergenceName = "symmetric_kl",
        horizon_weights: Sequence[float] | Tensor | None = None,
    ) -> ContinuationComparison:
        if reference_latent.shape != candidate_latent.shape:
            raise ValueError("reference and candidate latents must have identical shapes")
        reference = self.evaluate(
            context,
            reference_latent,
            horizon=horizon,
            seeds=seeds,
        )
        candidate = self.evaluate(
            context,
            candidate_latent,
            horizon=horizon,
            seeds=seeds,
        )
        return self.compare_evaluations(
            reference,
            candidate,
            divergence=divergence,
            horizon_weights=horizon_weights,
        )

    def evaluate(
        self,
        context: Any,
        latent: Tensor,
        *,
        horizon: int,
        seeds: Sequence[int],
    ) -> ContinuationEvaluation:
        if not seeds:
            raise ValueError("at least one seed is required")
        normalized_seeds = tuple(int(seed) for seed in seeds)
        if len(set(normalized_seeds)) != len(normalized_seeds):
            raise ValueError("seeds must be unique")
        outputs = tuple(
            self._run_once(context, latent, horizon=horizon, seed=seed)
            for seed in normalized_seeds
        )
        return ContinuationEvaluation(
            seeds=normalized_seeds,
            outputs=outputs,
            compute=self.ledger.snapshot(),
        )

    def compare_evaluations(
        self,
        reference: ContinuationEvaluation,
        candidate: ContinuationEvaluation,
        *,
        divergence: DivergenceName = "symmetric_kl",
        horizon_weights: Sequence[float] | Tensor | None = None,
    ) -> ContinuationComparison:
        if not isinstance(reference, ContinuationEvaluation) or not isinstance(
            candidate, ContinuationEvaluation
        ):
            raise TypeError("reference and candidate must be ContinuationEvaluation objects")
        if reference.seeds != candidate.seeds:
            raise ValueError("reference and candidate evaluations must use identical seeds")
        divergences: list[Tensor] = []
        reward_deltas: list[float] = []
        all_rewards_available = True
        for reference_output, candidate_output in zip(
            reference.outputs, candidate.outputs
        ):
            divergences.append(
                multi_horizon_divergence(
                    reference_output.logits,
                    candidate_output.logits,
                    horizon_weights=horizon_weights,
                    divergence=divergence,
                ).detach()
            )
            if reference_output.reward is None or candidate_output.reward is None:
                all_rewards_available = False
            else:
                reward_deltas.append(
                    float(candidate_output.reward - reference_output.reward)
                )

        per_seed = torch.stack(divergences).to(torch.float64).cpu()
        mean_divergence = float(per_seed.mean())
        standard_error = (
            float(per_seed.std(unbiased=True) / math.sqrt(per_seed.numel()))
            if per_seed.numel() > 1
            else 0.0
        )

        reward_tensor: Tensor | None = None
        mean_reward_delta: float | None = None
        if all_rewards_available:
            reward_tensor = torch.tensor(reward_deltas, dtype=torch.float64)
            mean_reward_delta = float(reward_tensor.mean())

        return ContinuationComparison(
            seeds=reference.seeds,
            per_seed_divergence=per_seed,
            mean_divergence=mean_divergence,
            standard_error=standard_error,
            per_seed_reward_delta=reward_tensor,
            mean_reward_delta=mean_reward_delta,
            reference_outputs=reference.outputs,
            candidate_outputs=candidate.outputs,
            compute=self.ledger.snapshot(),
        )

    def compare_record(
        self,
        record: LatentStepRecord,
        candidate_latent: Tensor,
        *,
        horizon: int,
        seeds: Sequence[int] | None = None,
        divergence: DivergenceName = "symmetric_kl",
        horizon_weights: Sequence[float] | Tensor | None = None,
    ) -> ContinuationComparison:
        """Replay a captured prefix and compare its factual latent to a candidate."""

        if not isinstance(record, LatentStepRecord):
            raise TypeError("record must be a LatentStepRecord")
        replay_seeds = tuple(seeds) if seeds is not None else (record.rng_seed,)
        return self.compare(
            record.prefix_state,
            record.latent.to(candidate_latent.device, dtype=candidate_latent.dtype),
            candidate_latent,
            horizon=horizon,
            seeds=replay_seeds,
            divergence=divergence,
            horizon_weights=horizon_weights,
        )
