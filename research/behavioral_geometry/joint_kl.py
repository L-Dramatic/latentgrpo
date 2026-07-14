from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import torch
from torch import Tensor

from research.coordinate_invariance.accounting import ComputeLedger, ComputeSnapshot
from research.coordinate_invariance.evaluator import ContinuationOutput
from research.coordinate_invariance.metrics import stable_categorical_kl_from_logits
from research.coordinate_invariance.rng import make_generator, seeded_rng


class ForcedAutoregressivePolicy(Protocol):
    def __call__(
        self,
        context: Any,
        latent: Tensor,
        *,
        horizon: int,
        generator: torch.Generator,
    ) -> ContinuationOutput: ...

    def force(
        self,
        context: Any,
        latent: Tensor,
        *,
        token_ids: Tensor,
        generator: torch.Generator,
    ) -> ContinuationOutput: ...


@dataclass(frozen=True)
class DirectionalJointKL:
    seeds: tuple[int, ...]
    per_seed_valid_steps: Tensor
    per_seed_step_kl: Tensor
    per_seed_total_kl: Tensor
    per_seed_mean_kl: Tensor
    per_seed_first_step_kl: Tensor
    per_seed_tail_kl: Tensor
    reference_outputs: tuple[ContinuationOutput, ...]
    forced_candidate_outputs: tuple[ContinuationOutput, ...]
    compute: ComputeSnapshot

    @property
    def mean_total_kl(self) -> float:
        return float(self.per_seed_total_kl.mean())

    @property
    def mean_per_token_kl(self) -> float:
        return float(self.per_seed_mean_kl.mean())

    @property
    def mean_first_step_kl(self) -> float:
        return float(self.per_seed_first_step_kl.mean())

    @property
    def mean_tail_kl(self) -> float:
        return float(self.per_seed_tail_kl.mean())

    @property
    def mean_step_kl(self) -> Tensor:
        return self.per_seed_step_kl.mean(dim=0)


@dataclass(frozen=True)
class ReferenceContinuationBatch:
    seeds: tuple[int, ...]
    horizon: int
    reference_latent: Tensor
    outputs: tuple[ContinuationOutput, ...]
    compute: ComputeSnapshot


@dataclass(frozen=True)
class SymmetricJointKL:
    forward: DirectionalJointKL
    reverse: DirectionalJointKL

    @property
    def mean_total_kl(self) -> float:
        return 0.5 * (self.forward.mean_total_kl + self.reverse.mean_total_kl)

    @property
    def mean_per_token_kl(self) -> float:
        return 0.5 * (
            self.forward.mean_per_token_kl + self.reverse.mean_per_token_kl
        )


class JointContinuationEvaluator:
    """Estimate autoregressive joint KL with both policies on the same histories.

    For ``KL(P || Q)``, continuations are sampled from ``P``. ``Q`` is then
    teacher-forced on those exact tokens, yielding the chain-rule estimator

    ``sum_t KL(P_t(. | y_<t) || Q_t(. | y_<t))``.
    """

    def __init__(
        self,
        policy: ForcedAutoregressivePolicy,
        *,
        ledger: ComputeLedger | None = None,
        use_no_grad: bool = True,
    ) -> None:
        if not callable(policy) or not callable(getattr(policy, "force", None)):
            raise TypeError("policy must support sampling and forced-history replay")
        self.policy = policy
        self.ledger = ledger if ledger is not None else ComputeLedger()
        self.use_no_grad = use_no_grad

    def _sample(
        self,
        context: Any,
        latent: Tensor,
        *,
        horizon: int,
        seed: int,
    ) -> ContinuationOutput:
        generator = make_generator(seed, latent.device)
        grad_context = torch.no_grad() if self.use_no_grad else torch.enable_grad()
        with self.ledger.timed("joint_kl_reference_sample"):
            with seeded_rng(seed), grad_context:
                output = self.policy(
                    context, latent, horizon=horizon, generator=generator
                )
        self._validate_output(output, horizon=horizon, require_tokens=True)
        self.ledger.record_rollout(
            prompt_tokens=output.prompt_tokens,
            generated_tokens=output.generated_tokens,
            model_forward_calls=output.model_forward_calls,
        )
        return output

    def _force(
        self,
        context: Any,
        latent: Tensor,
        *,
        token_ids: Tensor,
        seed: int,
    ) -> ContinuationOutput:
        generator = make_generator(seed, latent.device)
        grad_context = torch.no_grad() if self.use_no_grad else torch.enable_grad()
        with self.ledger.timed("joint_kl_candidate_force"):
            with seeded_rng(seed), grad_context:
                output = self.policy.force(
                    context,
                    latent,
                    token_ids=token_ids,
                    generator=generator,
                )
        self._validate_output(
            output, horizon=int(token_ids.numel()), require_tokens=True
        )
        if not torch.equal(output.token_ids.cpu(), token_ids.cpu()):
            raise ValueError("forced-history policy did not preserve the supplied tokens")
        self.ledger.record_rollout(
            prompt_tokens=output.prompt_tokens,
            generated_tokens=output.generated_tokens,
            model_forward_calls=output.model_forward_calls,
        )
        return output

    @staticmethod
    def _validate_output(
        output: ContinuationOutput, *, horizon: int, require_tokens: bool
    ) -> None:
        if not isinstance(output, ContinuationOutput):
            raise TypeError("policy must return ContinuationOutput")
        actual_horizon = int(output.logits.shape[0])
        if actual_horizon != horizon:
            valid_early_stop = (
                0 < actual_horizon < horizon
                and output.metadata.get("terminated") is True
                and output.metadata.get("requested_horizon") == horizon
                and output.metadata.get("valid_steps") == actual_horizon
            )
            if not valid_early_stop:
                raise ValueError("policy returned an unexpected continuation horizon")
        if require_tokens and (
            output.token_ids is None or output.token_ids.shape != (actual_horizon,)
        ):
            raise ValueError("policy must return one token id per continuation step")
        temperature = output.metadata.get("sampling_temperature")
        if temperature is not None and (
            not isinstance(temperature, (int, float)) or float(temperature) <= 0.0
        ):
            raise ValueError(
                "joint KL requires a positive stochastic sampling temperature"
            )

    @staticmethod
    def _distribution_logits(output: ContinuationOutput) -> Tensor:
        """Return logits of the categorical distribution used for sampling."""

        temperature = float(output.metadata.get("sampling_temperature", 1.0))
        return output.logits.to(torch.float64) / temperature

    def directional(
        self,
        context: Any,
        reference_latent: Tensor,
        candidate_latent: Tensor,
        *,
        horizon: int,
        seeds: Sequence[int],
    ) -> DirectionalJointKL:
        reference = self.sample_reference(
            context,
            reference_latent,
            horizon=horizon,
            seeds=seeds,
        )
        return self.directional_from_reference(context, candidate_latent, reference)

    def sample_reference(
        self,
        context: Any,
        reference_latent: Tensor,
        *,
        horizon: int,
        seeds: Sequence[int],
    ) -> ReferenceContinuationBatch:
        if not isinstance(reference_latent, Tensor) or not reference_latent.is_floating_point():
            raise TypeError("reference_latent must be a floating-point tensor")
        if reference_latent.ndim != 1:
            raise ValueError("reference_latent must be rank one")
        if horizon < 1:
            raise ValueError("horizon must be positive")
        normalized_seeds = tuple(int(seed) for seed in seeds)
        if not normalized_seeds or len(set(normalized_seeds)) != len(normalized_seeds):
            raise ValueError("seeds must be non-empty and unique")
        outputs = tuple(
            self._sample(context, reference_latent, horizon=horizon, seed=seed)
            for seed in normalized_seeds
        )
        return ReferenceContinuationBatch(
            seeds=normalized_seeds,
            horizon=horizon,
            reference_latent=reference_latent.detach().cpu().clone(),
            outputs=outputs,
            compute=self.ledger.snapshot(),
        )

    def directional_from_reference(
        self,
        context: Any,
        candidate_latent: Tensor,
        reference: ReferenceContinuationBatch,
    ) -> DirectionalJointKL:
        if not isinstance(reference, ReferenceContinuationBatch):
            raise TypeError("reference must be a ReferenceContinuationBatch")
        if reference.reference_latent.shape != candidate_latent.shape:
            raise ValueError("reference and candidate latents must have identical shapes")

        candidate_outputs: list[ContinuationOutput] = []
        valid_step_values: list[int] = []
        step_values: list[Tensor] = []
        total_values: list[Tensor] = []
        mean_values: list[Tensor] = []
        first_values: list[Tensor] = []
        tail_values: list[Tensor] = []
        for seed, reference_output in zip(reference.seeds, reference.outputs):
            candidate = self._force(
                context,
                candidate_latent,
                token_ids=reference_output.token_ids,
                seed=seed,
            )
            per_step = stable_categorical_kl_from_logits(
                self._distribution_logits(reference_output),
                self._distribution_logits(candidate),
            ).to(torch.float64).clamp_min(0.0)
            candidate_outputs.append(candidate)
            valid_steps = int(per_step.numel())
            padded_steps = torch.zeros(reference.horizon, dtype=torch.float64)
            padded_steps[:valid_steps] = per_step.detach().cpu()
            valid_step_values.append(valid_steps)
            step_values.append(padded_steps)
            total_values.append(per_step.sum().detach().cpu())
            mean_values.append(per_step.mean().detach().cpu())
            first_values.append(per_step[0].detach().cpu())
            tail_values.append(per_step[1:].sum().detach().cpu())
        return DirectionalJointKL(
            seeds=reference.seeds,
            per_seed_valid_steps=torch.tensor(valid_step_values, dtype=torch.long),
            per_seed_step_kl=torch.stack(step_values),
            per_seed_total_kl=torch.stack(total_values),
            per_seed_mean_kl=torch.stack(mean_values),
            per_seed_first_step_kl=torch.stack(first_values),
            per_seed_tail_kl=torch.stack(tail_values),
            reference_outputs=reference.outputs,
            forced_candidate_outputs=tuple(candidate_outputs),
            compute=self.ledger.snapshot(),
        )

    def symmetric(
        self,
        context: Any,
        left_latent: Tensor,
        right_latent: Tensor,
        *,
        horizon: int,
        seeds: Sequence[int],
    ) -> SymmetricJointKL:
        forward = self.directional(
            context,
            left_latent,
            right_latent,
            horizon=horizon,
            seeds=seeds,
        )
        reverse = self.directional(
            context,
            right_latent,
            left_latent,
            horizon=horizon,
            seeds=seeds,
        )
        return SymmetricJointKL(forward=forward, reverse=reverse)
