"""Source-faithful fake recursive adapter for the P1 preflight.

This module joins three pinned serving behaviors without loading a checkpoint:

1. the unmodified official sampler body is replayed on fake logits;
2. the scheduler rule is ``append proxy -> check_finished -> update_latent``;
3. the model input is the source's weighted top-k embedding, with ``-100``
   sentinel rows denoting a hard token.

It is deliberately restricted to the released platform's literal latent-end id
``524``.  The official sampler hard-codes that id in its noisy-branch fallback,
while the scheduler separately reads ``sampling_params.latent_end_token_id``.
Allowing a configurable substitute here would hide a real source mismatch.

No checkpoint, tokenizer, serving cache, reward, risk label, or training loss
is used.  The fake cache below exists only to test recursive ownership and
automatic differentiation through a candidate's independently recomputed
suffix.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Protocol

import torch
from torch import Tensor

from .p1_fake_preflight import (
    FinishEvent,
    LatentActionKind,
    LatentRequestState,
    StopConfig,
    check_finished,
)
from .p1_official_sampler_replay import OfficialSamplerReplay, replay_pinned_sampler_on_fake_request
from .p1_source_sampler_contract import SourceLatentSamplerConfig


SOURCE_LITERAL_LATENT_END_ID = 524


class SourceClosureEndpoint(str, Enum):
    """Endpoints admissible during the fake latent closure."""

    EXIT_VISIBLE = "LATENT_EXIT_VISIBLE"
    TIMEOUT = "LATENT_TIMEOUT"


@dataclass(frozen=True)
class SourceAdapterState:
    """Minimal request and independent-cache state required by the adapter."""

    output_ids: tuple[int, ...]
    decoded_text: str
    latent_mode: bool
    cache: Tensor
    to_abort: bool = False
    abort_message: str | None = None

    def request_view(self) -> LatentRequestState:
        return LatentRequestState(
            output_ids=self.output_ids,
            decoded_text=self.decoded_text,
            latent_mode=self.latent_mode,
            to_abort=self.to_abort,
            abort_message=self.abort_message,
        )


@dataclass(frozen=True)
class SourceActionProposal:
    """Sampler record plus the embedding proposed for the first latent action."""

    proxy: int
    proposed_embedding: Tensor
    topk_indices: Tensor
    topk_probs: Tensor
    topk_gumbels: Tensor
    original_topk_indices: Tensor
    original_topk_probs: Tensor
    rng_seed: int
    externally_forced_embedding: bool = False

    def with_forced_embedding(self, embedding: Tensor) -> "SourceActionProposal":
        if embedding.shape != self.proposed_embedding.shape:
            raise ValueError("forced embedding shape must match source proposal")
        return replace(
            self,
            proposed_embedding=embedding,
            externally_forced_embedding=True,
        )


@dataclass(frozen=True)
class SourceActionExecution:
    """The source scheduler's executed action after generic stopping."""

    proposal: SourceActionProposal
    state: SourceAdapterState
    kind: LatentActionKind
    consumed_embedding: Tensor | None
    executed_topk_indices: Tensor | None
    executed_topk_probs: Tensor | None
    finish_event: FinishEvent | None = None


@dataclass(frozen=True)
class SourceClosureRecord:
    latent_step: int
    execution: SourceActionExecution
    cache_before: Tensor
    cache_after: Tensor | None


@dataclass(frozen=True)
class SourceRecursiveClosure:
    endpoint: str
    state: SourceAdapterState
    records: tuple[SourceClosureRecord, ...]

    @property
    def consumed_actions(self) -> int:
        return sum(record.cache_after is not None for record in self.records)


class LatentCacheModel(Protocol):
    """Minimal differentiable model surface used by the fake recursive gate."""

    embedding_table: Tensor

    def logits(self, cache: Tensor) -> Tensor: ...

    def advance(self, cache: Tensor, consumed_embedding: Tensor) -> Tensor: ...


class IsolatedSamplerRNG:
    """Candidate-owned source sampler seed stream that leaves global RNG intact."""

    def __init__(self, seed: int) -> None:
        self._generator = torch.Generator(device="cpu").manual_seed(seed)

    def next_seed(self) -> int:
        return int(torch.randint(1, 2**31 - 1, (1,), generator=self._generator).item())


def _validate_source_config(config: SourceLatentSamplerConfig, embedding_table: Tensor) -> None:
    if config.latent_end_token_id != SOURCE_LITERAL_LATENT_END_ID:
        raise ValueError(
            "source-equivalent adapter requires literal latent_end_token_id=524; "
            "the pinned sampler hard-codes this id"
        )
    if embedding_table.ndim != 2 or embedding_table.shape[0] <= SOURCE_LITERAL_LATENT_END_ID:
        raise ValueError("embedding table must include the literal source row 524")
    if embedding_table.shape[0] < config.max_topk:
        raise ValueError("embedding table is smaller than max_topk")


def source_weighted_embedding(
    embedding_table: Tensor, topk_probs: Tensor, topk_indices: Tensor
) -> Tensor:
    """Mirror ``VocabParallelEmbedding.weighted_forward`` for one fake item."""

    if topk_probs.ndim != 1 or topk_indices.ndim != 1 or topk_probs.shape != topk_indices.shape:
        raise ValueError("top-k probabilities and ids must be matching rank-1 tensors")
    if topk_indices.numel() < 1 or int(topk_indices[0].item()) < 0:
        raise ValueError("the first source top-k id must be a valid hard token")
    if embedding_table.ndim != 2:
        raise ValueError("embedding_table must be rank 2")
    if torch.any((topk_indices < -100) | (topk_indices >= embedding_table.shape[0])):
        raise ValueError("top-k ids must be valid ids or the -100 hard-token sentinel")

    hard = bool(torch.all(topk_indices[1:] == -100).item())
    safe_indices = torch.where(
        topk_indices == -100,
        topk_indices[0].expand_as(topk_indices),
        topk_indices,
    )
    selected = embedding_table[safe_indices.long()]
    if hard:
        return selected[0]
    return torch.sum(topk_probs.to(dtype=selected.dtype).unsqueeze(-1) * selected, dim=0).to(
        selected.dtype
    )


def propose_source_action(
    logits: Tensor,
    embedding_table: Tensor,
    config: SourceLatentSamplerConfig,
    rng: IsolatedSamplerRNG,
) -> SourceActionProposal:
    """Obtain one proposal from the unmodified pinned sampler with isolated RNG."""

    _validate_source_config(config, embedding_table)
    if logits.ndim != 1 or logits.numel() != embedding_table.shape[0]:
        raise ValueError("one fake logit vector must match the embedding vocabulary")
    seed = rng.next_seed()
    # The source uses global CPU RNG in ``empty_like(...).exponential_()``.  A
    # fork makes each candidate's retained seed replayable without contaminating
    # other experiments or another candidate's stream.
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(seed)
        replay: OfficialSamplerReplay = replay_pinned_sampler_on_fake_request(logits, config)
    proposed_embedding = source_weighted_embedding(
        embedding_table, replay.topk_probs, replay.topk_indices
    )
    return SourceActionProposal(
        proxy=replay.next_token_id,
        proposed_embedding=proposed_embedding,
        topk_indices=replay.topk_indices,
        topk_probs=replay.topk_probs,
        topk_gumbels=replay.topk_gumbels,
        original_topk_indices=replay.topk_original_indices,
        original_topk_probs=replay.topk_original_probs,
        rng_seed=seed,
    )


def execute_source_scheduled_action(
    state: SourceAdapterState,
    proposal: SourceActionProposal,
    *,
    embedding_table: Tensor,
    stop_config: StopConfig,
) -> SourceActionExecution:
    """Apply the exact scheduler precedence then source latent-end overwrite."""

    if not state.latent_mode:
        raise ValueError("latent closure cannot execute another action after visible mode")
    appended_request = LatentRequestState(
        output_ids=state.output_ids + (proposal.proxy,),
        decoded_text=state.decoded_text,
        latent_mode=True,
        to_abort=state.to_abort,
        abort_message=state.abort_message,
    )
    after_append = replace(
        state,
        output_ids=appended_request.output_ids,
        latent_mode=True,
    )
    finish_event = check_finished(appended_request, stop_config)
    if finish_event is not None:
        return SourceActionExecution(
            proposal=proposal,
            state=after_append,
            kind=LatentActionKind.STOP,
            consumed_embedding=None,
            executed_topk_indices=None,
            executed_topk_probs=None,
            finish_event=finish_event,
        )

    if proposal.proxy == SOURCE_LITERAL_LATENT_END_ID:
        executed_indices = torch.full_like(proposal.topk_indices, -100)
        executed_indices[0] = SOURCE_LITERAL_LATENT_END_ID
        executed_probs = torch.zeros_like(proposal.topk_probs)
        executed_probs[0] = 1.0
        return SourceActionExecution(
            proposal=proposal,
            state=replace(after_append, latent_mode=False),
            kind=LatentActionKind.EXIT_VISIBLE,
            consumed_embedding=source_weighted_embedding(
                embedding_table, executed_probs, executed_indices
            ),
            executed_topk_indices=executed_indices,
            executed_topk_probs=executed_probs,
        )

    return SourceActionExecution(
        proposal=proposal,
        state=after_append,
        kind=LatentActionKind.CONSUME_LATENT,
        consumed_embedding=proposal.proposed_embedding,
        executed_topk_indices=proposal.topk_indices,
        executed_topk_probs=proposal.topk_probs,
    )


class SourceRecursiveAdapter:
    """Close a candidate-owned latent suffix without teacher forcing."""

    def __init__(
        self,
        model: LatentCacheModel,
        *,
        deterministic_config: SourceLatentSamplerConfig,
        stop_config: StopConfig,
    ) -> None:
        _validate_source_config(deterministic_config, model.embedding_table)
        if deterministic_config.add_noise_gumbel_softmax:
            raise ValueError("future recursive closure must use deterministic no-noise source actions")
        self.model = model
        self.deterministic_config = deterministic_config
        self.stop_config = stop_config

    def close(
        self,
        state: SourceAdapterState,
        first_action: SourceActionProposal,
        *,
        max_latent_steps: int,
        rng: IsolatedSamplerRNG,
    ) -> SourceRecursiveClosure:
        if max_latent_steps < 1:
            raise ValueError("max_latent_steps must be positive")
        if not state.latent_mode:
            raise ValueError("closure must start in latent mode")
        current_state = state
        proposal = first_action
        records: list[SourceClosureRecord] = []
        for latent_step in range(max_latent_steps):
            cache_before = current_state.cache
            execution = execute_source_scheduled_action(
                current_state,
                proposal,
                embedding_table=self.model.embedding_table,
                stop_config=self.stop_config,
            )
            if execution.kind is LatentActionKind.STOP:
                records.append(
                    SourceClosureRecord(latent_step, execution, cache_before, None)
                )
                event = execution.finish_event
                assert event is not None
                return SourceRecursiveClosure(
                    endpoint=event.kind.value,
                    state=execution.state,
                    records=tuple(records),
                )

            assert execution.consumed_embedding is not None
            # Even a latent-end proxy must be consumed as hard E_524 on the
            # following source model step before visible token one is observed.
            cache_after = self.model.advance(cache_before, execution.consumed_embedding)
            current_state = replace(execution.state, cache=cache_after)
            records.append(SourceClosureRecord(latent_step, execution, cache_before, cache_after))
            if execution.kind is LatentActionKind.EXIT_VISIBLE:
                return SourceRecursiveClosure(
                    endpoint=SourceClosureEndpoint.EXIT_VISIBLE.value,
                    state=current_state,
                    records=tuple(records),
                )
            proposal = propose_source_action(
                self.model.logits(cache_after),
                self.model.embedding_table,
                self.deterministic_config,
                rng,
            )
        return SourceRecursiveClosure(
            endpoint=SourceClosureEndpoint.TIMEOUT.value,
            state=current_state,
            records=tuple(records),
        )


class ToySourceCacheModel:
    """Small differentiable recurrence used only for source-adapter contracts."""

    def __init__(self) -> None:
        table = torch.zeros((525, 3), dtype=torch.float64)
        table[0] = torch.tensor([0.7, -0.1, 0.3], dtype=torch.float64)
        table[1] = torch.tensor([-0.4, 0.8, 0.2], dtype=torch.float64)
        table[2] = torch.tensor([0.1, 0.2, -0.6], dtype=torch.float64)
        table[3] = torch.tensor([0.3, -0.5, 0.7], dtype=torch.float64)
        table[524] = torch.tensor([0.9, 0.9, -0.9], dtype=torch.float64)
        self.embedding_table = table
        self.transition_matrix = torch.tensor(
            [[0.45, -0.10, 0.20], [0.05, 0.35, -0.15], [-0.20, 0.10, 0.40]],
            dtype=torch.float64,
        )
        self.action_matrix = torch.tensor(
            [[0.30, 0.15, -0.10], [-0.20, 0.25, 0.10], [0.05, -0.15, 0.35]],
            dtype=torch.float64,
        )
        self.logit_matrix = torch.tensor(
            [[0.20, -0.10, 0.10], [-0.15, 0.20, -0.20], [0.10, -0.30, 0.15], [-0.05, 0.10, -0.25]],
            dtype=torch.float64,
        )
        self.logit_bias = torch.tensor([1.20, 0.15, -0.10, -0.15], dtype=torch.float64)

    def logits(self, cache: Tensor) -> Tensor:
        leading = self.logit_matrix @ cache + self.logit_bias
        middle = torch.full((520,), -9.0, dtype=cache.dtype, device=cache.device)
        end = torch.full((1,), -8.0, dtype=cache.dtype, device=cache.device)
        return torch.cat((leading, middle, end), dim=0)

    def advance(self, cache: Tensor, consumed_embedding: Tensor) -> Tensor:
        return torch.tanh(
            self.transition_matrix.to(cache) @ cache
            + self.action_matrix.to(cache) @ consumed_embedding.to(cache)
        )
