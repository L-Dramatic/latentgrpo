"""CPU-only contracts for the P1 latent-action adapter.

This module is deliberately a toy implementation.  It does not load a public
checkpoint and it is not the future source-equivalent adapter.  Its purpose is
to make the P1 execution order, structural endpoints, recursive closure, and
directional-JVP requirements executable before checkpoint access is authorized.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Sequence

import torch
from torch import Tensor


class FinishKind(str, Enum):
    """Source-style request termination reasons relevant to P1."""

    ABORT = "EXECUTION_ABORT"
    LENGTH = "LATENT_FINISH_LENGTH"
    TOKEN = "LATENT_FINISH_TOKEN"
    STRING = "LATENT_FINISH_STRING"


class LatentActionKind(str, Enum):
    """What was actually executed after a proposed latent proxy."""

    STOP = "STOP"
    EXIT_VISIBLE = "EXIT_VISIBLE"
    CONSUME_LATENT = "CONSUME_LATENT"


@dataclass(frozen=True)
class FinishEvent:
    kind: FinishKind
    detail: str | int | None = None


@dataclass(frozen=True)
class StopConfig:
    """Frozen generic stopping semantics used before latent execution.

    This mirrors the relevant source ordering: abort, total length, token/EOS
    checks (unless ``ignore_eos``), then stop strings.  Stop strings remain
    active when ``ignore_eos`` is true, matching the audited scheduler.
    """

    max_new_tokens: int
    ignore_eos: bool = False
    stop_token_ids: frozenset[int] = frozenset()
    eos_token_ids: frozenset[int] = frozenset()
    tokenizer_eos_token_id: int | None = None
    additional_stop_token_ids: frozenset[int] = frozenset()
    stop_strings: tuple[str, ...] = ()
    stop_str_max_len: int = 0
    decode: Callable[[Sequence[int]], str] = field(
        default=lambda ids: " ".join(str(token_id) for token_id in ids)
    )

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.stop_str_max_len < 0:
            raise ValueError("stop_str_max_len must be non-negative")
        if any(not isinstance(token_id, int) for token_id in self.all_token_stops):
            raise TypeError("all stop ids must be integers")
        if any(not stop for stop in self.stop_strings):
            raise ValueError("stop strings must be non-empty")

    @property
    def all_token_stops(self) -> frozenset[int]:
        values = set(self.stop_token_ids) | set(self.eos_token_ids)
        values |= set(self.additional_stop_token_ids)
        if self.tokenizer_eos_token_id is not None:
            values.add(self.tokenizer_eos_token_id)
        return frozenset(values)


@dataclass(frozen=True)
class LatentRequestState:
    """Minimal immutable request state before a proposed proxy is appended."""

    output_ids: tuple[int, ...] = ()
    decoded_text: str = ""
    latent_mode: bool = True
    to_abort: bool = False
    abort_message: str | None = None


@dataclass(frozen=True)
class LatentExecution:
    """The post-stop-check execution record for one latent proxy."""

    kind: LatentActionKind
    state: LatentRequestState
    consumed_embedding: Tensor | None
    finish_event: FinishEvent | None = None

    @property
    def is_structural_endpoint(self) -> bool:
        return self.kind is LatentActionKind.STOP


def check_finished(
    state: LatentRequestState,
    config: StopConfig,
) -> FinishEvent | None:
    """Apply source-style generic stopping after a token/proxy was appended."""

    if state.to_abort:
        return FinishEvent(FinishKind.ABORT, state.abort_message)
    if len(state.output_ids) >= config.max_new_tokens:
        return FinishEvent(FinishKind.LENGTH, config.max_new_tokens)
    if not state.output_ids:
        return None

    last_token_id = state.output_ids[-1]
    if not config.ignore_eos and last_token_id in config.all_token_stops:
        return FinishEvent(FinishKind.TOKEN, last_token_id)

    if config.stop_strings:
        tail = state.output_ids[-(config.stop_str_max_len + 1) :]
        tail_text = config.decode(tail)
        for stop_string in config.stop_strings:
            if stop_string in tail_text or stop_string in state.decoded_text:
                return FinishEvent(FinishKind.STRING, stop_string)
    return None


def execute_latent_action(
    state: LatentRequestState,
    proposed_embedding: Tensor,
    proposed_proxy: int,
    *,
    latent_end_token_id: int,
    embedding_table: Tensor,
    stop_config: StopConfig,
) -> LatentExecution:
    """Append, stop-check, then execute a latent joint action.

    A generic stop preempts both a continuous embedding and the special latent
    exit embedding.  Only a non-stopping proxy equal to ``latent_end_token_id``
    executes the hard one-hot embedding and switches to visible mode.
    """

    if not state.latent_mode:
        raise ValueError("latent action cannot execute after visible mode begins")
    if not isinstance(proposed_proxy, int):
        raise TypeError("proposed_proxy must be an integer")
    if not isinstance(proposed_embedding, Tensor) or proposed_embedding.ndim != 1:
        raise ValueError("proposed_embedding must be a rank-1 tensor")
    if not proposed_embedding.is_floating_point():
        raise TypeError("proposed_embedding must be floating point")
    if embedding_table.ndim != 2 or embedding_table.shape[1] != proposed_embedding.numel():
        raise ValueError("embedding_table must match the proposed embedding width")
    if not 0 <= latent_end_token_id < embedding_table.shape[0]:
        raise ValueError("latent_end_token_id is outside the embedding table")

    appended = LatentRequestState(
        output_ids=state.output_ids + (proposed_proxy,),
        decoded_text=state.decoded_text,
        latent_mode=True,
        to_abort=state.to_abort,
        abort_message=state.abort_message,
    )
    finish_event = check_finished(appended, stop_config)
    if finish_event is not None:
        return LatentExecution(
            kind=LatentActionKind.STOP,
            state=appended,
            consumed_embedding=None,
            finish_event=finish_event,
        )
    if proposed_proxy == latent_end_token_id:
        visible_state = LatentRequestState(
            output_ids=appended.output_ids,
            decoded_text=appended.decoded_text,
            latent_mode=False,
        )
        return LatentExecution(
            kind=LatentActionKind.EXIT_VISIBLE,
            state=visible_state,
            consumed_embedding=embedding_table[latent_end_token_id],
        )
    return LatentExecution(
        kind=LatentActionKind.CONSUME_LATENT,
        state=appended,
        consumed_embedding=proposed_embedding,
    )


@dataclass(frozen=True)
class RecursiveActionRecord:
    latent_step: int
    proxy: int
    proposed_embedding: Tensor
    execution: LatentExecution


@dataclass(frozen=True)
class RecursiveClosure:
    terminal_kind: str
    state: LatentRequestState
    hidden: Tensor
    records: tuple[RecursiveActionRecord, ...]

    @property
    def consumed_latent_actions(self) -> int:
        return sum(
            record.execution.kind is LatentActionKind.CONSUME_LATENT
            for record in self.records
        )


class ToyRecursiveLatentPolicy:
    """Differentiable recurrence for CPU-only closure and JVP contract tests."""

    def __init__(self) -> None:
        self.embedding_table = torch.tensor(
            [
                [0.70, -0.10, 0.30],
                [-0.40, 0.80, 0.20],
                [0.10, 0.20, -0.60],
                [0.30, -0.50, 0.70],
                [-0.20, -0.30, 0.50],
                [0.90, 0.90, -0.90],
            ],
            dtype=torch.float64,
        )
        self.transition_matrix = torch.tensor(
            [[0.45, -0.10, 0.20], [0.05, 0.35, -0.15], [-0.20, 0.10, 0.40]],
            dtype=torch.float64,
        )
        self.action_matrix = torch.tensor(
            [[0.30, 0.15, -0.10], [-0.20, 0.25, 0.10], [0.05, -0.15, 0.35]],
            dtype=torch.float64,
        )
        # Token 0 is safely top-1 in the toy trace; token 5 is the latent end.
        self.logit_matrix = torch.tensor(
            [
                [0.20, -0.10, 0.10],
                [-0.15, 0.20, -0.20],
                [0.10, -0.30, 0.15],
                [-0.05, 0.10, -0.25],
                [0.15, 0.05, -0.10],
                [-0.30, -0.25, -0.20],
            ],
            dtype=torch.float64,
        )
        self.logit_bias = torch.tensor(
            [1.20, 0.15, -0.10, -0.15, -0.20, -1.50], dtype=torch.float64
        )

    @property
    def latent_end_token_id(self) -> int:
        return 5

    def logits(self, hidden: Tensor) -> Tensor:
        return self.logit_matrix @ hidden + self.logit_bias

    def deterministic_action(self, hidden: Tensor) -> tuple[Tensor, int]:
        logits = self.logits(hidden)
        weights = torch.softmax(logits, dim=0)
        embedding = weights @ self.embedding_table
        return embedding, int(torch.argmax(logits).item())

    def transition(self, hidden: Tensor, consumed_embedding: Tensor) -> Tensor:
        return torch.tanh(
            self.transition_matrix @ hidden + self.action_matrix @ consumed_embedding
        )

    def close(
        self,
        *,
        initial_hidden: Tensor,
        initial_embedding: Tensor,
        initial_proxy: int,
        max_latent_steps: int,
        stop_config: StopConfig,
    ) -> RecursiveClosure:
        """Run a candidate-owned deterministic closure without teacher forcing."""

        if max_latent_steps < 1:
            raise ValueError("max_latent_steps must be positive")
        hidden = initial_hidden
        state = LatentRequestState()
        records: list[RecursiveActionRecord] = []
        proposed_embedding = initial_embedding
        proposed_proxy = initial_proxy
        for latent_step in range(max_latent_steps):
            execution = execute_latent_action(
                state,
                proposed_embedding,
                proposed_proxy,
                latent_end_token_id=self.latent_end_token_id,
                embedding_table=self.embedding_table,
                stop_config=stop_config,
            )
            records.append(
                RecursiveActionRecord(
                    latent_step=latent_step,
                    proxy=proposed_proxy,
                    proposed_embedding=proposed_embedding,
                    execution=execution,
                )
            )
            state = execution.state
            if execution.kind is LatentActionKind.STOP:
                event = execution.finish_event
                assert event is not None
                return RecursiveClosure(
                    terminal_kind=event.kind.value,
                    state=state,
                    hidden=hidden,
                    records=tuple(records),
                )
            if execution.kind is LatentActionKind.EXIT_VISIBLE:
                return RecursiveClosure(
                    terminal_kind="LATENT_EXIT_VISIBLE",
                    state=state,
                    hidden=hidden,
                    records=tuple(records),
                )

            assert execution.consumed_embedding is not None
            hidden = self.transition(hidden, execution.consumed_embedding)
            proposed_embedding, proposed_proxy = self.deterministic_action(hidden)
        return RecursiveClosure(
            terminal_kind="LATENT_TIMEOUT",
            state=state,
            hidden=hidden,
            records=tuple(records),
        )


def directional_jvp(
    function: Callable[[Tensor], Tensor], point: Tensor, direction: Tensor
) -> Tensor:
    """Return the directional JVP and reject malformed fake-model interfaces."""

    if point.shape != direction.shape or point.ndim != 1:
        raise ValueError("point and direction must be matching rank-1 tensors")
    _, tangent = torch.autograd.functional.jvp(
        function,
        (point,),
        (direction,),
        create_graph=False,
        strict=True,
    )
    return tangent
