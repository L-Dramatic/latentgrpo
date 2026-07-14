from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from threading import Lock
from typing import Iterator


@dataclass(frozen=True)
class ComputeSnapshot:
    model_forward_calls: int
    rollout_calls: int
    backward_calls: int
    prompt_tokens: int
    generated_tokens: int
    wall_time_seconds: float
    section_wall_time_seconds: dict[str, float]
    custom_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ComputeLedger:
    """Thread-safe counters for matched-compute reporting."""

    model_forward_calls: int = 0
    rollout_calls: int = 0
    backward_calls: int = 0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    wall_time_seconds: float = 0.0
    section_wall_time_seconds: dict[str, float] = field(default_factory=dict)
    custom_counts: dict[str, int] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False, compare=False)

    def record_rollout(
        self,
        *,
        prompt_tokens: int,
        generated_tokens: int,
        model_forward_calls: int = 1,
    ) -> None:
        values = (prompt_tokens, generated_tokens, model_forward_calls)
        if any(not isinstance(value, int) or value < 0 for value in values):
            raise ValueError("compute counts must be non-negative integers")
        with self._lock:
            self.rollout_calls += 1
            self.prompt_tokens += prompt_tokens
            self.generated_tokens += generated_tokens
            self.model_forward_calls += model_forward_calls

    def record_backward(self, count: int = 1) -> None:
        if not isinstance(count, int) or count < 0:
            raise ValueError("count must be a non-negative integer")
        with self._lock:
            self.backward_calls += count

    def increment(self, name: str, count: int = 1) -> None:
        if not name:
            raise ValueError("custom counter name must be non-empty")
        if not isinstance(count, int) or count < 0:
            raise ValueError("count must be a non-negative integer")
        with self._lock:
            self.custom_counts[name] = self.custom_counts.get(name, 0) + count

    @contextmanager
    def timed(self, section: str) -> Iterator[None]:
        if not section:
            raise ValueError("section must be non-empty")
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            with self._lock:
                self.wall_time_seconds += elapsed
                self.section_wall_time_seconds[section] = (
                    self.section_wall_time_seconds.get(section, 0.0) + elapsed
                )

    def snapshot(self) -> ComputeSnapshot:
        with self._lock:
            return ComputeSnapshot(
                model_forward_calls=self.model_forward_calls,
                rollout_calls=self.rollout_calls,
                backward_calls=self.backward_calls,
                prompt_tokens=self.prompt_tokens,
                generated_tokens=self.generated_tokens,
                wall_time_seconds=self.wall_time_seconds,
                section_wall_time_seconds=dict(self.section_wall_time_seconds),
                custom_counts=dict(self.custom_counts),
            )

