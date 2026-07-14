from __future__ import annotations

import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import Tensor

try:
    import numpy as np
except ImportError:  # pragma: no cover - NumPy is optional for the core package.
    np = None


@dataclass(frozen=True)
class RNGSnapshot:
    python_state: object
    torch_cpu_state: Tensor
    torch_cuda_states: tuple[Tensor, ...]
    numpy_state: object | None

    @classmethod
    def capture(cls) -> "RNGSnapshot":
        cuda_states: tuple[Tensor, ...] = ()
        if torch.cuda.is_available():
            cuda_states = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        numpy_state = np.random.get_state() if np is not None else None
        return cls(
            python_state=random.getstate(),
            torch_cpu_state=torch.random.get_rng_state().clone(),
            torch_cuda_states=cuda_states,
            numpy_state=numpy_state,
        )

    def restore(self) -> None:
        random.setstate(self.python_state)
        torch.random.set_rng_state(self.torch_cpu_state)
        if self.torch_cuda_states and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(list(self.torch_cuda_states))
        if self.numpy_state is not None and np is not None:
            np.random.set_state(self.numpy_state)


def seed_all(seed: int) -> None:
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if np is not None:
        np.random.seed(seed % (2**32))


@contextmanager
def preserved_rng_state() -> Iterator[None]:
    snapshot = RNGSnapshot.capture()
    try:
        yield
    finally:
        snapshot.restore()


@contextmanager
def seeded_rng(seed: int) -> Iterator[None]:
    """Run a block with a fixed global RNG state without leaking state changes."""

    with preserved_rng_state():
        seed_all(seed)
        yield


def make_generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    resolved = torch.device(device)
    generator_device = resolved if resolved.type == "cuda" else torch.device("cpu")
    return torch.Generator(device=generator_device).manual_seed(seed)

