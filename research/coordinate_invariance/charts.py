from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ChartDiagnostics:
    """Numerical evidence that a chart preserves the represented latent value."""

    chart_name: str
    dimension: int
    max_abs_round_trip_error: float
    relative_l2_round_trip_error: float
    all_finite: bool


class LatentChart(nn.Module, ABC):
    """An invertible coordinate chart for the last dimension of a latent tensor."""

    def __init__(self, dimension: int, name: str) -> None:
        super().__init__()
        if dimension < 1:
            raise ValueError("dimension must be positive")
        if not name:
            raise ValueError("chart name must be non-empty")
        self.dimension = int(dimension)
        self.name = name

    def _validate_input(self, value: Tensor, argument_name: str) -> None:
        if not isinstance(value, Tensor):
            raise TypeError(f"{argument_name} must be a torch.Tensor")
        if value.ndim < 1 or value.shape[-1] != self.dimension:
            raise ValueError(
                f"{argument_name} must have last dimension {self.dimension}, "
                f"got shape {tuple(value.shape)}"
            )
        if not value.is_floating_point():
            raise TypeError(f"{argument_name} must use a floating-point dtype")
        if value.numel() == 0:
            raise ValueError(f"{argument_name} must not be empty")
        if not torch.isfinite(value).all():
            raise ValueError(f"{argument_name} contains non-finite values")

    @abstractmethod
    def encode(self, native: Tensor) -> Tensor:
        """Map native latent coordinates ``z`` to chart coordinates ``u``."""

    @abstractmethod
    def decode(self, charted: Tensor) -> Tensor:
        """Map chart coordinates ``u`` back to native coordinates ``z``."""

    @abstractmethod
    def log_abs_det_jacobian(self, native: Tensor) -> Tensor:
        """Return ``log |det d phi(z) / dz|`` for each latent vector."""

    def forward(self, native: Tensor) -> Tensor:
        return self.encode(native)

    def diagnose(self, native: Tensor) -> ChartDiagnostics:
        self._validate_input(native, "native")
        reconstructed = self.decode(self.encode(native))
        error = reconstructed - native
        denominator = torch.linalg.vector_norm(native).clamp_min(
            torch.finfo(native.dtype).tiny
        )
        relative_error = torch.linalg.vector_norm(error) / denominator
        return ChartDiagnostics(
            chart_name=self.name,
            dimension=self.dimension,
            max_abs_round_trip_error=float(error.abs().max().detach().cpu()),
            relative_l2_round_trip_error=float(relative_error.detach().cpu()),
            all_finite=bool(torch.isfinite(reconstructed).all().item()),
        )


def _random_orthogonal(
    dimension: int, generator: torch.Generator, dtype: torch.dtype
) -> Tensor:
    raw = torch.randn(dimension, dimension, generator=generator, dtype=dtype)
    q, r = torch.linalg.qr(raw)
    diagonal = torch.diagonal(r)
    signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
    return q * signs.unsqueeze(0)


class AffineChart(LatentChart):
    """An exact affine chart ``u = A z + b`` with a full-rank matrix ``A``."""

    def __init__(
        self,
        matrix: Tensor,
        bias: Tensor | None = None,
        *,
        name: str = "affine",
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        if not isinstance(matrix, Tensor) or matrix.ndim != 2:
            raise TypeError("matrix must be a rank-2 torch.Tensor")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be square")
        if not matrix.is_floating_point():
            raise TypeError("matrix must use a floating-point dtype")
        if not torch.isfinite(matrix).all():
            raise ValueError("matrix contains non-finite values")

        dimension = int(matrix.shape[0])
        super().__init__(dimension=dimension, name=name)
        if compute_dtype is not None and not torch.empty((), dtype=compute_dtype).is_floating_point():
            raise TypeError("compute_dtype must be a floating-point dtype")
        self.compute_dtype = compute_dtype

        matrix_copy = matrix.detach().clone()
        singular_values = torch.linalg.svdvals(matrix_copy.to(torch.float64))
        largest = singular_values.max()
        smallest = singular_values.min()
        numerical_floor = torch.finfo(torch.float64).eps * dimension * largest
        if smallest <= numerical_floor:
            raise ValueError("matrix must be numerically full rank")

        if bias is None:
            bias_copy = torch.zeros(dimension, dtype=matrix_copy.dtype, device=matrix_copy.device)
        else:
            if not isinstance(bias, Tensor) or bias.shape != (dimension,):
                raise ValueError(f"bias must have shape ({dimension},)")
            if not bias.is_floating_point() or not torch.isfinite(bias).all():
                raise ValueError("bias must contain finite floating-point values")
            bias_copy = bias.detach().clone().to(
                dtype=matrix_copy.dtype, device=matrix_copy.device
            )

        self.register_buffer("matrix", matrix_copy)
        self.register_buffer("bias", bias_copy)
        self._condition_number = float((largest / smallest).cpu())

    @property
    def condition_number(self) -> float:
        return self._condition_number

    def _typed_parameters(self, reference: Tensor) -> tuple[Tensor, Tensor]:
        dtype = self.compute_dtype if self.compute_dtype is not None else reference.dtype
        return (
            self.matrix.to(dtype=dtype, device=reference.device),
            self.bias.to(dtype=dtype, device=reference.device),
        )

    def encode(self, native: Tensor) -> Tensor:
        self._validate_input(native, "native")
        matrix, bias = self._typed_parameters(native)
        charted = torch.matmul(
            native.to(dtype=matrix.dtype), matrix.transpose(-1, -2)
        ) + bias
        if not torch.isfinite(charted).all():
            raise FloatingPointError("affine chart produced non-finite coordinates")
        return charted

    def decode(self, charted: Tensor) -> Tensor:
        self._validate_input(charted, "charted")
        matrix, bias = self._typed_parameters(charted)
        right_hand_side = (charted - bias).reshape(-1, self.dimension).transpose(0, 1)
        native = torch.linalg.solve(matrix, right_hand_side)
        native = native.transpose(0, 1).reshape_as(charted)
        if not torch.isfinite(native).all():
            raise FloatingPointError("affine chart inverse produced non-finite coordinates")
        return native

    def log_abs_det_jacobian(self, native: Tensor) -> Tensor:
        self._validate_input(native, "native")
        matrix, _ = self._typed_parameters(native)
        sign, log_abs_det = torch.linalg.slogdet(matrix)
        if sign == 0:
            raise FloatingPointError("affine chart has a singular Jacobian")
        return log_abs_det.expand(native.shape[:-1])

    @classmethod
    def identity(
        cls,
        dimension: int,
        *,
        dtype: torch.dtype = torch.float64,
        compute_dtype: torch.dtype | None = None,
    ) -> "AffineChart":
        return cls(
            torch.eye(dimension, dtype=dtype),
            name="identity",
            compute_dtype=compute_dtype,
        )

    @classmethod
    def random_orthogonal(
        cls,
        dimension: int,
        *,
        seed: int,
        dtype: torch.dtype = torch.float64,
        bias_scale: float = 0.0,
        compute_dtype: torch.dtype | None = None,
    ) -> "AffineChart":
        if bias_scale < 0 or not math.isfinite(bias_scale):
            raise ValueError("bias_scale must be finite and non-negative")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        matrix = _random_orthogonal(dimension, generator, dtype)
        bias = bias_scale * torch.randn(dimension, generator=generator, dtype=dtype)
        return cls(
            matrix,
            bias,
            name=f"orthogonal-seed-{seed}",
            compute_dtype=compute_dtype,
        )

    @classmethod
    def with_condition_number(
        cls,
        dimension: int,
        condition_number: float,
        *,
        seed: int,
        dtype: torch.dtype = torch.float64,
        bias_scale: float = 0.0,
        compute_dtype: torch.dtype | None = None,
    ) -> "AffineChart":
        if dimension < 2 and condition_number != 1.0:
            raise ValueError("non-unit condition numbers require dimension >= 2")
        if condition_number < 1.0 or not math.isfinite(condition_number):
            raise ValueError("condition_number must be finite and at least one")
        if bias_scale < 0 or not math.isfinite(bias_scale):
            raise ValueError("bias_scale must be finite and non-negative")

        generator = torch.Generator(device="cpu").manual_seed(seed)
        left = _random_orthogonal(dimension, generator, dtype)
        right = _random_orthogonal(dimension, generator, dtype)
        half_log_condition = 0.5 * math.log(condition_number)
        singular_values = torch.exp(
            torch.linspace(
                -half_log_condition,
                half_log_condition,
                dimension,
                dtype=dtype,
            )
        )
        matrix = left @ torch.diag(singular_values) @ right.transpose(0, 1)
        bias = bias_scale * torch.randn(dimension, generator=generator, dtype=dtype)
        return cls(
            matrix,
            bias,
            name=f"affine-cond-{condition_number:g}-seed-{seed}",
            compute_dtype=compute_dtype,
        )


class SinhChart(LatentChart):
    """A smooth nonlinear chart with an analytic inverse.

    For each coordinate, ``u_i = sinh(scale_i * z_i) / scale_i``. The scales
    must be positive. Moderate scales and normalized latent inputs should be used
    to avoid overflow; non-finite results are rejected rather than silently used.
    """

    def __init__(
        self,
        dimension: int,
        scale: float | Sequence[float] | Tensor = 0.5,
        *,
        name: str = "sinh",
        dtype: torch.dtype = torch.float64,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(dimension=dimension, name=name)
        if compute_dtype is not None and not torch.empty((), dtype=compute_dtype).is_floating_point():
            raise TypeError("compute_dtype must be a floating-point dtype")
        self.compute_dtype = compute_dtype
        if isinstance(scale, Tensor):
            scale_tensor = scale.detach().clone().to(dtype=dtype)
        elif isinstance(scale, Sequence):
            scale_tensor = torch.tensor(list(scale), dtype=dtype)
        else:
            scale_tensor = torch.full((dimension,), float(scale), dtype=dtype)
        if scale_tensor.shape != (dimension,):
            raise ValueError(f"scale must be scalar or have shape ({dimension},)")
        if not torch.isfinite(scale_tensor).all() or (scale_tensor <= 0).any():
            raise ValueError("all scale values must be finite and positive")
        self.register_buffer("scale", scale_tensor)

    def _typed_scale(self, reference: Tensor) -> Tensor:
        dtype = self.compute_dtype if self.compute_dtype is not None else reference.dtype
        return self.scale.to(dtype=dtype, device=reference.device)

    def encode(self, native: Tensor) -> Tensor:
        self._validate_input(native, "native")
        scale = self._typed_scale(native)
        charted = torch.sinh(scale * native.to(dtype=scale.dtype)) / scale
        if not torch.isfinite(charted).all():
            raise FloatingPointError("sinh chart produced non-finite coordinates")
        return charted

    def decode(self, charted: Tensor) -> Tensor:
        self._validate_input(charted, "charted")
        scale = self._typed_scale(charted)
        native = torch.asinh(scale * charted) / scale
        if not torch.isfinite(native).all():
            raise FloatingPointError("sinh chart inverse produced non-finite coordinates")
        return native

    def log_abs_det_jacobian(self, native: Tensor) -> Tensor:
        self._validate_input(native, "native")
        scaled = self._typed_scale(native) * native
        absolute = scaled.abs()
        log_cosh = absolute + torch.nn.functional.softplus(-2.0 * absolute) - math.log(2.0)
        return log_cosh.sum(dim=-1)

    def local_condition_number(self, native: Tensor) -> Tensor:
        self._validate_input(native, "native")
        derivative = torch.cosh(self._typed_scale(native) * native)
        return derivative.max(dim=-1).values / derivative.min(dim=-1).values
