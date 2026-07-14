from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from .charts import AffineChart, LatentChart, SinhChart
from .metrics import multi_horizon_divergence
from .operations import (
    add_isotropic_noise_in_chart,
    coordinate_distances,
    interpolate_in_chart,
    nearest_neighbor_in_chart,
)


CONFIG_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ToyGateConfig:
    experiment_name: str
    dimension: int
    hidden_dimension: int
    horizon: int
    vocabulary_size: int
    query_count: int
    candidate_count: int
    pair_count: int
    model_seed: int
    data_seed: int
    chart_seed: int
    affine_condition_number: float
    noise_standard_deviation: float
    sinh_scale: float
    thresholds: Mapping[str, float]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ToyGateConfig":
        expected = {
            "schema_version",
            "experiment_name",
            "dimension",
            "hidden_dimension",
            "horizon",
            "vocabulary_size",
            "query_count",
            "candidate_count",
            "pair_count",
            "model_seed",
            "data_seed",
            "chart_seed",
            "affine_condition_number",
            "noise_standard_deviation",
            "sinh_scale",
            "thresholds",
        }
        if set(raw) != expected:
            missing = sorted(expected - set(raw))
            extra = sorted(set(raw) - expected)
            raise ValueError(f"unexpected config schema; missing={missing}, extra={extra}")
        if raw["schema_version"] != CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported toy gate config version")

        integer_fields = (
            "dimension",
            "hidden_dimension",
            "horizon",
            "vocabulary_size",
            "query_count",
            "candidate_count",
            "pair_count",
            "model_seed",
            "data_seed",
            "chart_seed",
        )
        for field_name in integer_fields:
            if not isinstance(raw[field_name], int):
                raise TypeError(f"{field_name} must be an integer")
        positive_fields = integer_fields[:7]
        for field_name in positive_fields:
            if raw[field_name] < 1:
                raise ValueError(f"{field_name} must be positive")
        if raw["dimension"] < 2 or raw["vocabulary_size"] < 2:
            raise ValueError("dimension and vocabulary_size must be at least two")
        if raw["affine_condition_number"] <= 1.0:
            raise ValueError("affine_condition_number must be greater than one")
        if raw["noise_standard_deviation"] <= 0.0 or raw["sinh_scale"] <= 0.0:
            raise ValueError("noise_standard_deviation and sinh_scale must be positive")

        expected_thresholds = {
            "max_round_trip_error",
            "max_behavior_identity_error",
            "max_functional_invariance_error",
            "max_orthogonal_neighbor_flip_rate",
            "min_anisotropic_neighbor_flip_rate",
            "min_anisotropic_ranking_disagreement",
            "min_noise_functional_gap",
            "min_nonlinear_interpolation_gap",
        }
        thresholds = raw["thresholds"]
        if not isinstance(thresholds, dict) or set(thresholds) != expected_thresholds:
            raise ValueError("thresholds have an unexpected schema")
        normalized_thresholds = {key: float(value) for key, value in thresholds.items()}
        if any(not math.isfinite(value) or value < 0 for value in normalized_thresholds.values()):
            raise ValueError("thresholds must be finite and non-negative")

        return cls(
            experiment_name=str(raw["experiment_name"]),
            dimension=raw["dimension"],
            hidden_dimension=raw["hidden_dimension"],
            horizon=raw["horizon"],
            vocabulary_size=raw["vocabulary_size"],
            query_count=raw["query_count"],
            candidate_count=raw["candidate_count"],
            pair_count=raw["pair_count"],
            model_seed=raw["model_seed"],
            data_seed=raw["data_seed"],
            chart_seed=raw["chart_seed"],
            affine_condition_number=float(raw["affine_condition_number"]),
            noise_standard_deviation=float(raw["noise_standard_deviation"]),
            sinh_scale=float(raw["sinh_scale"]),
            thresholds=normalized_thresholds,
        )


@dataclass(frozen=True)
class ToyContinuationSystem:
    input_weights: Tensor
    hidden_bias: Tensor
    output_weights: Tensor
    output_bias: Tensor

    @classmethod
    def create(cls, config: ToyGateConfig) -> "ToyContinuationSystem":
        generator = torch.Generator(device="cpu").manual_seed(config.model_seed)
        dtype = torch.float64
        input_weights = torch.randn(
            config.horizon,
            config.dimension,
            config.hidden_dimension,
            generator=generator,
            dtype=dtype,
        ) / math.sqrt(config.dimension)
        hidden_bias = 0.2 * torch.randn(
            config.horizon,
            config.hidden_dimension,
            generator=generator,
            dtype=dtype,
        )
        output_weights = torch.randn(
            config.horizon,
            config.hidden_dimension,
            config.vocabulary_size,
            generator=generator,
            dtype=dtype,
        ) / math.sqrt(config.hidden_dimension)
        output_bias = 0.2 * torch.randn(
            config.horizon,
            config.vocabulary_size,
            generator=generator,
            dtype=dtype,
        )
        return cls(input_weights, hidden_bias, output_weights, output_bias)

    def logits(self, latent: Tensor) -> Tensor:
        if latent.shape != (self.input_weights.shape[1],):
            raise ValueError("toy latent has an unexpected shape")
        hidden = torch.tanh(
            torch.einsum("d,hdm->hm", latent, self.input_weights) + self.hidden_bias
        )
        return torch.einsum("hm,hmv->hv", hidden, self.output_weights) + self.output_bias

    def distance(self, left: Tensor, right: Tensor) -> float:
        value = multi_horizon_divergence(self.logits(left), self.logits(right))
        return float(value.detach().cpu())


def _ranking_disagreement(left: Tensor, right: Tensor) -> float:
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("rank vectors must be one-dimensional and shape matched")
    upper = torch.triu_indices(left.numel(), left.numel(), offset=1)
    left_order = torch.sign(left[upper[0]] - left[upper[1]])
    right_order = torch.sign(right[upper[0]] - right[upper[1]])
    non_ties = (left_order != 0) & (right_order != 0)
    if not non_ties.any():
        return 0.0
    return float((left_order[non_ties] != right_order[non_ties]).to(torch.float64).mean())


def _max_behavior_identity_error(
    system: ToyContinuationSystem, chart: LatentChart, latents: Tensor
) -> float:
    maximum = 0.0
    for latent in latents:
        reconstructed = chart.decode(chart.encode(latent))
        error = (system.logits(reconstructed) - system.logits(latent)).abs().max()
        maximum = max(maximum, float(error))
    return maximum


def _max_functional_invariance_error(
    system: ToyContinuationSystem,
    chart: LatentChart,
    left_latents: Tensor,
    right_latents: Tensor,
) -> float:
    maximum = 0.0
    for left, right in zip(left_latents, right_latents):
        native_distance = system.distance(left, right)
        charted_distance = system.distance(
            chart.decode(chart.encode(left)), chart.decode(chart.encode(right))
        )
        maximum = max(maximum, abs(native_distance - charted_distance))
    return maximum


def run_toy_gate(config: ToyGateConfig) -> dict[str, object]:
    dtype = torch.float64
    data_generator = torch.Generator(device="cpu").manual_seed(config.data_seed)
    system = ToyContinuationSystem.create(config)
    queries = 0.75 * torch.randn(
        config.query_count, config.dimension, generator=data_generator, dtype=dtype
    )
    candidate_sets = 0.75 * torch.randn(
        config.query_count,
        config.candidate_count,
        config.dimension,
        generator=data_generator,
        dtype=dtype,
    )
    pair_left = 0.75 * torch.randn(
        config.pair_count, config.dimension, generator=data_generator, dtype=dtype
    )
    pair_right = 0.75 * torch.randn(
        config.pair_count, config.dimension, generator=data_generator, dtype=dtype
    )

    identity = AffineChart.identity(config.dimension, dtype=dtype)
    orthogonal = AffineChart.random_orthogonal(
        config.dimension, seed=config.chart_seed, dtype=dtype
    )
    anisotropic = AffineChart.with_condition_number(
        config.dimension,
        config.affine_condition_number,
        seed=config.chart_seed + 1,
        dtype=dtype,
    )
    nonlinear = SinhChart(
        config.dimension,
        scale=config.sinh_scale,
        dtype=dtype,
        name=f"sinh-scale-{config.sinh_scale:g}",
    )
    charts = (identity, orthogonal, anisotropic, nonlinear)

    diagnostics = {chart.name: chart.diagnose(queries) for chart in charts}
    max_round_trip_error = max(
        diagnostic.max_abs_round_trip_error for diagnostic in diagnostics.values()
    )
    behavior_identity_errors = {
        chart.name: _max_behavior_identity_error(system, chart, queries) for chart in charts
    }
    max_behavior_identity_error = max(behavior_identity_errors.values())
    functional_invariance_errors = {
        chart.name: _max_functional_invariance_error(
            system, chart, pair_left, pair_right
        )
        for chart in charts
    }
    max_functional_invariance_error = max(functional_invariance_errors.values())

    orthogonal_flips = 0
    anisotropic_flips = 0
    anisotropic_rank_disagreements: list[float] = []
    for query, candidates in zip(queries, candidate_sets):
        native_neighbor = nearest_neighbor_in_chart(query, candidates, identity).index
        orthogonal_neighbor = nearest_neighbor_in_chart(
            query, candidates, orthogonal
        ).index
        anisotropic_neighbor = nearest_neighbor_in_chart(
            query, candidates, anisotropic
        ).index
        orthogonal_flips += int(native_neighbor != orthogonal_neighbor)
        anisotropic_flips += int(native_neighbor != anisotropic_neighbor)
        native_distances = coordinate_distances(query, candidates, identity)
        anisotropic_distances = coordinate_distances(query, candidates, anisotropic)
        anisotropic_rank_disagreements.append(
            _ranking_disagreement(native_distances, anisotropic_distances)
        )
    orthogonal_neighbor_flip_rate = orthogonal_flips / config.query_count
    anisotropic_neighbor_flip_rate = anisotropic_flips / config.query_count
    anisotropic_ranking_disagreement = sum(anisotropic_rank_disagreements) / len(
        anisotropic_rank_disagreements
    )

    noise_functional_gaps: list[float] = []
    for index, native in enumerate(queries[: config.pair_count]):
        identity_generator = torch.Generator(device="cpu").manual_seed(
            config.data_seed + 10_000 + index
        )
        anisotropic_generator = torch.Generator(device="cpu").manual_seed(
            config.data_seed + 10_000 + index
        )
        identity_perturbed = add_isotropic_noise_in_chart(
            native,
            identity,
            standard_deviation=config.noise_standard_deviation,
            generator=identity_generator,
        )
        anisotropic_perturbed = add_isotropic_noise_in_chart(
            native,
            anisotropic,
            standard_deviation=config.noise_standard_deviation,
            generator=anisotropic_generator,
        )
        identity_effect = system.distance(native, identity_perturbed)
        anisotropic_effect = system.distance(native, anisotropic_perturbed)
        noise_functional_gaps.append(abs(identity_effect - anisotropic_effect))
    mean_noise_functional_gap = sum(noise_functional_gaps) / len(noise_functional_gaps)

    nonlinear_interpolation_gaps: list[float] = []
    for left, right in zip(pair_left, pair_right):
        native_midpoint = interpolate_in_chart(left, right, identity, alpha=0.5)
        nonlinear_midpoint = interpolate_in_chart(left, right, nonlinear, alpha=0.5)
        nonlinear_interpolation_gaps.append(
            system.distance(native_midpoint, nonlinear_midpoint)
        )
    mean_nonlinear_interpolation_gap = sum(nonlinear_interpolation_gaps) / len(
        nonlinear_interpolation_gaps
    )

    metrics = {
        "max_round_trip_error": max_round_trip_error,
        "max_behavior_identity_error": max_behavior_identity_error,
        "max_functional_invariance_error": max_functional_invariance_error,
        "orthogonal_neighbor_flip_rate": orthogonal_neighbor_flip_rate,
        "anisotropic_neighbor_flip_rate": anisotropic_neighbor_flip_rate,
        "anisotropic_ranking_disagreement": anisotropic_ranking_disagreement,
        "mean_noise_functional_gap": mean_noise_functional_gap,
        "mean_nonlinear_interpolation_gap": mean_nonlinear_interpolation_gap,
    }
    thresholds = config.thresholds
    checks = {
        "round_trip_exactness": metrics["max_round_trip_error"]
        <= thresholds["max_round_trip_error"],
        "behavior_identity": metrics["max_behavior_identity_error"]
        <= thresholds["max_behavior_identity_error"],
        "functional_distance_invariance": metrics["max_functional_invariance_error"]
        <= thresholds["max_functional_invariance_error"],
        "orthogonal_negative_control": metrics["orthogonal_neighbor_flip_rate"]
        <= thresholds["max_orthogonal_neighbor_flip_rate"],
        "anisotropic_neighbor_sensitivity": metrics["anisotropic_neighbor_flip_rate"]
        >= thresholds["min_anisotropic_neighbor_flip_rate"],
        "anisotropic_rank_sensitivity": metrics["anisotropic_ranking_disagreement"]
        >= thresholds["min_anisotropic_ranking_disagreement"],
        "coordinate_noise_sensitivity": metrics["mean_noise_functional_gap"]
        >= thresholds["min_noise_functional_gap"],
        "nonlinear_interpolation_sensitivity": metrics[
            "mean_nonlinear_interpolation_gap"
        ]
        >= thresholds["min_nonlinear_interpolation_gap"],
    }

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "experiment_name": config.experiment_name,
        "status": "pass" if all(checks.values()) else "fail",
        "scientific_evidence": False,
        "interpretation": (
            "This gate validates implementation contracts only. It is not evidence that "
            "coordinate dependence affects a trained latent reasoning model."
        ),
        "config": {
            "dimension": config.dimension,
            "hidden_dimension": config.hidden_dimension,
            "horizon": config.horizon,
            "vocabulary_size": config.vocabulary_size,
            "query_count": config.query_count,
            "candidate_count": config.candidate_count,
            "pair_count": config.pair_count,
            "model_seed": config.model_seed,
            "data_seed": config.data_seed,
            "chart_seed": config.chart_seed,
            "affine_condition_number": config.affine_condition_number,
            "noise_standard_deviation": config.noise_standard_deviation,
            "sinh_scale": config.sinh_scale,
        },
        "thresholds": dict(thresholds),
        "metrics": metrics,
        "checks": checks,
        "diagnostics": {
            chart_name: {
                "max_abs_round_trip_error": diagnostic.max_abs_round_trip_error,
                "relative_l2_round_trip_error": diagnostic.relative_l2_round_trip_error,
                "all_finite": diagnostic.all_finite,
            }
            for chart_name, diagnostic in diagnostics.items()
        },
        "behavior_identity_errors": behavior_identity_errors,
        "functional_invariance_errors": functional_invariance_errors,
    }


def load_config(path: str | Path) -> ToyGateConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("toy gate config must contain a JSON object")
    return ToyGateConfig.from_mapping(raw)


def main() -> int:
    default_config = Path(__file__).with_name("configs") / "toy_contract_v1.json"
    parser = argparse.ArgumentParser(description="Run coordinate-invariance toy contracts")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    result = run_toy_gate(load_config(arguments.config))
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

