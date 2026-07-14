from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .charts import AffineChart
from .fctr import (
    euclidean_rms_step,
    functional_trust_region_step,
    gradient_to_linear_chart,
    metric_to_linear_chart,
    relative_l2_error,
    vector_from_linear_chart,
)


def canonical_config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left, right, dim=0).detach().cpu())


def _observable_system(config: dict[str, Any]) -> dict[str, Any]:
    dimension = int(config["system"]["latent_dimension"])
    classes = int(config["system"]["output_classes"])
    horizon = int(config["system"]["horizon"])
    generator = torch.Generator(device="cpu").manual_seed(
        int(config["system"]["seed"])
    )
    latent = (
        float(config["system"]["latent_scale"])
        * torch.randn(dimension, generator=generator, dtype=torch.float64)
    ).requires_grad_(True)
    maps = [
        torch.randn(classes, dimension, generator=generator, dtype=torch.float64)
        / dimension**0.5
        for _ in range(horizon)
    ]
    biases = [
        0.1 * torch.randn(classes, generator=generator, dtype=torch.float64)
        for _ in range(horizon)
    ]
    rewards = [
        torch.randn(classes, generator=generator, dtype=torch.float64)
        for _ in range(horizon)
    ]

    utility = torch.zeros((), dtype=torch.float64)
    metric = torch.zeros((dimension, dimension), dtype=torch.float64)
    for mapping, bias, reward in zip(maps, biases, rewards):
        probabilities = torch.softmax(mapping @ latent + bias, dim=0)
        utility = utility + torch.dot(probabilities, reward)
        categorical_fisher = torch.diag(probabilities) - torch.outer(
            probabilities, probabilities
        )
        metric = metric + mapping.transpose(0, 1) @ categorical_fisher @ mapping
    gradient = torch.autograd.grad(utility, latent)[0].detach()
    metric = metric.detach()

    def evaluate(candidate: torch.Tensor) -> float:
        value = torch.zeros((), dtype=torch.float64)
        for mapping, bias, reward in zip(maps, biases, rewards):
            value = value + torch.dot(
                torch.softmax(mapping @ candidate + bias, dim=0), reward
            )
        return float(value.detach().cpu())

    return {
        "latent": latent.detach(),
        "gradient": gradient,
        "metric": metric,
        "utility": float(utility.detach().cpu()),
        "evaluate": evaluate,
    }


def _charts(config: dict[str, Any], dimension: int) -> list[AffineChart]:
    chart_config = config["charts"]
    charts = [AffineChart.identity(dimension, dtype=torch.float64)]
    charts.append(
        AffineChart.random_orthogonal(
            dimension,
            seed=int(chart_config["orthogonal_seed"]),
            dtype=torch.float64,
        )
    )
    for index, condition in enumerate(chart_config["condition_numbers"]):
        charts.append(
            AffineChart.with_condition_number(
                dimension,
                float(condition),
                seed=int(chart_config["anisotropic_seed"]) + index,
                dtype=torch.float64,
            )
        )
    return charts


def run(config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    system = _observable_system(config)
    latent = system["latent"]
    gradient = system["gradient"]
    metric = system["metric"]
    evaluate = system["evaluate"]
    trust_budget = float(config["updates"]["trust_budget"])
    rms_step = float(config["updates"]["euclidean_rms_step"])

    native_fctr = functional_trust_region_step(
        gradient, metric, trust_budget=trust_budget
    )
    native_euclidean = euclidean_rms_step(gradient, rms_step=rms_step)
    native_utility = float(system["utility"])
    chart_records: list[dict[str, Any]] = []
    for chart in _charts(config, gradient.numel()):
        matrix = chart.matrix.to(torch.float64)
        gradient_charted = gradient_to_linear_chart(gradient, matrix)
        metric_charted = metric_to_linear_chart(metric, matrix)
        charted_fctr = functional_trust_region_step(
            gradient_charted,
            metric_charted,
            trust_budget=trust_budget,
        )
        fctr_native = vector_from_linear_chart(charted_fctr.step, matrix)
        charted_euclidean = euclidean_rms_step(
            gradient_charted, rms_step=rms_step
        )
        euclidean_native = vector_from_linear_chart(charted_euclidean, matrix)

        chart_records.append(
            {
                "name": chart.name,
                "condition_number": chart.condition_number,
                "fctr_transport_relative_error": relative_l2_error(
                    native_fctr.step, fctr_native
                ),
                "fctr_transport_cosine": _cosine(native_fctr.step, fctr_native),
                "fctr_trust_cost_abs_error": abs(
                    charted_fctr.trust_cost - trust_budget
                ),
                "fctr_predicted_gain_abs_error": abs(
                    charted_fctr.predicted_gain - native_fctr.predicted_gain
                ),
                "fctr_actual_gain_abs_error": abs(
                    (evaluate(latent + fctr_native) - native_utility)
                    - (evaluate(latent + native_fctr.step) - native_utility)
                ),
                "euclidean_transport_relative_error": relative_l2_error(
                    native_euclidean, euclidean_native
                ),
                "euclidean_transport_cosine": _cosine(
                    native_euclidean, euclidean_native
                ),
                "native_euclidean_actual_gain": evaluate(
                    latent + native_euclidean
                )
                - native_utility,
                "charted_euclidean_actual_gain": evaluate(
                    latent + euclidean_native
                )
                - native_utility,
            }
        )

    thresholds = config["thresholds"]
    identity = next(row for row in chart_records if row["name"] == "identity")
    orthogonal = next(
        row for row in chart_records if row["name"].startswith("orthogonal-")
    )
    strongest = max(chart_records, key=lambda row: row["condition_number"])
    gates = {
        "metric_positive_definite": float(torch.linalg.eigvalsh(metric).min())
        > 0.0,
        "all_fctr_transport": max(
            row["fctr_transport_relative_error"] for row in chart_records
        )
        <= float(thresholds["max_fctr_transport_relative_error"]),
        "all_fctr_budget": max(
            row["fctr_trust_cost_abs_error"] for row in chart_records
        )
        <= float(thresholds["max_fctr_trust_cost_abs_error"]),
        "all_fctr_gain": max(
            row["fctr_predicted_gain_abs_error"] for row in chart_records
        )
        <= float(thresholds["max_fctr_predicted_gain_abs_error"]),
        "identity_euclidean_control": identity[
            "euclidean_transport_relative_error"
        ]
        <= float(thresholds["max_control_euclidean_relative_error"]),
        "orthogonal_euclidean_control": orthogonal[
            "euclidean_transport_relative_error"
        ]
        <= float(thresholds["max_control_euclidean_relative_error"]),
        "anisotropic_euclidean_positive_control": strongest[
            "euclidean_transport_relative_error"
        ]
        >= float(thresholds["min_anisotropic_euclidean_relative_error"]),
    }
    return {
        "experiment_name": config["experiment_name"],
        "status": "pass" if all(gates.values()) else "fail",
        "evidence_level": "deterministic implementation contract only",
        "config_sha256": canonical_config_hash(config),
        "system": {
            "latent_dimension": int(gradient.numel()),
            "metric_min_eigenvalue": float(torch.linalg.eigvalsh(metric).min()),
            "metric_max_eigenvalue": float(torch.linalg.eigvalsh(metric).max()),
            "gradient_norm": float(torch.linalg.vector_norm(gradient)),
            "native_utility": native_utility,
        },
        "native_fctr": {
            "trust_cost": native_fctr.trust_cost,
            "predicted_gain": native_fctr.predicted_gain,
            "metric_condition_number": native_fctr.metric_condition_number,
        },
        "charts": chart_records,
        "gates": gates,
        "runtime": {
            "seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "interpretation": (
            "A pass validates coordinate transport and the local FCTR solver. "
            "It is not evidence that a trained latent-reasoning method needs FCTR."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = run(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        sys.exit(2)


if __name__ == "__main__":
    main()
