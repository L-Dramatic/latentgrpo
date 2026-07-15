from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from .charts import AffineChart
from .fctr import (
    euclidean_rms_step,
    functional_trust_region_step,
    gradient_to_linear_chart,
    metric_to_linear_chart,
    relative_l2_error,
    vector_from_linear_chart,
)
from .real_models.coconut import (
    CoconutChartRunner,
    CoconutContinuationAdapter,
    load_public_coconut,
    prepare_coconut_input,
)
from .rng import make_generator


LogitFunction = Callable[[Tensor], Tensor]


def canonical_config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _relative_tensor_error(reference: Tensor, candidate: Tensor) -> float:
    denominator = torch.linalg.vector_norm(reference)
    if float(denominator) <= 0.0:
        raise ValueError("reference tensor must be nonzero")
    return float(
        (torch.linalg.vector_norm(candidate - reference) / denominator)
        .detach()
        .cpu()
    )


def _measure_jacobian(
    function: LogitFunction, dimension: int, *, device: torch.device
) -> tuple[Tensor, Tensor]:
    origin = torch.zeros(dimension, dtype=torch.float32, device=device)
    base = function(origin).detach()
    columns: list[Tensor] = []
    for index in range(dimension):
        tangent = torch.zeros_like(origin)
        tangent[index] = 1.0
        value, derivative = torch.autograd.functional.jvp(
            function,
            origin,
            tangent,
            create_graph=False,
            strict=True,
        )
        if not torch.equal(value.detach(), base):
            raise AssertionError("JVP evaluations changed the zero-point logits")
        columns.append(derivative.detach())
    return base, torch.stack(columns, dim=-1)


def _prefix_geometry(
    logits: Tensor, jacobian: Tensor, target_ids: Tensor
) -> tuple[Tensor, list[Tensor]]:
    logits = logits.to(torch.float64)
    jacobian = jacobian.to(torch.float64)
    target_ids = target_ids.to(device=logits.device, dtype=torch.long)
    horizon, _, dimension = jacobian.shape
    if target_ids.numel() != horizon:
        raise ValueError("target ids must match the measured horizon")

    gradient = torch.zeros(dimension, dtype=torch.float64, device=logits.device)
    cumulative = torch.zeros(
        (dimension, dimension), dtype=torch.float64, device=logits.device
    )
    metrics: list[Tensor] = []
    for step in range(horizon):
        probabilities = torch.softmax(logits[step], dim=0)
        current_jacobian = jacobian[step]
        mean_jacobian = probabilities @ current_jacobian
        score = current_jacobian[target_ids[step]] - mean_jacobian
        gradient = gradient + score / horizon
        weighted = probabilities.unsqueeze(-1) * current_jacobian
        step_metric = (
            current_jacobian.transpose(0, 1) @ weighted
            - torch.outer(mean_jacobian, mean_jacobian)
        )
        cumulative = cumulative + 0.5 * (
            step_metric + step_metric.transpose(0, 1)
        )
        metrics.append(cumulative.clone())
    return gradient, metrics


def _utility(logits: Tensor, target_ids: Tensor) -> float:
    log_probabilities = torch.log_softmax(logits.to(torch.float64), dim=-1)
    indices = torch.arange(target_ids.numel(), device=logits.device)
    return float(
        log_probabilities[
            indices, target_ids.to(device=logits.device, dtype=torch.long)
        ]
        .mean()
        .detach()
        .cpu()
    )


def _orthonormal_basis(
    ambient_dimension: int,
    subspace_dimension: int,
    *,
    seed: int,
    device: torch.device,
) -> Tensor:
    if not 1 <= subspace_dimension <= ambient_dimension:
        raise ValueError("subspace dimension must lie within the ambient dimension")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn(
        ambient_dimension,
        subspace_dimension,
        generator=generator,
        dtype=torch.float64,
    )
    basis, _ = torch.linalg.qr(raw, mode="reduced")
    return basis.to(device=device, dtype=torch.float32)


def _charts(config: dict[str, Any], dimension: int) -> list[AffineChart]:
    chart_config = config["charts"]
    result = [AffineChart.identity(dimension, dtype=torch.float64)]
    result.append(
        AffineChart.random_orthogonal(
            dimension,
            seed=int(chart_config["orthogonal_seed"]),
            dtype=torch.float64,
        )
    )
    for index, condition in enumerate(chart_config["condition_numbers"]):
        result.append(
            AffineChart.with_condition_number(
                dimension,
                float(condition),
                seed=int(chart_config["anisotropic_seed"]) + index,
                dtype=torch.float64,
            )
        )
    return result


def run(config: dict[str, Any], workspace_root: Path) -> dict[str, Any]:
    import transformers

    started = time.perf_counter()
    device = torch.device(str(config["runtime"]["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("the frozen integration smoke requires CUDA")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    dataset_path = workspace_root / str(config["dataset_relative_path"])
    dataset_bytes = dataset_path.read_bytes()
    dataset_sha256 = hashlib.sha256(dataset_bytes).hexdigest()
    if dataset_sha256 != str(config["dataset_sha256"]):
        raise ValueError(
            f"dataset SHA-256 is {dataset_sha256}, expected {config['dataset_sha256']}"
        )
    dataset = json.loads(dataset_bytes.decode("utf-8"))
    dataset_index = int(config["dataset_index"])
    example = dataset[dataset_index]
    bundle = load_public_coconut(
        checkpoint_path=workspace_root / str(config["checkpoint_relative_path"]),
        coconut_source_directory=workspace_root
        / str(config["coconut_source_relative_path"]),
        model_id=str(config["model_id"]),
        model_revision=str(config["model_revision"]),
        expected_checkpoint_sha256=str(config["checkpoint_sha256"]),
        attention_implementation=config.get("attention_implementation"),
        device=device,
        cache_directory=workspace_root / str(config["model_cache_relative_path"]),
    )
    if bundle.coconut_source_commit != str(config["coconut_source_commit"]):
        raise ValueError("loaded Coconut source does not match the frozen commit")
    for parameter in bundle.model.parameters():
        parameter.requires_grad_(False)
    bundle.model.eval()

    ambient_dimension = int(bundle.model.embedding.weight.shape[-1])
    identity = AffineChart.identity(
        ambient_dimension, dtype=torch.float64, compute_dtype=torch.float32
    )
    runner = CoconutChartRunner(bundle.model, identity)
    input_ids, attention_mask = prepare_coconut_input(
        bundle,
        str(example["query"]),
        num_latents=int(config["num_latents"]),
    )
    sample_id = f"gsm8k-c1-{dataset_index}"
    trace_run = runner.run(
        input_ids,
        attention_mask,
        seed=int(config["collection_seed"]),
        capture_trace=True,
        sample_id=sample_id,
    )
    target_pass = int(config["target_pass"])
    record = trace_run.trace.get(sample_id, target_pass)
    factual_latent = record.latent.to(device=device, dtype=torch.float32)

    target_text = str(config["target_template"]).format(answer=example["answer"])
    target_list = bundle.tokenizer.encode(target_text, add_special_tokens=False)
    target_list.append(int(bundle.tokenizer.eos_token_id))
    target_list = target_list[: int(config["target_tokens"])]
    if len(target_list) != int(config["target_tokens"]):
        raise ValueError("target text produced too few tokens")
    target_ids = torch.tensor(target_list, dtype=torch.long, device=device)
    eos_id = int(bundle.tokenizer.eos_token_id)

    adapter = CoconutContinuationAdapter(
        runner,
        temperature=1.0,
        stop_at_eos=True,
        track_gradients=True,
    )
    subspace_dimension = int(config["subspace"]["dimension"])
    basis = _orthonormal_basis(
        ambient_dimension,
        subspace_dimension,
        seed=int(config["subspace"]["seed"]),
        device=device,
    )
    rollout_seed = int(config["rollout_seed"])
    compute = {"function_evaluations": 0, "model_forward_calls": 0}

    def native_logits(coefficients: Tensor) -> Tensor:
        candidate = factual_latent + basis @ coefficients
        output = adapter.force(
            record.prefix_state,
            candidate,
            token_ids=target_ids,
            generator=make_generator(rollout_seed, device),
        )
        if output.logits.shape[0] != target_ids.numel():
            raise ValueError("forced continuation terminated before the target horizon")
        compute["function_evaluations"] += 1
        compute["model_forward_calls"] += output.model_forward_calls
        return output.logits

    base_logits, native_jacobian = _measure_jacobian(
        native_logits, subspace_dimension, device=device
    )
    native_gradient, native_metrics = _prefix_geometry(
        base_logits, native_jacobian, target_ids
    )
    metric_eigenvalues = [torch.linalg.eigvalsh(metric) for metric in native_metrics]
    if any(float(values.min()) <= 0.0 for values in metric_eigenvalues):
        raise ValueError("a measured prefix metric is not positive definite")

    finite_difference_config = config["finite_difference"]
    fd_generator = torch.Generator(device="cpu").manual_seed(
        int(finite_difference_config["seed"])
    )
    fd_direction = torch.randn(
        subspace_dimension, generator=fd_generator, dtype=torch.float64
    )
    fd_direction = fd_direction / torch.linalg.vector_norm(fd_direction)
    epsilon = float(finite_difference_config["epsilon"])
    plus = native_logits((epsilon * fd_direction).to(device=device, dtype=torch.float32))
    minus = native_logits((-epsilon * fd_direction).to(device=device, dtype=torch.float32))
    finite_difference = (_utility(plus, target_ids) - _utility(minus, target_ids)) / (
        2.0 * epsilon
    )
    analytic_directional = float(torch.dot(native_gradient.cpu(), fd_direction))
    fd_denominator = max(abs(analytic_directional), 1e-12)
    finite_difference_relative_error = abs(
        finite_difference - analytic_directional
    ) / fd_denominator

    trust_budget = float(config["updates"]["trust_budget"])
    euclidean_rms = float(config["updates"]["euclidean_rms_step"])
    native_fctr = functional_trust_region_step(
        native_gradient.cpu(), native_metrics[-1].cpu(), trust_budget=trust_budget
    )
    native_euclidean = euclidean_rms_step(
        native_gradient.cpu(), rms_step=euclidean_rms
    )
    base_utility = _utility(base_logits, target_ids)
    native_fctr_logits = native_logits(
        native_fctr.step.to(device=device, dtype=torch.float32)
    )
    native_fctr_actual_gain = _utility(native_fctr_logits, target_ids) - base_utility

    chart_records: list[dict[str, Any]] = []
    for chart in _charts(config, subspace_dimension):
        matrix = chart.matrix.to(torch.float64)
        inverse_float = torch.linalg.inv(matrix).to(device=device, dtype=torch.float32)

        def charted_logits(coordinates: Tensor) -> Tensor:
            return native_logits(inverse_float @ coordinates)

        chart_base, chart_jacobian = _measure_jacobian(
            charted_logits, subspace_dimension, device=device
        )
        chart_gradient, chart_metrics = _prefix_geometry(
            chart_base, chart_jacobian, target_ids
        )
        expected_jacobian = native_jacobian.to(
            device="cpu", dtype=torch.float64
        ) @ torch.linalg.inv(matrix)
        expected_gradient = gradient_to_linear_chart(
            native_gradient.cpu(), matrix
        )
        expected_metrics = [
            metric_to_linear_chart(metric.cpu(), matrix)
            for metric in native_metrics
        ]
        chart_fctr = functional_trust_region_step(
            chart_gradient.cpu(), chart_metrics[-1].cpu(), trust_budget=trust_budget
        )
        transported_fctr = vector_from_linear_chart(chart_fctr.step, matrix)
        chart_euclidean = euclidean_rms_step(
            chart_gradient.cpu(), rms_step=euclidean_rms
        )
        transported_euclidean = vector_from_linear_chart(
            chart_euclidean, matrix
        )
        chart_fctr_logits = charted_logits(
            chart_fctr.step.to(device=device, dtype=torch.float32)
        )
        chart_records.append(
            {
                "name": chart.name,
                "condition_number": chart.condition_number,
                "noop_logit_max_abs_error": float(
                    (chart_base - base_logits).abs().max().detach().cpu()
                ),
                "jacobian_transport_relative_error": _relative_tensor_error(
                    expected_jacobian.cpu(), chart_jacobian.to(torch.float64).cpu()
                ),
                "gradient_transport_relative_error": relative_l2_error(
                    expected_gradient, chart_gradient.cpu()
                ),
                "metric_transport_relative_error_max": max(
                    _relative_tensor_error(expected.cpu(), measured.cpu())
                    for expected, measured in zip(expected_metrics, chart_metrics)
                ),
                "fctr_transport_relative_error": relative_l2_error(
                    native_fctr.step, transported_fctr
                ),
                "fctr_actual_gain_abs_error": abs(
                    (_utility(chart_fctr_logits, target_ids) - base_utility)
                    - native_fctr_actual_gain
                ),
                "euclidean_transport_relative_error": relative_l2_error(
                    native_euclidean, transported_euclidean
                ),
                "euclidean_transport_cosine": float(
                    F.cosine_similarity(
                        native_euclidean, transported_euclidean, dim=0
                    )
                ),
            }
        )

    thresholds = config["thresholds"]
    identity_record = next(row for row in chart_records if row["name"] == "identity")
    orthogonal_record = next(
        row for row in chart_records if row["name"].startswith("orthogonal-")
    )
    strongest_record = max(
        chart_records, key=lambda row: row["condition_number"]
    )
    gates = {
        "finite_logits_and_jacobians": bool(
            torch.isfinite(base_logits).all()
            and torch.isfinite(native_jacobian).all()
        ),
        "valid_pre_eos_target": eos_id not in target_list[:-1]
        and target_list[-1] == eos_id,
        "finite_difference": finite_difference_relative_error
        <= float(thresholds["max_finite_difference_relative_error"]),
        "no_op_logits": max(
            row["noop_logit_max_abs_error"] for row in chart_records
        )
        <= float(thresholds["max_noop_logit_error"]),
        "jacobian_transport": max(
            row["jacobian_transport_relative_error"] for row in chart_records
        )
        <= float(thresholds["max_jacobian_transport_relative_error"]),
        "gradient_transport": max(
            row["gradient_transport_relative_error"] for row in chart_records
        )
        <= float(thresholds["max_gradient_transport_relative_error"]),
        "metric_transport": max(
            row["metric_transport_relative_error_max"] for row in chart_records
        )
        <= float(thresholds["max_metric_transport_relative_error"]),
        "fctr_transport": max(
            row["fctr_transport_relative_error"] for row in chart_records
        )
        <= float(thresholds["max_fctr_transport_relative_error"]),
        "identity_euclidean_control": identity_record[
            "euclidean_transport_relative_error"
        ]
        <= float(thresholds["max_control_euclidean_relative_error"]),
        "orthogonal_euclidean_control": orthogonal_record[
            "euclidean_transport_relative_error"
        ]
        <= float(thresholds["max_control_euclidean_relative_error"]),
        "anisotropic_euclidean_positive_control": strongest_record[
            "euclidean_transport_relative_error"
        ]
        >= float(thresholds["min_anisotropic_euclidean_relative_error"]),
    }

    if device.type == "cuda":
        torch.cuda.synchronize()
    return {
        "experiment_name": config["experiment_name"],
        "status": "pass" if all(gates.values()) else "fail",
        "evidence_level": "real-checkpoint differentiable integration contract only",
        "config_sha256": canonical_config_hash(config),
        "checkpoint": {
            "model_id": bundle.model_id,
            "model_revision": bundle.model_revision,
            "attention_implementation": bundle.attention_implementation,
            "checkpoint_path": str(bundle.checkpoint_path),
            "checkpoint_sha256": bundle.checkpoint_sha256,
            "coconut_source_commit": bundle.coconut_source_commit,
        },
        "example": {
            "dataset_sha256": dataset_sha256,
            "dataset_index": dataset_index,
            "question_sha256": hashlib.sha256(
                str(example["query"]).encode("utf-8")
            ).hexdigest(),
            "target_text": target_text,
            "target_ids": target_list,
            "target_tokens": bundle.tokenizer.convert_ids_to_tokens(target_list),
            "target_pass": target_pass,
        },
        "subspace": {
            "ambient_dimension": ambient_dimension,
            "dimension": subspace_dimension,
            "basis_orthogonality_max_abs_error": float(
                (
                    basis.to(torch.float64).T @ basis.to(torch.float64)
                    - torch.eye(subspace_dimension, device=device, dtype=torch.float64)
                )
                .abs()
                .max()
                .detach()
                .cpu()
            ),
        },
        "geometry": {
            "base_utility": base_utility,
            "gradient_norm": float(torch.linalg.vector_norm(native_gradient)),
            "prefix_metric_min_eigenvalues": [
                float(values.min()) for values in metric_eigenvalues
            ],
            "prefix_metric_max_eigenvalues": [
                float(values.max()) for values in metric_eigenvalues
            ],
            "finite_difference": finite_difference,
            "analytic_directional_derivative": analytic_directional,
            "finite_difference_relative_error": finite_difference_relative_error,
        },
        "charts": chart_records,
        "gates": gates,
        "runtime": {
            "seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": str(device),
            "cuda_device": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "cuda_peak_allocated_mib": (
                float(torch.cuda.max_memory_allocated(device) / 1024**2)
                if device.type == "cuda"
                else None
            ),
            "function_evaluations": compute["function_evaluations"],
            "model_forward_calls": compute["model_forward_calls"],
        },
        "interpretation": (
            "A pass validates differentiable replay and H1/H2/H3 transport in a "
            "frozen three-dimensional update subspace. It cannot authorize FCTR "
            "because this checkpoint has a trivial visible continuation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = run(config, args.workspace_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        sys.exit(2)


if __name__ == "__main__":
    main()
