from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor

from .charts import AffineChart
from .fctr import (
    gradient_to_linear_chart,
    metric_to_linear_chart,
    relative_l2_error,
    vector_from_linear_chart,
)
from .metrics import stable_categorical_kl_from_logits
from .real_models.switch import (
    SWITCH_ADAPTER_ID,
    SWITCH_ADAPTER_REVISION,
    SWITCH_BASE_MODEL_ID,
    SWITCH_BASE_REVISION,
    SwitchAuditRunner,
    SwitchDifferentiableReplay,
    SwitchReplayPlan,
    load_public_switch,
)
from .statistics import bootstrap_mean
from .switch_c2_geometry import (
    chart_euclidean_update,
    consequential_basis,
    fit_diagonal_whitening_precision,
    mean_factual_log_probability,
    metric_update,
    prefix_geometry,
    projected_diagonal_metric,
    regularize_with_whitening,
    semantic_prefix_horizon,
    spearman_correlation,
    summed_categorical_kl,
    top_risk_recall,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def implementation_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    return {
        "scientific_runner": _sha256(Path(__file__).resolve()),
        "geometry": _sha256(root / "switch_c2_geometry.py"),
        "switch_adapter": _sha256(root / "real_models" / "switch.py"),
        "fctr_solver": _sha256(root / "fctr.py"),
    }


def _load_plan(payload: dict[str, Any]) -> SwitchReplayPlan:
    return SwitchReplayPlan(
        prompt_input_ids=torch.tensor(payload["prompt_input_ids"], dtype=torch.long),
        prompt_attention_mask=torch.tensor(
            payload["prompt_attention_mask"], dtype=torch.long
        ),
        visible_prefix_ids=torch.tensor(payload["visible_prefix_ids"], dtype=torch.long),
        latent_steps=int(payload["latent_steps"]),
        visible_target_ids=torch.tensor(
            payload["visible_target_ids"], dtype=torch.long
        ),
        visible_decision_start_index=int(payload["visible_decision_start_index"]),
    )


def _chart_matrices(config: dict[str, Any], dimension: int) -> dict[str, Tensor]:
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
    return {chart.name: chart.matrix.detach().cpu().to(torch.float64) for chart in charts}


def _objective_tensor(logits: Tensor, target_ids: Tensor, horizon: int) -> Tensor:
    targets = target_ids[:horizon].to(device=logits.device, dtype=torch.long)
    rows = torch.arange(horizon, device=logits.device)
    return torch.log_softmax(logits[:horizon].to(torch.float32), dim=-1)[
        rows, targets
    ].mean()


def _measure_jacobian(
    replay: SwitchDifferentiableReplay,
    basis: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
    dimension = int(basis.shape[1])
    device = basis.device
    origin = torch.zeros(dimension, dtype=torch.float32, device=device)
    base_latent = None
    base_visible = None
    latent_columns: list[Tensor] = []
    visible_columns: list[Tensor] = []
    for index in range(dimension):
        tangent = torch.zeros_like(origin)
        tangent[index] = 1.0
        values, derivatives = torch.autograd.functional.jvp(
            lambda coefficients: replay.logits(coefficients, basis),
            origin,
            tangent,
            create_graph=False,
            strict=True,
        )
        latent_value, visible_value = values
        latent_derivative, visible_derivative = derivatives
        if base_latent is None:
            base_latent = latent_value.detach()
            base_visible = visible_value.detach()
        elif not torch.equal(base_latent, latent_value.detach()) or not torch.equal(
            base_visible, visible_value.detach()
        ):
            raise AssertionError("JVP zero point changed across basis directions")
        latent_columns.append(latent_derivative.detach())
        visible_columns.append(visible_derivative.detach())
    if base_latent is None or base_visible is None:
        raise AssertionError("the frozen subspace must have positive dimension")
    return (
        base_latent,
        base_visible,
        torch.stack(latent_columns, dim=-1),
        torch.stack(visible_columns, dim=-1),
        dimension,
    )


def _relative_error(reference: Tensor, candidate: Tensor) -> float:
    denominator = torch.linalg.vector_norm(reference)
    if float(denominator) <= 0.0:
        raise ValueError("relative-error reference is zero")
    return float(torch.linalg.vector_norm(candidate - reference) / denominator)


def _regularized_metrics(
    raw_metrics: dict[str, Tensor],
    whitening: Tensor,
    *,
    relative_ridge: float,
) -> tuple[dict[str, Tensor], dict[str, float]]:
    regularized = {"activation_whitening": whitening}
    ridges = {"activation_whitening": 0.0}
    for name, metric in raw_metrics.items():
        if name == "activation_whitening":
            continue
        value, ridge = regularize_with_whitening(
            metric,
            whitening,
            relative_generalized_ridge=relative_ridge,
        )
        regularized[name] = value
        ridges[name] = ridge
    return regularized, ridges


def _metric_dictionary(
    config: dict[str, Any],
    tokenizer: Any,
    target_ids: Tensor,
    latent_geometry: Any,
    visible_geometry: Any,
    whitening: Tensor,
) -> tuple[dict[str, Tensor], int]:
    metrics: dict[str, Tensor] = {"activation_whitening": whitening}
    for horizon in config["metrics"]["latent_exit_prefix_horizons"]:
        metrics[f"latent_exit_prefix_{int(horizon)}"] = (
            latent_geometry.cumulative_metrics[int(horizon) - 1].detach().cpu()
        )
    for horizon in config["metrics"]["visible_prefix_horizons"]:
        metrics[f"visible_prefix_{int(horizon)}"] = (
            visible_geometry.cumulative_metrics[int(horizon) - 1].detach().cpu()
        )
    semantic_config = config["metrics"]["semantic_prefix"]
    semantic_horizon = semantic_prefix_horizon(
        tokenizer,
        target_ids,
        minimum_tokens=int(semantic_config["minimum_tokens"]),
        maximum_tokens=int(semantic_config["maximum_tokens"]),
        boundary_regex=str(semantic_config["boundary_regex"]),
    )
    metrics["semantic_prefix"] = (
        visible_geometry.cumulative_metrics[semantic_horizon - 1].detach().cpu()
    )
    return metrics, semantic_horizon


def _evaluate_logits(
    replay: SwitchDifferentiableReplay,
    coefficients: Tensor,
    basis: Tensor,
) -> tuple[Tensor, Tensor]:
    with torch.no_grad():
        return replay.logits(
            coefficients.to(device=basis.device, dtype=torch.float32), basis
        )


def _outcome(
    base_visible: Tensor,
    candidate_visible: Tensor,
    target_ids: Tensor,
    *,
    objective_horizon: int,
    secondary_start: int,
    strict_start: int,
) -> dict[str, float]:
    horizon = int(base_visible.shape[0])
    base_objective = mean_factual_log_probability(
        base_visible, target_ids, start=0, end=objective_horizon
    )
    candidate_objective = mean_factual_log_probability(
        candidate_visible, target_ids, start=0, end=objective_horizon
    )
    base_secondary = mean_factual_log_probability(
        base_visible, target_ids, start=secondary_start, end=horizon
    )
    candidate_secondary = mean_factual_log_probability(
        candidate_visible, target_ids, start=secondary_start, end=horizon
    )
    base_strict = mean_factual_log_probability(
        base_visible, target_ids, start=strict_start, end=horizon
    )
    candidate_strict = mean_factual_log_probability(
        candidate_visible, target_ids, start=strict_start, end=horizon
    )
    return {
        "objective_utility_gain": candidate_objective - base_objective,
        "secondary_utility_gain": candidate_secondary - base_secondary,
        "strict_utility_gain": candidate_strict - base_strict,
        "prefix8_kl": summed_categorical_kl(
            base_visible, candidate_visible, start=0, end=objective_horizon
        ),
        "secondary_kl": summed_categorical_kl(
            base_visible, candidate_visible, start=secondary_start, end=horizon
        ),
        "strict_kl": summed_categorical_kl(
            base_visible, candidate_visible, start=strict_start, end=horizon
        ),
    }


def _probe_directions(dimension: int, *, pairs: int, seed: int) -> Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    directions = torch.randn(pairs, dimension, generator=generator, dtype=torch.float64)
    directions = directions / torch.linalg.vector_norm(
        directions, dim=1, keepdim=True
    )
    return torch.cat([directions, -directions], dim=0)


def _measure_probes(
    config: dict[str, Any],
    replay: SwitchDifferentiableReplay,
    basis: Tensor,
    factual_norm: float,
    base_visible: Tensor,
    target_ids: Tensor,
    raw_metrics: dict[str, Tensor],
    *,
    scan_rank: int,
    scales: Iterable[float],
) -> list[dict[str, Any]]:
    probe_config = config["probe_bank"]
    directions = _probe_directions(
        basis.shape[1],
        pairs=int(probe_config["paired_native_directions"]),
        seed=int(probe_config["seed"]) + scan_rank,
    )
    strict_start = int(config["intervention"]["strict_holdout_visible_tokens"][0]) - 1
    secondary_start = (
        int(config["intervention"]["secondary_holdout_visible_tokens"][0]) - 1
    )
    fraction = float(probe_config["top_risk_fraction"])
    results = []
    for relative_scale in scales:
        actual_strict: list[float] = []
        actual_secondary: list[float] = []
        predicted = {name: [] for name in raw_metrics}
        coefficient_vectors = []
        for direction in directions:
            coefficients = float(relative_scale) * factual_norm * direction
            _, candidate_visible = _evaluate_logits(replay, coefficients, basis)
            actual_strict.append(
                summed_categorical_kl(
                    base_visible,
                    candidate_visible,
                    start=strict_start,
                    end=base_visible.shape[0],
                )
            )
            actual_secondary.append(
                summed_categorical_kl(
                    base_visible,
                    candidate_visible,
                    start=secondary_start,
                    end=base_visible.shape[0],
                )
            )
            for name, metric in raw_metrics.items():
                predicted[name].append(
                    float(0.5 * torch.dot(coefficients, metric @ coefficients))
                )
            coefficient_vectors.append(coefficients.tolist())
        metric_records = {}
        for name, values in predicted.items():
            metric_records[name] = {
                "predicted_risk": values,
                "strict_spearman": spearman_correlation(values, actual_strict),
                "strict_top_risk_recall": top_risk_recall(
                    values, actual_strict, fraction=fraction
                ),
            }
        results.append(
            {
                "relative_hidden_l2": float(relative_scale),
                "coefficient_vectors": coefficient_vectors,
                "actual_strict_kl": actual_strict,
                "actual_secondary_kl": actual_secondary,
                "metrics": metric_records,
            }
        )
    return results


def _metric_transport_controls(
    gradient: Tensor,
    metrics: dict[str, Tensor],
    charts: dict[str, Tensor],
    *,
    predicted_gain: float,
) -> dict[str, float]:
    max_update_error = 0.0
    for metric in metrics.values():
        native = metric_update(metric, gradient, predicted_gain=predicted_gain)
        for matrix in charts.values():
            chart_gradient = gradient_to_linear_chart(gradient, matrix)
            chart_metric = metric_to_linear_chart(metric, matrix)
            chart_step = metric_update(
                chart_metric, chart_gradient, predicted_gain=predicted_gain
            )
            transported = vector_from_linear_chart(chart_step, matrix)
            max_update_error = max(
                max_update_error, relative_l2_error(native, transported)
            )
    identity_name = next(name for name in charts if name == "identity")
    orthogonal_name = next(name for name in charts if name.startswith("orthogonal-"))
    identity_step = chart_euclidean_update(
        gradient, charts[identity_name], predicted_gain=predicted_gain
    )
    orthogonal_step = chart_euclidean_update(
        gradient, charts[orthogonal_name], predicted_gain=predicted_gain
    )
    strongest_name = max(
        charts,
        key=lambda name: float(torch.linalg.cond(charts[name])),
    )
    strongest_step = chart_euclidean_update(
        gradient, charts[strongest_name], predicted_gain=predicted_gain
    )
    return {
        "maximum_metric_update_transport_relative_error": max_update_error,
        "orthogonal_euclidean_direction_discrepancy": relative_l2_error(
            identity_step, orthogonal_step
        ),
        "condition12_euclidean_direction_discrepancy": relative_l2_error(
            identity_step, strongest_step
        ),
    }


def _measure_updates(
    config: dict[str, Any],
    replay: SwitchDifferentiableReplay,
    basis: Tensor,
    factual_norm: float,
    base_visible: Tensor,
    target_ids: Tensor,
    gradient: Tensor,
    metrics: dict[str, Tensor],
    charts: dict[str, Tensor],
    *,
    gains: Iterable[float],
) -> list[dict[str, Any]]:
    intervention = config["intervention"]
    objective_horizon = int(intervention["objective_visible_tokens"][1])
    secondary_start = int(intervention["secondary_holdout_visible_tokens"][0]) - 1
    strict_start = int(intervention["strict_holdout_visible_tokens"][0]) - 1
    maximum_relative = float(config["updates"]["maximum_relative_hidden_l2"])
    results = []
    for gain in gains:
        methods = {}
        for name, metric in metrics.items():
            coefficients = metric_update(metric, gradient, predicted_gain=float(gain))
            relative = float(torch.linalg.vector_norm(coefficients) / factual_norm)
            record: dict[str, Any] = {
                "coefficients": coefficients.tolist(),
                "relative_hidden_l2": relative,
                "predicted_objective_gain": float(torch.dot(gradient, coefficients)),
                "valid_radius": relative <= maximum_relative,
            }
            if record["valid_radius"]:
                _, candidate_visible = _evaluate_logits(replay, coefficients, basis)
                record.update(
                    _outcome(
                        base_visible,
                        candidate_visible,
                        target_ids,
                        objective_horizon=objective_horizon,
                        secondary_start=secondary_start,
                        strict_start=strict_start,
                    )
                )
            methods[name] = record
        coordinate = {}
        for chart_name, matrix in charts.items():
            coefficients = chart_euclidean_update(
                gradient, matrix, predicted_gain=float(gain)
            )
            relative = float(torch.linalg.vector_norm(coefficients) / factual_norm)
            record = {
                "coefficients": coefficients.tolist(),
                "relative_hidden_l2": relative,
                "predicted_objective_gain": float(torch.dot(gradient, coefficients)),
                "valid_radius": relative <= maximum_relative,
            }
            if record["valid_radius"]:
                _, candidate_visible = _evaluate_logits(replay, coefficients, basis)
                record.update(
                    _outcome(
                        base_visible,
                        candidate_visible,
                        target_ids,
                        objective_horizon=objective_horizon,
                        secondary_start=secondary_start,
                        strict_start=strict_start,
                    )
                )
            coordinate[chart_name] = record
        results.append(
            {
                "predicted_objective_gain": float(gain),
                "methods": methods,
                "coordinate_euclidean": coordinate,
            }
        )
    return results


def _retune_chart_to_exact_v8_kl(
    config: dict[str, Any],
    replay: SwitchDifferentiableReplay,
    basis: Tensor,
    factual_norm: float,
    base_visible: Tensor,
    target_ids: Tensor,
    gradient: Tensor,
    chart_matrix: Tensor,
    *,
    target_kl: float,
) -> tuple[dict[str, Any], Tensor | None]:
    direction = chart_euclidean_update(
        gradient, chart_matrix, predicted_gain=1.0
    )
    maximum_relative = float(config["updates"]["maximum_relative_hidden_l2"])
    high = maximum_relative * factual_norm / float(torch.linalg.vector_norm(direction))
    objective_horizon = int(config["intervention"]["objective_visible_tokens"][1])
    _, high_visible = _evaluate_logits(replay, high * direction, basis)
    high_kl = summed_categorical_kl(
        base_visible, high_visible, start=0, end=objective_horizon
    )
    if high_kl < target_kl:
        return {
            "success": False,
            "reason": "target_kl_not_reached_within_radius",
            "maximum_prefix8_kl": high_kl,
        }, None
    low = 0.0
    final_visible = high_visible
    for _ in range(int(config["updates"]["bisection_iterations"])):
        middle = 0.5 * (low + high)
        _, visible = _evaluate_logits(replay, middle * direction, basis)
        measured = summed_categorical_kl(
            base_visible, visible, start=0, end=objective_horizon
        )
        if measured < target_kl:
            low = middle
        else:
            high = middle
            final_visible = visible
    coefficients = high * direction
    record = {
        "success": True,
        "coefficients": coefficients.tolist(),
        "relative_hidden_l2": float(
            torch.linalg.vector_norm(coefficients) / factual_norm
        ),
        "target_prefix8_kl": target_kl,
    }
    record.update(
        _outcome(
            base_visible,
            final_visible,
            target_ids,
            objective_horizon=objective_horizon,
            secondary_start=int(
                config["intervention"]["secondary_holdout_visible_tokens"][0]
            )
            - 1,
            strict_start=int(
                config["intervention"]["strict_holdout_visible_tokens"][0]
            )
            - 1,
        )
    )
    return record, final_visible


def _free_rollout(
    config: dict[str, Any],
    switch_model: Any,
    plan: SwitchReplayPlan,
    basis: Tensor,
    coefficients: Tensor,
) -> tuple[dict[str, Any], int]:
    device = switch_model.embedding.weight.device
    delta = basis @ coefficients.to(device=basis.device, dtype=torch.float32)

    def operation(proposed: Tensor, block: int, step: int) -> Tensor:
        if block == 0 and step == 0:
            return proposed + delta.to(device=proposed.device, dtype=proposed.dtype)
        return proposed

    prompt = plan.prompt_input_ids.to(device).view(1, -1)
    mask = plan.prompt_attention_mask.to(device).view(1, -1)
    run = SwitchAuditRunner(switch_model).run(
        prompt,
        mask,
        max_new_tokens=int(config["selection"]["maximum_new_visible_tokens"]),
        min_latent_steps=int(config["selection"]["minimum_latent_dwell"]),
        operation=operation,
        capture_trace=False,
    )
    generated = run.output_ids[0, prompt.shape[1] :].detach().cpu()
    post_block = None
    if run.latent_info:
        end = int(run.latent_info[0]["position"])
        candidate = generated[end + 1 : end + 65]
        if candidate.numel() == 64:
            post_block = candidate
    factual = plan.visible_target_ids.detach().cpu()
    mismatch = True
    mismatch_fraction = 1.0
    first_mismatch = 0
    if post_block is not None:
        differences = post_block != factual
        mismatch = bool(differences.any())
        mismatch_fraction = float(differences.to(torch.float64).mean())
        positions = torch.nonzero(differences).flatten()
        first_mismatch = int(positions[0]) if positions.numel() else None
    payload = json.dumps(generated.tolist(), separators=(",", ":")).encode("utf-8")
    return (
        {
            "generated_visible_token_count": int(generated.numel()),
            "generated_ids_sha256": hashlib.sha256(payload).hexdigest(),
            "post_block_ids": post_block.tolist() if post_block is not None else None,
            "post_block_token_mismatch": mismatch,
            "post_block_mismatch_fraction": mismatch_fraction,
            "first_mismatch_index": first_mismatch,
            "latent_steps": [int(item["n_latent_steps"]) for item in run.latent_info],
            "natural_exits": [bool(item["natural_exit"]) for item in run.latent_info],
            "model_forward_calls": run.model_forward_calls,
        },
        run.model_forward_calls,
    )


def _finite_difference_controls(
    config: dict[str, Any],
    replay: SwitchDifferentiableReplay,
    basis: Tensor,
    factual_norm: float,
    target_ids: Tensor,
    gradient: Tensor,
    *,
    scan_rank: int,
) -> list[dict[str, float]]:
    dimension = int(basis.shape[1])
    generator = torch.Generator(device="cpu").manual_seed(
        int(config["derivatives"]["finite_difference_seed"]) + scan_rank
    )
    direction = torch.randn(dimension, generator=generator, dtype=torch.float64)
    direction = direction / torch.linalg.vector_norm(direction)
    analytic = float(torch.dot(gradient, direction))
    objective_horizon = int(config["intervention"]["objective_visible_tokens"][1])
    records = []
    for relative in config["derivatives"]["finite_difference_relative_step_grid"]:
        step = float(relative) * factual_norm
        _, plus = _evaluate_logits(replay, step * direction, basis)
        _, minus = _evaluate_logits(replay, -step * direction, basis)
        plus_utility = mean_factual_log_probability(
            plus, target_ids, start=0, end=objective_horizon
        )
        minus_utility = mean_factual_log_probability(
            minus, target_ids, start=0, end=objective_horizon
        )
        measured = (plus_utility - minus_utility) / (2.0 * step)
        error = abs(measured - analytic) / max(abs(analytic), 1e-12)
        records.append(
            {
                "relative_hidden_l2": float(relative),
                "analytic_directional_derivative": analytic,
                "finite_difference_directional_derivative": measured,
                "relative_error": error,
            }
        )
    return records


def _measure_prompt(
    config: dict[str, Any],
    bundle: Any,
    selected: dict[str, Any],
    whitening_precision: Tensor,
    *,
    probe_scales: list[float],
    gains: list[float],
    run_finite_difference: bool,
    exact_kl_budget: float | None,
    free_method_names: list[str] | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    plan = _load_plan(selected["replay_plan"])
    replay = SwitchDifferentiableReplay(bundle.model, plan)
    device = bundle.device
    factual = replay.factual_latent().detach().to(device=device, dtype=torch.float32)
    factual_norm = float(torch.linalg.vector_norm(factual))
    candidate = factual.detach().clone().requires_grad_(True)
    target_ids = plan.visible_target_ids.to(device)
    objective_horizon = int(config["intervention"]["objective_visible_tokens"][1])
    latent_direct, visible_direct = replay.logits_from_candidate(
        candidate, visible_horizon=objective_horizon
    )
    objective = _objective_tensor(visible_direct, target_ids, objective_horizon)
    full_gradient = torch.autograd.grad(objective, candidate, create_graph=False)[0].detach()
    if float(torch.linalg.vector_norm(full_gradient)) < float(
        config["subspace"]["minimum_full_gradient_l2"]
    ):
        raise ValueError("full hidden objective gradient is below the frozen minimum")
    basis_cpu = consequential_basis(
        full_gradient,
        dimension=int(config["subspace"]["dimension"]),
        seed=int(config["subspace"]["seed"]) + int(selected["scan_rank"]),
    )
    basis = basis_cpu.to(device=device, dtype=torch.float32)
    basis_error = float(
        (
            basis_cpu.transpose(0, 1) @ basis_cpu
            - torch.eye(basis_cpu.shape[1], dtype=torch.float64)
        )
        .abs()
        .max()
    )
    base_latent, base_visible, latent_jacobian, visible_jacobian, jvp_calls = (
        _measure_jacobian(replay, basis)
    )
    zero_point_error = max(
        float((base_latent - latent_direct.detach()).abs().max()),
        float(
            (base_visible[:objective_horizon] - visible_direct.detach()).abs().max()
        ),
    )
    latent_geometry = prefix_geometry(base_latent, latent_jacobian)
    visible_geometry = prefix_geometry(
        base_visible,
        visible_jacobian,
        target_ids=target_ids,
        objective_horizon=objective_horizon,
    )
    gradient = visible_geometry.objective_gradient.detach().cpu()
    projected_full = (basis.transpose(0, 1) @ full_gradient).detach().cpu().to(
        torch.float64
    )
    projected_gradient_error = _relative_error(projected_full, gradient)
    whitening = projected_diagonal_metric(basis_cpu, whitening_precision)
    raw_metrics, semantic_horizon = _metric_dictionary(
        config,
        bundle.tokenizer,
        plan.visible_target_ids,
        latent_geometry,
        visible_geometry,
        whitening,
    )
    regularized, ridges = _regularized_metrics(
        raw_metrics,
        whitening,
        relative_ridge=float(
            config["metrics"]["regularization"]["relative_generalized_ridge"]
        ),
    )
    charts = _chart_matrices(config, int(basis.shape[1]))
    transport = _metric_transport_controls(
        gradient,
        regularized,
        charts,
        predicted_gain=float(gains[0]),
    )
    probes = _measure_probes(
        config,
        replay,
        basis,
        factual_norm,
        base_visible,
        target_ids,
        raw_metrics,
        scan_rank=int(selected["scan_rank"]),
        scales=probe_scales,
    )
    updates = _measure_updates(
        config,
        replay,
        basis,
        factual_norm,
        base_visible,
        target_ids,
        gradient,
        regularized,
        charts,
        gains=gains,
    )
    finite_difference = (
        _finite_difference_controls(
            config,
            replay,
            basis,
            factual_norm,
            target_ids,
            gradient,
            scan_rank=int(selected["scan_rank"]),
        )
        if run_finite_difference
        else []
    )
    retuned = None
    free_rollouts = None
    free_forward_calls = 0
    free_visible_tokens = 0
    if exact_kl_budget is not None:
        identity_name = "identity"
        strongest_name = max(
            charts, key=lambda name: float(torch.linalg.cond(charts[name]))
        )
        identity_record, identity_logits = _retune_chart_to_exact_v8_kl(
            config,
            replay,
            basis,
            factual_norm,
            base_visible,
            target_ids,
            gradient,
            charts[identity_name],
            target_kl=exact_kl_budget,
        )
        strongest_record, strongest_logits = _retune_chart_to_exact_v8_kl(
            config,
            replay,
            basis,
            factual_norm,
            base_visible,
            target_ids,
            gradient,
            charts[strongest_name],
            target_kl=exact_kl_budget,
        )
        pair_symmetric_kl = None
        if identity_logits is not None and strongest_logits is not None:
            start = int(config["intervention"]["strict_holdout_visible_tokens"][0]) - 1
            forward = summed_categorical_kl(
                identity_logits,
                strongest_logits,
                start=start,
                end=identity_logits.shape[0],
            )
            reverse = summed_categorical_kl(
                strongest_logits,
                identity_logits,
                start=start,
                end=identity_logits.shape[0],
            )
            pair_symmetric_kl = 0.5 * (forward + reverse)
        retuned = {
            "identity_chart": identity_record,
            "condition12_chart_name": strongest_name,
            "condition12_chart": strongest_record,
            "strict_pair_symmetric_kl": pair_symmetric_kl,
        }
        if free_method_names is not None:
            gain_record = updates[0]
            coefficient_map: dict[str, Tensor] = {}
            if identity_record.get("success"):
                coefficient_map["coordinate_euclidean_identity"] = torch.tensor(
                    identity_record["coefficients"], dtype=torch.float64
                )
            if strongest_record.get("success"):
                coefficient_map["coordinate_euclidean_condition_12"] = torch.tensor(
                    strongest_record["coefficients"], dtype=torch.float64
                )
            for name in free_method_names:
                if name in gain_record["methods"]:
                    coefficient_map[name] = torch.tensor(
                        gain_record["methods"][name]["coefficients"],
                        dtype=torch.float64,
                    )
            free_rollouts = {}
            for name, coefficients in coefficient_map.items():
                result, calls = _free_rollout(
                    config, bundle.model, plan, basis, coefficients
                )
                free_rollouts[name] = result
                free_forward_calls += calls
                free_visible_tokens += int(result["generated_visible_token_count"])
    del latent_jacobian, visible_jacobian, latent_direct, visible_direct
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "kind": "prompt",
        "scan_rank": int(selected["scan_rank"]),
        "dataset_index": int(selected["dataset_index"]),
        "unique_id": str(selected["unique_id"]),
        "subject": str(selected["subject"]),
        "level": int(selected["level"]),
        "selected_split": str(selected["selected_split"]),
        "selected_index": int(selected["selected_index"]),
        "semantic_horizon": semantic_horizon,
        "factual_hidden_l2": factual_norm,
        "full_gradient_l2": float(torch.linalg.vector_norm(full_gradient)),
        "subspace_gradient": gradient.tolist(),
        "basis_orthogonality_max_abs_error": basis_error,
        "zero_point_logit_max_abs_error": zero_point_error,
        "projected_gradient_relative_error": projected_gradient_error,
        "metric_ridges": ridges,
        "transport_controls": transport,
        "finite_difference": finite_difference,
        "probes": probes,
        "updates": updates,
        "exact_kl_retuned_coordinate_updates": retuned,
        "free_rollouts": free_rollouts,
        "runtime": {
            "seconds": time.perf_counter() - started,
            "prefix_evaluations": replay.prefix_evaluations,
            "prefix_cache_builds": replay.prefix_cache_builds,
            "prefix_cache_hits": replay.prefix_cache_hits,
            "logit_evaluations": replay.logit_evaluations,
            "model_forward_calls": replay.model_forward_calls + free_forward_calls,
            "full_gradient_backward_calls": 1,
            "jvp_calls": jvp_calls,
            "visible_logit_tokens": replay.visible_logit_tokens,
            "latent_logit_steps": replay.latent_logit_steps,
            "free_rollout_visible_tokens": free_visible_tokens,
        },
    }


def _read_journal(
    path: Path,
    *,
    phase: str,
    config_sha256: str,
    eligibility_sha256: str,
    calibration_sha256: str | None,
    implementation: dict[str, str],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if not path.exists():
        return None, []
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or rows[0].get("kind") != "header":
        raise ValueError("scientific journal lacks a header")
    header = rows[0]
    expected = {
        "phase": phase,
        "config_sha256": config_sha256,
        "eligibility_sha256": eligibility_sha256,
        "calibration_sha256": calibration_sha256,
        "implementation_sha256": implementation,
    }
    for key, value in expected.items():
        if header.get(key) != value:
            raise ValueError(f"scientific journal header mismatch for {key}")
    return header, rows[1:]


def _write_header(path: Path, header: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(header, sort_keys=True) + "\n", encoding="utf-8")


def _append_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def _find_probe(record: dict[str, Any], scale: float) -> dict[str, Any]:
    return next(
        item
        for item in record["probes"]
        if math.isclose(float(item["relative_hidden_l2"]), scale, abs_tol=1e-15)
    )


def _find_update(record: dict[str, Any], gain: float) -> dict[str, Any]:
    return next(
        item
        for item in record["updates"]
        if math.isclose(float(item["predicted_objective_gain"]), gain, abs_tol=1e-15)
    )


def _select_calibration(
    config: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    simple_order = list(config["updates"]["simple_baseline_selection_order"])
    tolerance = float(config["updates"]["selection_tie_tolerance"])
    scale_records = []
    for scale in config["probe_bank"]["relative_hidden_l2_grid"]:
        means = {}
        for name in simple_order + ["visible_prefix_32", "visible_prefix_64"]:
            values = [
                _find_probe(record, float(scale))["metrics"][name]["strict_spearman"]
                for record in records
            ]
            means[name] = statistics.fmean(values)
        best_simple = simple_order[0]
        for name in simple_order[1:]:
            if means[name] > means[best_simple] + tolerance:
                best_simple = name
        margin = means["visible_prefix_32"] - means[best_simple]
        scale_records.append(
            {
                "relative_hidden_l2": float(scale),
                "mean_spearman": means,
                "best_simple": best_simple,
                "v32_margin": margin,
            }
        )
    selected_scale = scale_records[0]
    for item in scale_records[1:]:
        if item["v32_margin"] > selected_scale["v32_margin"] + tolerance:
            selected_scale = item

    best_simple = selected_scale["best_simple"]
    gain_records = []
    minimum_objective_ratio = float(
        config["updates"]["minimum_calibration_objective_gain_ratio"]
    )
    for gain in config["updates"]["predicted_objective_gain_grid"]:
        candidate = [
            _find_update(record, float(gain))["methods"]["visible_prefix_32"]
            for record in records
        ]
        baseline = [
            _find_update(record, float(gain))["methods"][best_simple]
            for record in records
        ]
        valid = all(
            row["valid_radius"] and "strict_kl" in row
            for row in candidate + baseline
        )
        if valid:
            candidate_kl = statistics.fmean(row["strict_kl"] for row in candidate)
            baseline_kl = statistics.fmean(row["strict_kl"] for row in baseline)
            candidate_objective = statistics.fmean(
                row["objective_utility_gain"] for row in candidate
            )
            baseline_objective = statistics.fmean(
                row["objective_utility_gain"] for row in baseline
            )
            ratio = candidate_kl / max(baseline_kl, 1e-30)
            objective_ratio = (
                candidate_objective / baseline_objective
                if baseline_objective > 0.0
                else -math.inf
            )
            valid = (
                baseline_objective > 0.0
                and candidate_objective
                >= minimum_objective_ratio * baseline_objective
            )
        else:
            ratio = math.inf
            objective_ratio = -math.inf
        gain_records.append(
            {
                "predicted_objective_gain": float(gain),
                "valid": valid,
                "v32_to_best_simple_strict_kl_ratio": ratio,
                "v32_to_best_simple_objective_gain_ratio": objective_ratio,
            }
        )
    valid_gains = [item for item in gain_records if item["valid"]]
    if not valid_gains:
        raise ValueError("no calibration gain satisfies radius and objective controls")
    selected_gain = valid_gains[0]
    for item in valid_gains[1:]:
        if (
            item["v32_to_best_simple_strict_kl_ratio"]
            < selected_gain["v32_to_best_simple_strict_kl_ratio"] - tolerance
        ):
            selected_gain = item
    selected_gain_value = float(selected_gain["predicted_objective_gain"])
    exact_kl_budget = statistics.median(
        _find_update(record, selected_gain_value)["methods"]["visible_prefix_32"][
            "prefix8_kl"
        ]
        for record in records
    )

    fd_records = [record for record in records if record["finite_difference"]]
    fd_selection = None
    if fd_records:
        candidates = []
        for relative in config["derivatives"]["finite_difference_relative_step_grid"]:
            errors = [
                next(
                    item["relative_error"]
                    for item in record["finite_difference"]
                    if math.isclose(
                        float(item["relative_hidden_l2"]),
                        float(relative),
                        abs_tol=1e-15,
                    )
                )
                for record in fd_records
            ]
            candidates.append(
                {
                    "relative_hidden_l2": float(relative),
                    "median_relative_error": statistics.median(errors),
                    "p90_relative_error": float(
                        torch.quantile(torch.tensor(errors), 0.9)
                    ),
                }
            )
        fd_selection = min(
            candidates,
            key=lambda item: (
                item["median_relative_error"], item["relative_hidden_l2"]
            ),
        )
    return {
        "probe_scale_candidates": scale_records,
        "selected_probe_relative_hidden_l2": selected_scale["relative_hidden_l2"],
        "selected_best_simple_baseline": best_simple,
        "gain_candidates": gain_records,
        "selected_predicted_objective_gain": selected_gain_value,
        "selected_exact_v8_kl_budget": exact_kl_budget,
        "finite_difference_selection": fd_selection,
    }


def _bootstrap_ratio(
    numerator: list[float],
    denominator: list[float],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> dict[str, float]:
    left = torch.tensor(numerator, dtype=torch.float64)
    right = torch.tensor(denominator, dtype=torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(left.numel(), (samples, left.numel()), generator=generator)
    ratios = left[indices].mean(dim=1) / right[indices].mean(dim=1).clamp_min(1e-30)
    tail = (1.0 - confidence) / 2.0
    interval = torch.quantile(
        ratios, torch.tensor([tail, 1.0 - tail], dtype=ratios.dtype)
    )
    return {
        "value": float(left.mean() / right.mean().clamp_min(1e-30)),
        "ci_low": float(interval[0]),
        "ci_high": float(interval[1]),
        "count": int(left.numel()),
    }


def _analyze_test(
    config: dict[str, Any],
    records: list[dict[str, Any]],
    calibration: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, bool]]:
    selections = calibration["selections"]
    scale = float(selections["selected_probe_relative_hidden_l2"])
    gain = float(selections["selected_predicted_objective_gain"])
    best_simple = str(selections["selected_best_simple_baseline"])
    v32_spearman = []
    simple_spearman = []
    oracle_spearman = []
    v32_recall = []
    simple_recall = []
    v32_kl = []
    simple_kl = []
    utility_difference = []
    euclidean_discrepancy = []
    chart_affected = []
    retune_success = []
    for record in records:
        probe = _find_probe(record, scale)
        v32_spearman.append(probe["metrics"]["visible_prefix_32"]["strict_spearman"])
        simple_spearman.append(probe["metrics"][best_simple]["strict_spearman"])
        oracle_spearman.append(probe["metrics"]["visible_prefix_64"]["strict_spearman"])
        v32_recall.append(
            probe["metrics"]["visible_prefix_32"]["strict_top_risk_recall"]
        )
        simple_recall.append(
            probe["metrics"][best_simple]["strict_top_risk_recall"]
        )
        update = _find_update(record, gain)
        candidate = update["methods"]["visible_prefix_32"]
        baseline = update["methods"][best_simple]
        v32_kl.append(candidate["strict_kl"])
        simple_kl.append(baseline["strict_kl"])
        utility_difference.append(
            candidate["strict_utility_gain"] - baseline["strict_utility_gain"]
        )
        euclidean_discrepancy.append(
            record["transport_controls"][
                "condition12_euclidean_direction_discrepancy"
            ]
        )
        retuned = record["exact_kl_retuned_coordinate_updates"]
        success = bool(
            retuned["identity_chart"]["success"]
            and retuned["condition12_chart"]["success"]
        )
        retune_success.append(float(success))
        free = record["free_rollouts"] or {}
        token_change = False
        left = free.get("coordinate_euclidean_identity")
        right = free.get("coordinate_euclidean_condition_12")
        if left is not None and right is not None:
            token_change = left.get("post_block_ids") != right.get("post_block_ids")
        chart_affected.append(
            bool(
                token_change
                or (
                    retuned["strict_pair_symmetric_kl"] is not None
                    and retuned["strict_pair_symmetric_kl"]
                    >= float(
                        config["decision_rules"][
                            "chart_affected_min_strict_symmetric_kl"
                        ]
                    )
                )
            )
        )

    stats_config = config["statistics"]
    samples = int(stats_config["bootstrap_samples"])
    confidence = float(stats_config["confidence"])
    seed = int(stats_config["seed"])
    spearman_margin = bootstrap_mean(
        [left - right for left, right in zip(v32_spearman, simple_spearman)],
        bootstrap_samples=samples,
        confidence=confidence,
        seed=seed,
    ).to_dict()
    recall_margin = bootstrap_mean(
        [left - right for left, right in zip(v32_recall, simple_recall)],
        bootstrap_samples=samples,
        confidence=confidence,
        seed=seed + 1,
    ).to_dict()
    kl_ratio = _bootstrap_ratio(
        v32_kl,
        simple_kl,
        samples=samples,
        confidence=confidence,
        seed=seed + 2,
    )
    utility = bootstrap_mean(
        utility_difference,
        bootstrap_samples=samples,
        confidence=confidence,
        seed=seed + 3,
    ).to_dict()
    oracle_gap = statistics.fmean(oracle_spearman) - statistics.fmean(v32_spearman)
    summary = {
        "selected_probe_relative_hidden_l2": scale,
        "selected_predicted_objective_gain": gain,
        "selected_best_simple_baseline": best_simple,
        "v32_spearman_mean": statistics.fmean(v32_spearman),
        "best_simple_spearman_mean": statistics.fmean(simple_spearman),
        "v64_oracle_spearman_mean": statistics.fmean(oracle_spearman),
        "v32_spearman_margin": spearman_margin,
        "v32_top_risk_recall_margin": recall_margin,
        "v32_to_best_simple_strict_kl_ratio": kl_ratio,
        "v32_minus_best_simple_strict_utility": utility,
        "v64_minus_v32_spearman_gap": oracle_gap,
        "condition12_euclidean_direction_discrepancy_median": statistics.median(
            euclidean_discrepancy
        ),
        "chart_affected_prompt_fraction": statistics.fmean(chart_affected),
        "exact_kl_retuning_success_fraction": statistics.fmean(retune_success),
    }
    rules = config["decision_rules"]
    numerical = all(
        record["zero_point_logit_max_abs_error"]
        <= float(config["derivatives"]["maximum_zero_point_logit_absolute_error"])
        and record["basis_orthogonality_max_abs_error"]
        <= float(config["subspace"]["maximum_basis_orthogonality_error"])
        and record["projected_gradient_relative_error"]
        <= float(rules["maximum_projected_gradient_relative_error"])
        and record["transport_controls"][
            "maximum_metric_update_transport_relative_error"
        ]
        <= float(rules["maximum_metric_update_transport_relative_error"])
        and record["transport_controls"][
            "orthogonal_euclidean_direction_discrepancy"
        ]
        <= float(rules["maximum_orthogonal_euclidean_direction_discrepancy"])
        for record in records
    )
    gates = {
        "identity_precision_transport_controls": numerical,
        "condition12_euclidean_direction_discrepancy": summary[
            "condition12_euclidean_direction_discrepancy_median"
        ]
        >= float(rules["minimum_median_condition12_euclidean_direction_discrepancy"]),
        "chart_affected_prompt_fraction": summary["chart_affected_prompt_fraction"]
        >= float(rules["minimum_chart_affected_prompt_fraction"]),
        "v32_spearman_margin": spearman_margin["value"]
        >= float(rules["minimum_v32_spearman_margin_over_best_simple"]),
        "v32_spearman_margin_ci": spearman_margin["ci_low"]
        >= float(rules["minimum_v32_spearman_margin_ci_low"]),
        "v32_top_risk_recall_margin": recall_margin["value"]
        >= float(rules["minimum_v32_top_risk_recall_margin"]),
        "v32_strict_kl_ratio": kl_ratio["value"]
        <= float(rules["maximum_v32_to_best_simple_strict_holdout_kl_ratio"]),
        "v32_strict_kl_ratio_ci": kl_ratio["ci_high"]
        < float(rules["maximum_v32_to_best_simple_strict_holdout_kl_ratio_ci_high"]),
        "v32_strict_utility": utility["ci_low"]
        >= float(rules["minimum_v32_minus_best_simple_strict_holdout_utility_ci_low"]),
        "v32_near_v64_oracle": oracle_gap
        <= float(rules["maximum_v32_spearman_gap_to_v64_oracle"]),
        "exact_kl_retuning_success": summary["exact_kl_retuning_success_fraction"]
        >= float(rules["minimum_exact_kl_retuning_success_fraction"]),
    }
    return summary, gates


def run_phase(
    config: dict[str, Any],
    workspace_root: Path,
    *,
    phase: str,
    eligibility_path: Path,
    journal_path: Path,
    calibration_path: Path | None,
) -> dict[str, Any]:
    import accelerate
    import peft
    import transformers

    started = time.perf_counter()
    workspace_root = workspace_root.resolve()
    config_sha256 = canonical_config_hash(config)
    implementation = implementation_hashes()
    eligibility_path = eligibility_path.resolve()
    eligibility_sha256 = _sha256(eligibility_path)
    eligibility = json.loads(eligibility_path.read_text(encoding="utf-8"))
    if eligibility.get("status") != "pass" or eligibility.get(
        "config_sha256"
    ) != config_sha256:
        raise ValueError("eligibility artifact did not pass this frozen C2 config")
    if phase not in {"calibration", "test"}:
        raise ValueError("phase must be calibration or test")
    calibration = None
    calibration_sha256 = None
    if phase == "test":
        if calibration_path is None:
            raise ValueError("test phase requires a calibration artifact")
        calibration_path = calibration_path.resolve()
        calibration_sha256 = _sha256(calibration_path)
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration.get("status") != "pass" or calibration.get(
            "config_sha256"
        ) != config_sha256:
            raise ValueError("calibration artifact did not pass this frozen C2 config")
    selected = list(eligibility["scan"][f"selected_{phase}"])
    expected_count = int(config["selection"][f"{phase}_eligible_prompts"])
    if len(selected) != expected_count:
        raise ValueError(f"eligibility artifact has the wrong {phase} count")

    device = torch.device(str(config["runtime"]["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("the frozen SWITCH C2 scientific gate requires CUDA")
    torch.manual_seed(int(config["runtime"]["seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(config["runtime"]["seed"]))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        if bool(config["runtime"].get("disable_tf32", True)):
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    journal_path = journal_path.resolve()
    header, records = _read_journal(
        journal_path,
        phase=phase,
        config_sha256=config_sha256,
        eligibility_sha256=eligibility_sha256,
        calibration_sha256=calibration_sha256,
        implementation=implementation,
    )
    for expected, record in zip(selected, records):
        if int(record["scan_rank"]) != int(expected["scan_rank"]):
            raise ValueError("scientific journal does not follow frozen assignment")

    model_config = config["model"]
    bundle = load_public_switch(
        source_directory=workspace_root / str(model_config["source_relative_path"]),
        cache_directory=workspace_root / str(model_config["cache_relative_path"]),
        device=device,
        attention_implementation=str(model_config["attention_implementation"]),
    )
    if header is None:
        if phase == "calibration":
            state_rows = []
            state_calls = 0
            for item in selected:
                replay = SwitchDifferentiableReplay(
                    bundle.model, _load_plan(item["replay_plan"])
                )
                state_rows.append(replay.factual_latent().detach().cpu().to(torch.float64))
                state_calls += replay.model_forward_calls
            states = torch.stack(state_rows)
            regularization = config["metrics"]["regularization"]
            precision, variance_floor = fit_diagonal_whitening_precision(
                states,
                variance_floor_fraction_of_median=float(
                    regularization["variance_floor_fraction_of_median"]
                ),
            )
        else:
            precision = torch.tensor(
                calibration["whitening"]["diagonal_precision"], dtype=torch.float64
            )
            variance_floor = float(calibration["whitening"]["variance_floor"])
            state_calls = 0
        header = {
            "kind": "header",
            "phase": phase,
            "config_sha256": config_sha256,
            "eligibility_sha256": eligibility_sha256,
            "calibration_sha256": calibration_sha256,
            "implementation_sha256": implementation,
            "whitening_precision": precision.tolist(),
            "whitening_variance_floor": variance_floor,
            "state_collection_model_forward_calls": state_calls,
        }
        _write_header(journal_path, header)
    else:
        precision = torch.tensor(header["whitening_precision"], dtype=torch.float64)

    if phase == "calibration":
        probe_scales = [
            float(value) for value in config["probe_bank"]["relative_hidden_l2_grid"]
        ]
        gains = [
            float(value)
            for value in config["updates"]["predicted_objective_gain_grid"]
        ]
        exact_budget = None
        free_names = None
    else:
        selections = calibration["selections"]
        probe_scales = [float(selections["selected_probe_relative_hidden_l2"])]
        gains = [float(selections["selected_predicted_objective_gain"])]
        exact_budget = float(selections["selected_exact_v8_kl_budget"])
        free_names = list(
            dict.fromkeys(
                [
                    "visible_prefix_1",
                    str(selections["selected_best_simple_baseline"]),
                    "visible_prefix_32",
                ]
            )
        )

    fd_count = int(config["derivatives"]["finite_difference_calibration_prompts"])
    for item in selected[len(records) :]:
        record = _measure_prompt(
            config,
            bundle,
            item,
            precision,
            probe_scales=probe_scales,
            gains=gains,
            run_finite_difference=(
                phase == "calibration" and int(item["selected_index"]) < fd_count
            ),
            exact_kl_budget=exact_budget,
            free_method_names=free_names,
        )
        records.append(record)
        _append_record(journal_path, record)

    complete = len(records) == expected_count
    if phase == "calibration":
        selections = _select_calibration(config, records) if complete else None
        fd = selections["finite_difference_selection"] if selections else None
        controls = bool(
            complete
            and fd is not None
            and fd["median_relative_error"]
            <= float(config["derivatives"]["maximum_median_relative_error"])
            and fd["p90_relative_error"]
            <= float(config["derivatives"]["maximum_p90_relative_error"])
            and all(
                record["zero_point_logit_max_abs_error"]
                <= float(config["derivatives"]["maximum_zero_point_logit_absolute_error"])
                and record["basis_orthogonality_max_abs_error"]
                <= float(config["subspace"]["maximum_basis_orthogonality_error"])
                for record in records
            )
        )
        gates = {
            "complete_calibration": complete,
            "finite_difference": controls,
        }
        analysis = {"selections": selections}
    else:
        analysis, scientific_gates = _analyze_test(config, records, calibration)
        test_subjects = {record["subject"] for record in records}
        test_levels = {int(record["level"]) for record in records}
        gates = {
            "complete_test": complete,
            "test_subject_diversity": len(test_subjects)
            >= int(config["decision_rules"]["required_test_subjects"]),
            "test_level_diversity": len(test_levels)
            >= int(config["decision_rules"]["required_test_levels"]),
            **scientific_gates,
        }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "experiment_name": config["experiment_name"] + f"-{phase}",
        "phase": phase,
        "status": "pass" if all(gates.values()) else "fail",
        "evidence_level": (
            "calibration-only hyperparameter selection"
            if phase == "calibration"
            else "frozen held-out SWITCH C2 scientific gate"
        ),
        "config_sha256": config_sha256,
        "implementation_sha256": implementation,
        "eligibility_artifact_sha256": eligibility_sha256,
        "calibration_artifact_sha256": calibration_sha256,
        "checkpoint": {
            "base_id": SWITCH_BASE_MODEL_ID,
            "base_revision": SWITCH_BASE_REVISION,
            "adapter_id": SWITCH_ADAPTER_ID,
            "adapter_revision": SWITCH_ADAPTER_REVISION,
            "source_commit": bundle.source_commit,
        },
        "whitening": {
            "diagonal_precision": precision.tolist(),
            "variance_floor": float(header["whitening_variance_floor"]),
        },
        "selections": analysis.get("selections") if phase == "calibration" else calibration["selections"],
        "analysis": analysis,
        "gates": gates,
        "records": records,
        "runtime": {
            "seconds_current_process": time.perf_counter() - started,
            "summed_prompt_seconds": sum(record["runtime"]["seconds"] for record in records),
            "model_forward_calls": int(header["state_collection_model_forward_calls"])
            + sum(record["runtime"]["model_forward_calls"] for record in records),
            "full_gradient_backward_calls": sum(
                record["runtime"]["full_gradient_backward_calls"] for record in records
            ),
            "jvp_calls": sum(record["runtime"]["jvp_calls"] for record in records),
            "visible_logit_tokens": sum(
                record["runtime"]["visible_logit_tokens"] for record in records
            ),
            "latent_logit_steps": sum(
                record["runtime"]["latent_logit_steps"] for record in records
            ),
            "free_rollout_visible_tokens": sum(
                record["runtime"]["free_rollout_visible_tokens"]
                for record in records
            ),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "accelerate": accelerate.__version__,
            "device": str(device),
            "cuda_device": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "cuda_peak_allocated_mib_current_process": (
                float(torch.cuda.max_memory_allocated(device) / 1024**2)
                if device.type == "cuda"
                else None
            ),
        },
        "interpretation": (
            "Calibration status selects frozen nuisance settings only. A test "
            "pass is required before an FCTR estimator or training pilot is authorized."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=["calibration", "test"])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--eligibility-artifact", required=True, type=Path)
    parser.add_argument("--calibration-artifact", type=Path)
    parser.add_argument("--journal", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = run_phase(
        config,
        args.workspace_root,
        phase=args.phase,
        eligibility_path=args.eligibility_artifact,
        journal_path=args.journal,
        calibration_path=args.calibration_artifact,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    compact = {key: value for key, value in report.items() if key not in {"records", "whitening"}}
    print(json.dumps(compact, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
