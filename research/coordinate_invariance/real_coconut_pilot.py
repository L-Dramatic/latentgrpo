from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor

from .charts import AffineChart, LatentChart, SinhChart
from .evaluator import ContinuationEvaluation, ContinuationEvaluator
from .operations import add_isotropic_noise_in_chart, interpolate_in_chart
from .real_models.coconut import (
    CoconutChartRunner,
    CoconutContinuationAdapter,
    load_public_coconut,
    prepare_coconut_input,
)
from .statistics import Estimate, bootstrap_mean
from .trace import LatentStepRecord, LatentTrace


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("unsupported public Coconut pilot config")
    return config


def _load_examples(path: Path, count: int, seed: int) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) < count:
        raise ValueError("dataset does not contain the requested number of examples")
    generator = random.Random(seed)
    indices = list(range(len(raw)))
    generator.shuffle(indices)
    selected: list[dict[str, Any]] = []
    for index in indices[:count]:
        item = raw[index]
        if not isinstance(item, dict) or not isinstance(item.get("query"), str):
            raise ValueError("dataset examples must contain a string query")
        selected.append(
            {
                "dataset_index": index,
                "query": item["query"],
                "answer": str(item.get("answer", "")),
            }
        )
    return selected


def _build_charts(config: dict[str, Any], dimension: int) -> dict[str, LatentChart]:
    charts: dict[str, LatentChart] = {
        "identity": AffineChart.identity(
            dimension, dtype=torch.float64, compute_dtype=torch.float64
        )
    }
    for seed in config["chart_seeds"]:
        orthogonal = AffineChart.random_orthogonal(
            dimension,
            seed=int(seed),
            dtype=torch.float64,
            compute_dtype=torch.float64,
        )
        charts[orthogonal.name] = orthogonal
        for condition in config["affine_condition_numbers"]:
            affine = AffineChart.with_condition_number(
                dimension,
                float(condition),
                seed=int(seed),
                dtype=torch.float64,
                compute_dtype=torch.float64,
            )
            charts[affine.name] = affine
    for scale in config["nonlinear_scales"]:
        chart = SinhChart(
            dimension,
            scale=float(scale),
            dtype=torch.float64,
            compute_dtype=torch.float64,
            name=f"sinh-scale-{float(scale):g}",
        )
        charts[chart.name] = chart
    return charts


def _distance_matrix(latents: Tensor, chart: LatentChart, metric: str) -> Tensor:
    encoded = chart.encode(latents)
    if metric == "euclidean":
        distances = torch.cdist(encoded, encoded)
    elif metric == "cosine":
        normalized = encoded / torch.linalg.vector_norm(
            encoded, dim=-1, keepdim=True
        ).clamp_min(1e-12)
        distances = 1.0 - (normalized @ normalized.transpose(0, 1)).clamp(-1.0, 1.0)
    else:
        raise ValueError(f"unsupported metric: {metric}")
    distances = distances.clone()
    distances.fill_diagonal_(float("inf"))
    return distances


def _mean_ranking_disagreement(reference: Tensor, candidate: Tensor) -> float:
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("distance matrices must be shape matched")
    values: list[Tensor] = []
    for row_index in range(reference.shape[0]):
        mask = torch.arange(reference.shape[0]) != row_index
        left = reference[row_index, mask]
        right = candidate[row_index, mask]
        upper = torch.triu_indices(left.numel(), left.numel(), offset=1)
        left_order = torch.sign(left[upper[0]] - left[upper[1]])
        right_order = torch.sign(right[upper[0]] - right[upper[1]])
        non_ties = (left_order != 0) & (right_order != 0)
        if non_ties.any():
            values.append(
                (left_order[non_ties] != right_order[non_ties]).to(torch.float64).mean()
            )
    return float(torch.stack(values).mean()) if values else 0.0


def _token_mismatch_values(
    reference: ContinuationEvaluation, candidate: ContinuationEvaluation
) -> list[float]:
    if reference.seeds != candidate.seeds:
        raise ValueError("token comparisons require matched seeds")
    return [
        float(not torch.equal(left.token_ids, right.token_ids))
        for left, right in zip(reference.outputs, candidate.outputs)
    ]


def _estimate(
    values: Iterable[float], config: dict[str, Any], seed_offset: int
) -> Estimate:
    materialized = list(values)
    if not materialized:
        return Estimate(value=0.0, ci_low=0.0, ci_high=0.0, count=0)
    return bootstrap_mean(
        materialized,
        bootstrap_samples=int(config["bootstrap_samples"]),
        seed=int(config["bootstrap_seed"]) + seed_offset,
    )


def _evaluate_candidate(
    evaluator: ContinuationEvaluator,
    record: LatentStepRecord,
    latent: Tensor,
    config: dict[str, Any],
) -> ContinuationEvaluation:
    return evaluator.evaluate(
        record.prefix_state,
        latent,
        horizon=int(config["continuation_horizon"]),
        seeds=[int(seed) for seed in config["continuation_seeds"]],
    )


def run_pilot(
    config: dict[str, Any],
    workspace_root: Path,
    *,
    trace_output: Path | None = None,
) -> dict[str, object]:
    import transformers

    started = time.perf_counter()
    torch.set_num_threads(min(16, torch.get_num_threads()))
    bundle = load_public_coconut(
        checkpoint_path=workspace_root / str(config["checkpoint_relative_path"]),
        coconut_source_directory=workspace_root
        / str(config["coconut_source_relative_path"]),
        model_id=str(config["model_id"]),
        device="cpu",
        cache_directory=workspace_root / str(config["model_cache_relative_path"]),
    )
    examples = _load_examples(
        workspace_root / str(config["dataset_relative_path"]),
        int(config["sample_count"]),
        int(config["sample_selection_seed"]),
    )
    dimension = int(bundle.model.embedding.weight.shape[-1])
    charts = _build_charts(config, dimension)
    identity = charts["identity"]
    identity_runner = CoconutChartRunner(bundle.model, identity)

    records: list[LatentStepRecord] = []
    inputs: list[tuple[Tensor, Tensor]] = []
    complete_trace = LatentTrace()
    collection_forward_calls = 0
    for order, example in enumerate(examples):
        input_ids, attention_mask = prepare_coconut_input(
            bundle, example["query"], num_latents=int(config["num_latents"])
        )
        inputs.append((input_ids.detach().cpu(), attention_mask.detach().cpu()))
        sample_id = f"gsm8k-{example['dataset_index']}"
        result = identity_runner.run(
            input_ids,
            attention_mask,
            seed=int(config["collection_seed"]) + order,
            capture_trace=True,
            sample_id=sample_id,
        )
        collection_forward_calls += result.model_forward_calls
        for trace_record in result.trace.records:
            complete_trace.append(trace_record)
        records.append(result.trace.get(sample_id, int(config["target_pass"])))
    if trace_output is not None:
        complete_trace.save(trace_output)

    noop_embedding_errors: dict[str, float] = {}
    noop_logit_errors: dict[str, float] = {}
    noop_bitwise_recovery: dict[str, bool] = {}
    noop_forward_calls = 0
    for chart_name, chart in charts.items():
        if chart_name == "identity":
            continue
        runner = CoconutChartRunner(bundle.model, chart)
        maximum_embedding_error = 0.0
        maximum_logit_error = 0.0
        all_latents_bitwise_equal = True
        for input_ids_cpu, attention_mask_cpu in inputs[: int(config["noop_prompt_count"])]:
            input_ids = input_ids_cpu.to(bundle.device)
            attention_mask = attention_mask_cpu.to(bundle.device)
            baseline = identity_runner.run(
                input_ids, attention_mask, seed=int(config["collection_seed"])
            )
            charted = runner.run(
                input_ids, attention_mask, seed=int(config["collection_seed"])
            )
            noop_forward_calls += baseline.model_forward_calls + charted.model_forward_calls
            maximum_embedding_error = max(
                maximum_embedding_error,
                float(
                    (baseline.filled_inputs_embeds - charted.filled_inputs_embeds)
                    .abs()
                    .max()
                ),
            )
            maximum_logit_error = max(
                maximum_logit_error,
                float((baseline.logits - charted.logits).abs().max()),
            )
            all_latents_bitwise_equal = all_latents_bitwise_equal and all(
                torch.equal(step.proposed_native, step.consumed_native)
                for step in charted.latent_steps
            )
        noop_embedding_errors[chart_name] = maximum_embedding_error
        noop_logit_errors[chart_name] = maximum_logit_error
        noop_bitwise_recovery[chart_name] = all_latents_bitwise_equal

    latent_matrix = torch.stack([record.latent for record in records]).to(torch.float64)
    identity_euclidean = _distance_matrix(latent_matrix, identity, "euclidean")
    identity_cosine = _distance_matrix(latent_matrix, identity, "cosine")
    identity_neighbors = torch.argmin(identity_euclidean, dim=1)
    identity_cosine_neighbors = torch.argmin(identity_cosine, dim=1)
    neighbor_audit: dict[str, dict[str, object]] = {}
    condition_flip_events: dict[str, list[float]] = {
        f"{float(condition):g}": [] for condition in config["affine_condition_numbers"]
    }
    orthogonal_flip_events: list[float] = []
    chart_neighbor_indices: dict[str, Tensor] = {}
    for chart_name, chart in charts.items():
        if chart_name == "identity":
            continue
        euclidean = _distance_matrix(latent_matrix, chart, "euclidean")
        cosine = _distance_matrix(latent_matrix, chart, "cosine")
        euclidean_neighbors = torch.argmin(euclidean, dim=1)
        cosine_neighbors = torch.argmin(cosine, dim=1)
        chart_neighbor_indices[chart_name] = euclidean_neighbors
        euclidean_flips = (euclidean_neighbors != identity_neighbors).to(torch.float64)
        cosine_flips = (cosine_neighbors != identity_cosine_neighbors).to(torch.float64)
        neighbor_audit[chart_name] = {
            "euclidean_neighbor_flip_rate": float(euclidean_flips.mean()),
            "cosine_neighbor_flip_rate": float(cosine_flips.mean()),
            "euclidean_ranking_disagreement": _mean_ranking_disagreement(
                identity_euclidean, euclidean
            ),
            "cosine_ranking_disagreement": _mean_ranking_disagreement(
                identity_cosine, cosine
            ),
        }
        if chart_name.startswith("orthogonal-"):
            orthogonal_flip_events.extend(euclidean_flips.tolist())
        if chart_name.startswith("affine-cond-"):
            condition = chart_name.split("-")[2]
            condition_flip_events[condition].extend(euclidean_flips.tolist())

    condition_flip_estimates = {
        condition: _estimate(values, config, 10 + index).to_dict()
        for index, (condition, values) in enumerate(condition_flip_events.items())
    }
    orthogonal_flip_estimate = _estimate(orthogonal_flip_events, config, 20)

    behavior_affine_name = (
        f"affine-cond-{float(config['behavior_affine_condition_number']):g}"
        f"-seed-{int(config['behavior_chart_seed'])}"
    )
    behavior_orthogonal_name = f"orthogonal-seed-{int(config['behavior_chart_seed'])}"
    behavior_nonlinear_name = f"sinh-scale-{float(config['behavior_nonlinear_scale']):g}"
    behavior_affine = charts[behavior_affine_name]
    behavior_orthogonal = charts[behavior_orthogonal_name]
    behavior_nonlinear = charts[behavior_nonlinear_name]
    behavior_neighbors = chart_neighbor_indices[behavior_affine_name]
    flipped_indices = [
        index
        for index in range(len(records))
        if int(identity_neighbors[index]) != int(behavior_neighbors[index])
    ]
    audited_indices = flipped_indices[: int(config["audit_query_count"])]

    evaluator = ContinuationEvaluator(
        CoconutContinuationAdapter(
            identity_runner,
            temperature=float(config["continuation_temperature"]),
        )
    )
    neighbor_direct_divergences: list[float] = []
    neighbor_token_mismatches: list[float] = []
    neighbor_effect_gaps: list[float] = []
    identity_noise_effects: list[float] = []
    orthogonal_noise_effects: list[float] = []
    affine_noise_effects: list[float] = []
    norm_matched_affine_noise_effects: list[float] = []
    norm_matched_noise_direct_divergences: list[float] = []
    noise_token_mismatches: list[float] = []
    nonlinear_interpolation_direct_divergences: list[float] = []
    interpolation_token_mismatches: list[float] = []
    affine_interpolation_native_errors: list[float] = []
    per_query_results: list[dict[str, object]] = []

    for audit_order, query_index in enumerate(audited_indices):
        record = records[query_index]
        factual = _evaluate_candidate(evaluator, record, record.latent, config)

        identity_peer_index = int(identity_neighbors[query_index])
        affine_peer_index = int(behavior_neighbors[query_index])
        identity_peer = records[identity_peer_index].latent
        affine_peer = records[affine_peer_index].latent
        identity_peer_evaluation = _evaluate_candidate(
            evaluator, record, identity_peer, config
        )
        affine_peer_evaluation = _evaluate_candidate(
            evaluator, record, affine_peer, config
        )
        identity_peer_effect = evaluator.compare_evaluations(
            factual, identity_peer_evaluation
        )
        affine_peer_effect = evaluator.compare_evaluations(
            factual, affine_peer_evaluation
        )
        neighbor_direct = evaluator.compare_evaluations(
            identity_peer_evaluation, affine_peer_evaluation
        )
        neighbor_direct_divergences.append(neighbor_direct.mean_divergence)
        neighbor_effect_gaps.append(
            abs(affine_peer_effect.mean_divergence - identity_peer_effect.mean_divergence)
        )
        neighbor_token_mismatches.extend(
            _token_mismatch_values(identity_peer_evaluation, affine_peer_evaluation)
        )

        noise_seed = int(config["collection_seed"]) + 100_000 + query_index
        identity_noise = add_isotropic_noise_in_chart(
            record.latent,
            identity,
            standard_deviation=float(config["noise_standard_deviation"]),
            generator=torch.Generator(device="cpu").manual_seed(noise_seed),
        )
        orthogonal_noise = add_isotropic_noise_in_chart(
            record.latent,
            behavior_orthogonal,
            standard_deviation=float(config["noise_standard_deviation"]),
            generator=torch.Generator(device="cpu").manual_seed(noise_seed),
        )
        affine_noise = add_isotropic_noise_in_chart(
            record.latent,
            behavior_affine,
            standard_deviation=float(config["noise_standard_deviation"]),
            generator=torch.Generator(device="cpu").manual_seed(noise_seed),
        )
        identity_delta = identity_noise - record.latent
        affine_delta = affine_noise - record.latent
        norm_matched_affine_noise = record.latent + affine_delta * (
            torch.linalg.vector_norm(identity_delta)
            / torch.linalg.vector_norm(affine_delta).clamp_min(1e-12)
        )
        identity_noise_evaluation = _evaluate_candidate(
            evaluator, record, identity_noise, config
        )
        orthogonal_noise_evaluation = _evaluate_candidate(
            evaluator, record, orthogonal_noise, config
        )
        affine_noise_evaluation = _evaluate_candidate(
            evaluator, record, affine_noise, config
        )
        norm_matched_noise_evaluation = _evaluate_candidate(
            evaluator, record, norm_matched_affine_noise, config
        )
        identity_noise_effect = evaluator.compare_evaluations(
            factual, identity_noise_evaluation
        )
        orthogonal_noise_effect = evaluator.compare_evaluations(
            factual, orthogonal_noise_evaluation
        )
        affine_noise_effect = evaluator.compare_evaluations(
            factual, affine_noise_evaluation
        )
        norm_matched_noise_effect = evaluator.compare_evaluations(
            factual, norm_matched_noise_evaluation
        )
        norm_matched_noise_direct = evaluator.compare_evaluations(
            identity_noise_evaluation, norm_matched_noise_evaluation
        )
        identity_noise_effects.append(identity_noise_effect.mean_divergence)
        orthogonal_noise_effects.append(orthogonal_noise_effect.mean_divergence)
        affine_noise_effects.append(affine_noise_effect.mean_divergence)
        norm_matched_affine_noise_effects.append(
            norm_matched_noise_effect.mean_divergence
        )
        norm_matched_noise_direct_divergences.append(
            norm_matched_noise_direct.mean_divergence
        )
        noise_token_mismatches.extend(
            _token_mismatch_values(
                identity_noise_evaluation, norm_matched_noise_evaluation
            )
        )

        identity_midpoint = interpolate_in_chart(
            record.latent,
            identity_peer,
            identity,
            alpha=float(config["interpolation_alpha"]),
        )
        affine_midpoint = interpolate_in_chart(
            record.latent,
            identity_peer,
            behavior_affine,
            alpha=float(config["interpolation_alpha"]),
        )
        nonlinear_midpoint = interpolate_in_chart(
            record.latent,
            identity_peer,
            behavior_nonlinear,
            alpha=float(config["interpolation_alpha"]),
        )
        affine_interpolation_native_errors.append(
            float(torch.linalg.vector_norm(identity_midpoint - affine_midpoint))
        )
        identity_midpoint_evaluation = _evaluate_candidate(
            evaluator, record, identity_midpoint, config
        )
        nonlinear_midpoint_evaluation = _evaluate_candidate(
            evaluator, record, nonlinear_midpoint, config
        )
        interpolation_direct = evaluator.compare_evaluations(
            identity_midpoint_evaluation, nonlinear_midpoint_evaluation
        )
        nonlinear_interpolation_direct_divergences.append(
            interpolation_direct.mean_divergence
        )
        interpolation_token_mismatches.extend(
            _token_mismatch_values(
                identity_midpoint_evaluation, nonlinear_midpoint_evaluation
            )
        )

        per_query_results.append(
            {
                "audit_order": audit_order,
                "dataset_index": examples[query_index]["dataset_index"],
                "identity_neighbor_dataset_index": examples[identity_peer_index][
                    "dataset_index"
                ],
                "affine_neighbor_dataset_index": examples[affine_peer_index][
                    "dataset_index"
                ],
                "identity_neighbor_effect": identity_peer_effect.mean_divergence,
                "affine_neighbor_effect": affine_peer_effect.mean_divergence,
                "neighbor_candidate_direct_divergence": neighbor_direct.mean_divergence,
                "identity_noise_effect": identity_noise_effect.mean_divergence,
                "orthogonal_noise_effect": orthogonal_noise_effect.mean_divergence,
                "affine_noise_effect": affine_noise_effect.mean_divergence,
                "norm_matched_affine_noise_effect": norm_matched_noise_effect.mean_divergence,
                "norm_matched_noise_direct_divergence": norm_matched_noise_direct.mean_divergence,
                "nonlinear_interpolation_direct_divergence": interpolation_direct.mean_divergence,
                "affine_interpolation_native_error": affine_interpolation_native_errors[-1],
            }
        )

    behavior_estimates = {
        "neighbor_candidate_direct_divergence": _estimate(
            neighbor_direct_divergences, config, 30
        ).to_dict(),
        "neighbor_candidate_token_mismatch_rate": _estimate(
            neighbor_token_mismatches, config, 31
        ).to_dict(),
        "neighbor_effect_absolute_gap": _estimate(
            neighbor_effect_gaps, config, 32
        ).to_dict(),
        "identity_noise_effect": _estimate(
            identity_noise_effects, config, 33
        ).to_dict(),
        "orthogonal_noise_effect": _estimate(
            orthogonal_noise_effects, config, 34
        ).to_dict(),
        "affine_noise_effect": _estimate(affine_noise_effects, config, 35).to_dict(),
        "norm_matched_affine_noise_effect": _estimate(
            norm_matched_affine_noise_effects, config, 36
        ).to_dict(),
        "norm_matched_noise_direct_divergence": _estimate(
            norm_matched_noise_direct_divergences, config, 37
        ).to_dict(),
        "norm_matched_noise_token_mismatch_rate": _estimate(
            noise_token_mismatches, config, 38
        ).to_dict(),
        "nonlinear_interpolation_direct_divergence": _estimate(
            nonlinear_interpolation_direct_divergences, config, 39
        ).to_dict(),
        "nonlinear_interpolation_token_mismatch_rate": _estimate(
            interpolation_token_mismatches, config, 40
        ).to_dict(),
    }

    thresholds = config["thresholds"]
    condition_4 = condition_flip_estimates["4"]
    condition_12 = condition_flip_estimates["12"]
    neighbor_divergence = behavior_estimates[
        "neighbor_candidate_direct_divergence"
    ]
    neighbor_token_rate = behavior_estimates[
        "neighbor_candidate_token_mismatch_rate"
    ]
    noise_divergence = behavior_estimates[
        "norm_matched_noise_direct_divergence"
    ]
    interpolation_divergence = behavior_estimates[
        "nonlinear_interpolation_direct_divergence"
    ]
    max_noop_embedding_error = max(noop_embedding_errors.values())
    max_noop_logit_error = max(noop_logit_errors.values())
    max_affine_interpolation_error = max(
        affine_interpolation_native_errors, default=float("inf")
    )
    checks = {
        "noop_embedding_exactness": max_noop_embedding_error
        <= float(thresholds["max_noop_embedding_error"]),
        "noop_logit_exactness": max_noop_logit_error
        <= float(thresholds["max_noop_logit_error"]),
        "noop_latent_bitwise_recovery": all(noop_bitwise_recovery.values()),
        "orthogonal_neighbor_negative_control": orthogonal_flip_estimate.value
        <= float(thresholds["max_orthogonal_euclidean_neighbor_flip_rate"]),
        "affine_interpolation_negative_control": max_affine_interpolation_error
        <= float(thresholds["max_affine_interpolation_native_error"]),
        "moderate_affine_neighbor_dependence": condition_4["ci_low"]
        >= float(thresholds["min_condition_4_neighbor_flip_rate"]),
        "strong_affine_neighbor_dependence": condition_12["ci_low"]
        >= float(thresholds["min_condition_12_neighbor_flip_rate"]),
        "behavioral_audit_sample_size": len(audited_indices)
        >= int(thresholds["min_behaviorally_audited_flips"]),
        "neighbor_behavioral_consequence": (
            neighbor_divergence["ci_low"]
            >= float(thresholds["min_neighbor_candidate_direct_divergence"])
            or neighbor_token_rate["ci_low"]
            >= float(thresholds["min_neighbor_candidate_token_mismatch_rate"])
        ),
        "norm_matched_noise_behavioral_consequence": noise_divergence["ci_low"]
        >= float(thresholds["min_norm_matched_noise_direct_divergence"]),
        "nonlinear_interpolation_behavioral_consequence": interpolation_divergence[
            "ci_low"
        ]
        >= float(thresholds["min_nonlinear_interpolation_direct_divergence"]),
    }
    instrument_checks = [
        "noop_embedding_exactness",
        "noop_logit_exactness",
        "noop_latent_bitwise_recovery",
        "orthogonal_neighbor_negative_control",
        "affine_interpolation_negative_control",
    ]
    phenomenon_checks = [
        "moderate_affine_neighbor_dependence",
        "strong_affine_neighbor_dependence",
        "behavioral_audit_sample_size",
        "neighbor_behavioral_consequence",
        "norm_matched_noise_behavioral_consequence",
        "nonlinear_interpolation_behavioral_consequence",
    ]
    instrument_pass = all(checks[name] for name in instrument_checks)
    phenomenon_pass = all(checks[name] for name in phenomenon_checks)
    evaluator_compute = evaluator.ledger.snapshot()
    elapsed = time.perf_counter() - started

    return {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "status": "pass" if instrument_pass and phenomenon_pass else "fail",
        "instrument_status": "pass" if instrument_pass else "fail",
        "phenomenon_status": "pass" if phenomenon_pass else "fail",
        "evidence_level": "single-checkpoint exploratory pilot",
        "decision": (
            "proceed-to-minimum-fctr"
            if instrument_pass and phenomenon_pass
            else "hold-and-diagnose-before-fctr"
        ),
        "interpretation_limits": [
            "The checkpoint is a trained GPT-2 Coconut model on GSM8K, not the current repository's random projection model.",
            "Nearest-neighbor replacement is an operational diagnostic, not a support-valid causal intervention.",
            "The pilot covers one checkpoint, one dataset, one target latent step, and CPU inference only.",
            "Passing this gate motivates a second architecture and training experiment; it is not a paper-level result.",
        ],
        "model": {
            "model_id": bundle.model_id,
            "checkpoint_repo": config["checkpoint_repo"],
            "checkpoint_file": config["checkpoint_file"],
            "checkpoint_sha256": bundle.checkpoint_sha256,
            "coconut_source_commit": bundle.coconut_source_commit,
            "hidden_dimension": dimension,
        },
        "dataset": {
            "path": config["dataset_relative_path"],
            "sample_count": len(examples),
            "selected_dataset_indices": [item["dataset_index"] for item in examples],
            "target_pass": config["target_pass"],
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
        },
        "config": {
            key: config[key]
            for key in (
                "sample_selection_seed",
                "num_latents",
                "collection_seed",
                "noop_prompt_count",
                "chart_seeds",
                "affine_condition_numbers",
                "nonlinear_scales",
                "behavior_chart_seed",
                "behavior_affine_condition_number",
                "behavior_nonlinear_scale",
                "audit_query_count",
                "continuation_horizon",
                "continuation_seeds",
                "continuation_temperature",
                "noise_standard_deviation",
                "interpolation_alpha",
                "bootstrap_samples",
                "bootstrap_seed",
            )
        },
        "thresholds": thresholds,
        "checks": checks,
        "noop": {
            "max_embedding_error": max_noop_embedding_error,
            "max_logit_error": max_noop_logit_error,
            "embedding_errors_by_chart": noop_embedding_errors,
            "logit_errors_by_chart": noop_logit_errors,
            "bitwise_latent_recovery_by_chart": noop_bitwise_recovery,
        },
        "neighbor_geometry": {
            "orthogonal_euclidean_flip_rate": orthogonal_flip_estimate.to_dict(),
            "affine_flip_rate_by_condition": condition_flip_estimates,
            "by_chart": neighbor_audit,
            "behavior_affine_chart": behavior_affine_name,
            "available_flipped_queries": len(flipped_indices),
            "audited_flipped_queries": len(audited_indices),
        },
        "behavioral_consequences": behavior_estimates,
        "affine_interpolation_max_native_error": max_affine_interpolation_error,
        "per_query_results": per_query_results,
        "compute": {
            "collection_model_forward_calls": collection_forward_calls,
            "noop_model_forward_calls": noop_forward_calls,
            "continuation_model_forward_calls": evaluator_compute.model_forward_calls,
            "continuation_rollout_calls": evaluator_compute.rollout_calls,
            "continuation_generated_tokens": evaluator_compute.generated_tokens,
            "continuation_wall_time_seconds": evaluator_compute.wall_time_seconds,
            "total_wall_time_seconds": elapsed,
        },
    }


def main() -> int:
    default_config = Path(__file__).with_name("configs") / "public_gpt2_coconut_pilot_v1.json"
    parser = argparse.ArgumentParser(
        description="Run the preregistered GPT-2 Coconut coordinate pilot"
    )
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--workspace-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace-output", type=Path)
    arguments = parser.parse_args()

    result = run_pilot(
        _load_config(arguments.config),
        arguments.workspace_root.resolve(),
        trace_output=arguments.trace_output,
    )
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
