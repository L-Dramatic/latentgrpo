from __future__ import annotations

import argparse
import json
import platform
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch

from .charts import AffineChart, SinhChart
from .evaluator import ContinuationEvaluation, ContinuationEvaluator
from .real_coconut_pilot import (
    _distance_matrix,
    _mean_ranking_disagreement,
    _token_mismatch_values,
)
from .real_models.coconut import (
    CoconutChartRunner,
    CoconutContinuationAdapter,
    load_public_coconut,
    prepare_coconut_input,
)
from .statistics import bootstrap_mean
from .trace import LatentStepRecord, LatentTrace


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _select_holdout_examples(
    dataset: list[dict[str, Any]],
    excluded_indices: set[int],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    available = [index for index in range(len(dataset)) if index not in excluded_indices]
    generator = random.Random(seed)
    generator.shuffle(available)
    if len(available) < count:
        raise ValueError("not enough examples remain after excluding pilot data")
    return [
        {
            "dataset_index": index,
            "query": str(dataset[index]["query"]),
            "answer": str(dataset[index].get("answer", "")),
        }
        for index in available[:count]
    ]


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    probabilities = torch.tensor(
        [0.0, 0.05, 0.5, 0.95, 1.0], dtype=torch.float64
    )
    result = torch.quantile(values.to(torch.float64), probabilities)
    return {
        "min": float(result[0]),
        "p05": float(result[1]),
        "median": float(result[2]),
        "p95": float(result[3]),
        "max": float(result[4]),
    }


def _extract_numeric_answer(text: str) -> str | None:
    matches = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if not matches:
        return None
    value = matches[-1].replace(",", "")
    try:
        numeric = float(value)
    except ValueError:
        return value
    return str(int(numeric)) if numeric.is_integer() else str(numeric)


def _accuracy_values(
    evaluation: ContinuationEvaluation, tokenizer: Any, expected: str
) -> list[float]:
    normalized_expected = _extract_numeric_answer(expected) or expected.strip()
    return [
        float(
            (_extract_numeric_answer(tokenizer.decode(output.token_ids)) or "")
            == normalized_expected
        )
        for output in evaluation.outputs
    ]


def run_holdout(
    config: dict[str, Any],
    workspace_root: Path,
    *,
    trace_output: Path | None = None,
) -> dict[str, object]:
    import transformers

    started = time.perf_counter()
    dataset = _load_json(workspace_root / str(config["dataset_relative_path"]))
    pilot = _load_json(workspace_root / str(config["pilot_artifact_relative_path"]))
    if pilot.get("experiment_name") != "public-gpt2-coconut-coordinate-pilot-v1":
        raise ValueError("holdout exclusion artifact is not pilot v1")
    excluded_indices = set(pilot["dataset"]["selected_dataset_indices"])
    examples = _select_holdout_examples(
        dataset,
        excluded_indices,
        count=int(config["holdout_sample_count"]),
        seed=int(config["holdout_selection_seed"]),
    )
    selected_indices = {item["dataset_index"] for item in examples}
    if excluded_indices & selected_indices:
        raise AssertionError("holdout data overlap detected")

    torch.set_num_threads(min(16, torch.get_num_threads()))
    bundle = load_public_coconut(
        checkpoint_path=workspace_root / str(config["checkpoint_relative_path"]),
        coconut_source_directory=workspace_root
        / str(config["coconut_source_relative_path"]),
        model_id=str(config["model_id"]),
        device="cpu",
        cache_directory=workspace_root / str(config["model_cache_relative_path"]),
    )
    dimension = int(bundle.model.embedding.weight.shape[-1])
    identity = AffineChart.identity(
        dimension, dtype=torch.float64, compute_dtype=torch.float64
    )
    orthogonal = AffineChart.random_orthogonal(
        dimension,
        seed=int(config["orthogonal_seed"]),
        dtype=torch.float64,
        compute_dtype=torch.float64,
    )
    nonlinear = SinhChart(
        dimension,
        scale=float(config["nonlinear_scale"]),
        dtype=torch.float64,
        compute_dtype=torch.float64,
        name=f"sinh-scale-{float(config['nonlinear_scale']):g}",
    )
    identity_runner = CoconutChartRunner(bundle.model, identity)
    nonlinear_runner = CoconutChartRunner(bundle.model, nonlinear)

    records: list[LatentStepRecord] = []
    inputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    complete_trace = LatentTrace()
    collection_forward_calls = 0
    for order, example in enumerate(examples):
        input_ids, attention_mask = prepare_coconut_input(
            bundle, example["query"], num_latents=int(config["num_latents"])
        )
        inputs.append((input_ids.detach().cpu(), attention_mask.detach().cpu()))
        sample_id = f"gsm8k-holdout-{example['dataset_index']}"
        result = identity_runner.run(
            input_ids,
            attention_mask,
            seed=int(config["collection_seed"]) + order,
            capture_trace=True,
            sample_id=sample_id,
        )
        collection_forward_calls += result.model_forward_calls
        for record in result.trace.records:
            complete_trace.append(record)
        records.append(result.trace.get(sample_id, int(config["target_pass"])))
    if trace_output is not None:
        complete_trace.save(trace_output)

    noop_embedding_error = 0.0
    noop_logit_error = 0.0
    noop_bitwise_recovery = True
    noop_forward_calls = 0
    for input_ids_cpu, attention_mask_cpu in inputs[: int(config["noop_prompt_count"])]:
        input_ids = input_ids_cpu.to(bundle.device)
        attention_mask = attention_mask_cpu.to(bundle.device)
        baseline = identity_runner.run(
            input_ids, attention_mask, seed=int(config["collection_seed"])
        )
        transformed = nonlinear_runner.run(
            input_ids, attention_mask, seed=int(config["collection_seed"])
        )
        noop_forward_calls += baseline.model_forward_calls + transformed.model_forward_calls
        noop_embedding_error = max(
            noop_embedding_error,
            float(
                (baseline.filled_inputs_embeds - transformed.filled_inputs_embeds)
                .abs()
                .max()
            ),
        )
        noop_logit_error = max(
            noop_logit_error,
            float((baseline.logits - transformed.logits).abs().max()),
        )
        noop_bitwise_recovery = noop_bitwise_recovery and all(
            torch.equal(step.proposed_native, step.consumed_native)
            for step in transformed.latent_steps
        )

    latents = torch.stack([record.latent for record in records]).to(torch.float64)
    identity_distances = _distance_matrix(latents, identity, "euclidean")
    orthogonal_distances = _distance_matrix(latents, orthogonal, "euclidean")
    nonlinear_distances = _distance_matrix(latents, nonlinear, "euclidean")
    identity_neighbors = torch.argmin(identity_distances, dim=1)
    orthogonal_neighbors = torch.argmin(orthogonal_distances, dim=1)
    nonlinear_neighbors = torch.argmin(nonlinear_distances, dim=1)
    orthogonal_flips = (orthogonal_neighbors != identity_neighbors).to(torch.float64)
    nonlinear_flips = (nonlinear_neighbors != identity_neighbors).to(torch.float64)
    orthogonal_flip_estimate = bootstrap_mean(
        orthogonal_flips,
        bootstrap_samples=int(config["bootstrap_samples"]),
        seed=int(config["bootstrap_seed"]),
    )
    nonlinear_flip_estimate = bootstrap_mean(
        nonlinear_flips,
        bootstrap_samples=int(config["bootstrap_samples"]),
        seed=int(config["bootstrap_seed"]) + 1,
    )
    local_conditions = nonlinear.local_condition_number(latents)
    local_condition_summary = _quantiles(local_conditions)
    distance_ratios = torch.pdist(nonlinear.encode(latents)) / torch.pdist(latents)
    distance_ratio_summary = _quantiles(distance_ratios)

    flipped_indices = [
        index for index in range(len(records)) if bool(nonlinear_flips[index])
    ]
    audited_indices = flipped_indices[: int(config["audit_query_count"])]
    evaluator = ContinuationEvaluator(
        CoconutContinuationAdapter(
            identity_runner,
            temperature=float(config["continuation_temperature"]),
        )
    )
    direct_divergences: list[float] = []
    token_mismatches: list[float] = []
    effect_gaps: list[float] = []
    identity_neighbor_accuracy: list[float] = []
    nonlinear_neighbor_accuracy: list[float] = []
    per_query: list[dict[str, object]] = []
    seeds = [int(seed) for seed in config["continuation_seeds"]]
    horizon = int(config["continuation_horizon"])
    for audit_order, query_index in enumerate(audited_indices):
        record = records[query_index]
        factual = evaluator.evaluate(
            record.prefix_state, record.latent, horizon=horizon, seeds=seeds
        )
        identity_peer_index = int(identity_neighbors[query_index])
        nonlinear_peer_index = int(nonlinear_neighbors[query_index])
        identity_evaluation = evaluator.evaluate(
            record.prefix_state,
            records[identity_peer_index].latent,
            horizon=horizon,
            seeds=seeds,
        )
        nonlinear_evaluation = evaluator.evaluate(
            record.prefix_state,
            records[nonlinear_peer_index].latent,
            horizon=horizon,
            seeds=seeds,
        )
        identity_effect = evaluator.compare_evaluations(factual, identity_evaluation)
        nonlinear_effect = evaluator.compare_evaluations(factual, nonlinear_evaluation)
        direct = evaluator.compare_evaluations(
            identity_evaluation, nonlinear_evaluation
        )
        direct_divergences.append(direct.mean_divergence)
        token_mismatches.extend(
            _token_mismatch_values(identity_evaluation, nonlinear_evaluation)
        )
        effect_gaps.append(
            abs(nonlinear_effect.mean_divergence - identity_effect.mean_divergence)
        )
        expected = examples[query_index]["answer"]
        identity_accuracy_values = _accuracy_values(
            identity_evaluation, bundle.tokenizer, expected
        )
        nonlinear_accuracy_values = _accuracy_values(
            nonlinear_evaluation, bundle.tokenizer, expected
        )
        identity_neighbor_accuracy.extend(identity_accuracy_values)
        nonlinear_neighbor_accuracy.extend(nonlinear_accuracy_values)
        per_query.append(
            {
                "audit_order": audit_order,
                "dataset_index": examples[query_index]["dataset_index"],
                "identity_neighbor_dataset_index": examples[identity_peer_index][
                    "dataset_index"
                ],
                "nonlinear_neighbor_dataset_index": examples[nonlinear_peer_index][
                    "dataset_index"
                ],
                "identity_neighbor_effect": identity_effect.mean_divergence,
                "nonlinear_neighbor_effect": nonlinear_effect.mean_divergence,
                "direct_divergence": direct.mean_divergence,
                "identity_neighbor_accuracy": sum(identity_accuracy_values)
                / len(identity_accuracy_values),
                "nonlinear_neighbor_accuracy": sum(nonlinear_accuracy_values)
                / len(nonlinear_accuracy_values),
            }
        )

    def estimate(values: list[float], offset: int) -> dict[str, float | int]:
        if not values:
            return {"value": 0.0, "ci_low": 0.0, "ci_high": 0.0, "count": 0}
        return bootstrap_mean(
            values,
            bootstrap_samples=int(config["bootstrap_samples"]),
            seed=int(config["bootstrap_seed"]) + offset,
        ).to_dict()

    behavior = {
        "neighbor_direct_divergence": estimate(direct_divergences, 2),
        "neighbor_token_mismatch_rate": estimate(token_mismatches, 3),
        "neighbor_effect_absolute_gap": estimate(effect_gaps, 4),
        "identity_neighbor_accuracy": estimate(identity_neighbor_accuracy, 5),
        "nonlinear_neighbor_accuracy": estimate(nonlinear_neighbor_accuracy, 6),
    }
    thresholds = config["thresholds"]
    checks = {
        "holdout_disjointness": not bool(excluded_indices & selected_indices),
        "noop_embedding_exactness": noop_embedding_error
        <= float(thresholds["max_noop_embedding_error"]),
        "noop_logit_exactness": noop_logit_error
        <= float(thresholds["max_noop_logit_error"]),
        "noop_latent_bitwise_recovery": noop_bitwise_recovery,
        "orthogonal_neighbor_negative_control": orthogonal_flip_estimate.value
        <= float(thresholds["max_orthogonal_neighbor_flip_rate"]),
        "moderate_local_condition_median": local_condition_summary["median"]
        <= float(thresholds["max_local_condition_median"]),
        "moderate_local_condition_p95": local_condition_summary["p95"]
        <= float(thresholds["max_local_condition_p95"]),
        "moderate_local_condition_max": local_condition_summary["max"]
        <= float(thresholds["max_local_condition_max"]),
        "heldout_neighbor_dependence": nonlinear_flip_estimate.ci_low
        >= float(thresholds["min_nonlinear_neighbor_flip_rate_ci_low"]),
        "behavioral_audit_sample_size": len(audited_indices)
        >= int(thresholds["min_behaviorally_audited_flips"]),
        "heldout_neighbor_direct_divergence": behavior[
            "neighbor_direct_divergence"
        ]["ci_low"]
        >= float(thresholds["min_neighbor_direct_divergence_ci_low"]),
        "heldout_neighbor_token_mismatch": behavior[
            "neighbor_token_mismatch_rate"
        ]["ci_low"]
        >= float(thresholds["min_neighbor_token_mismatch_rate_ci_low"]),
    }
    passed = all(checks.values())
    compute = evaluator.ledger.snapshot()
    return {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "status": "pass" if passed else "fail",
        "decision": "proceed-to-second-architecture" if passed else "stop-neighbor-evidence-line",
        "evidence_level": "heldout single-checkpoint confirmation after pilot-selected chart",
        "provenance": config["provenance"],
        "interpretation_limits": [
            "The nonlinear chart scale was selected using pilot traces, then frozen for this disjoint holdout.",
            "The result remains limited to one trained Coconut checkpoint and one dataset.",
            "Neighbor replacement is an operational consequence test, not a causal intervention claim.",
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
            "pilot_excluded_count": len(excluded_indices),
            "holdout_sample_count": len(examples),
            "selected_dataset_indices": sorted(selected_indices),
            "overlap_count": len(excluded_indices & selected_indices),
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
                "holdout_selection_seed",
                "num_latents",
                "target_pass",
                "collection_seed",
                "noop_prompt_count",
                "orthogonal_seed",
                "nonlinear_scale",
                "audit_query_count",
                "continuation_horizon",
                "continuation_seeds",
                "continuation_temperature",
                "bootstrap_samples",
                "bootstrap_seed",
            )
        },
        "thresholds": thresholds,
        "checks": checks,
        "noop": {
            "embedding_max_abs_error": noop_embedding_error,
            "logit_max_abs_error": noop_logit_error,
            "bitwise_latent_recovery": noop_bitwise_recovery,
        },
        "geometry": {
            "orthogonal_neighbor_flip_rate": orthogonal_flip_estimate.to_dict(),
            "nonlinear_neighbor_flip_rate": nonlinear_flip_estimate.to_dict(),
            "nonlinear_ranking_disagreement": _mean_ranking_disagreement(
                identity_distances, nonlinear_distances
            ),
            "local_condition_number": local_condition_summary,
            "pairwise_distance_ratio": distance_ratio_summary,
            "available_flipped_queries": len(flipped_indices),
            "audited_flipped_queries": len(audited_indices),
        },
        "behavioral_consequences": behavior,
        "per_query_results": per_query,
        "compute": {
            "collection_model_forward_calls": collection_forward_calls,
            "noop_model_forward_calls": noop_forward_calls,
            "continuation_model_forward_calls": compute.model_forward_calls,
            "continuation_rollout_calls": compute.rollout_calls,
            "continuation_generated_tokens": compute.generated_tokens,
            "continuation_wall_time_seconds": compute.wall_time_seconds,
            "total_wall_time_seconds": time.perf_counter() - started,
        },
    }


def main() -> int:
    default_config = Path(__file__).with_name("configs") / "public_gpt2_coconut_holdout_v1.json"
    parser = argparse.ArgumentParser(
        description="Run the disjoint GPT-2 Coconut nonlinear-chart holdout"
    )
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--workspace-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace-output", type=Path)
    arguments = parser.parse_args()
    config = _load_json(arguments.config)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("unsupported holdout config")
    result = run_holdout(
        config,
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

