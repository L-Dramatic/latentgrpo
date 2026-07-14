from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from research.coordinate_invariance.charts import AffineChart
from research.coordinate_invariance.real_models.coconut import (
    CoconutChartRunner,
    CoconutContinuationAdapter,
    load_public_coconut,
    prepare_coconut_input,
)
from research.coordinate_invariance.trace import LatentStepRecord, LatentTrace

from .analysis import continuation_rank_report
from .joint_kl import JointContinuationEvaluator


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("unsupported BCG pilot config")
    horizons = [int(value) for value in config["reported_horizons"]]
    if sorted(set(horizons)) != horizons or horizons[0] != 1:
        raise ValueError("reported_horizons must be sorted, unique, and begin at one")
    if horizons[-1] != int(config["continuation_horizon"]):
        raise ValueError("the final reported horizon must equal continuation_horizon")
    prefix_horizons = [
        int(value) for value in config.get("prefix_predictor_horizons", [1])
    ]
    if (
        sorted(set(prefix_horizons)) != prefix_horizons
        or not prefix_horizons
        or prefix_horizons[0] != 1
        or prefix_horizons[-1] > int(config["continuation_horizon"])
    ):
        raise ValueError(
            "prefix_predictor_horizons must be sorted, unique, begin at one, "
            "and fit within continuation_horizon"
        )
    return config


def _load_examples(path: Path, count: int, seed: int) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) < count:
        raise ValueError("dataset does not contain the requested number of examples")
    indices = list(range(len(raw)))
    random.Random(seed).shuffle(indices)
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


def _candidate_coordinates(
    reference: Tensor,
    candidate: Tensor,
    diagonal_variance: Tensor,
) -> dict[str, float]:
    reference64 = reference.to(torch.float64)
    candidate64 = candidate.to(torch.float64)
    delta = candidate64 - reference64
    denominator = torch.linalg.vector_norm(reference64) * torch.linalg.vector_norm(
        candidate64
    )
    cosine_distance = (
        0.0
        if denominator <= 0
        else float(1.0 - (reference64 @ candidate64) / denominator)
    )
    return {
        "euclidean": float(torch.linalg.vector_norm(delta)),
        "cosine": cosine_distance,
        "diagonal_mahalanobis": float(
            torch.sqrt(torch.sum(delta.square() / diagonal_variance))
        ),
    }


def _quantile_summary(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    quantiles = torch.quantile(
        tensor, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
    )
    return {
        "min": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "max": float(quantiles[4]),
        "mean": float(tensor.mean()),
    }


def _hidden_candidate_ids(
    rows: list[dict[str, Any]], top_fraction: float, screen_fraction: float
) -> list[str]:
    count = len(rows)
    top_count = max(1, int(math.ceil(count * top_fraction)))
    screen_count = max(1, int(math.ceil(count * screen_fraction)))
    total_order = sorted(
        range(count), key=lambda index: (-rows[index]["total_kl"], index)
    )
    first_order = sorted(
        range(count), key=lambda index: (-rows[index]["first_step_kl"], index)
    )
    top_risk = set(total_order[:top_count])
    screened = set(first_order[:screen_count])
    return [rows[index]["candidate_id"] for index in sorted(top_risk - screened)]


def _rank_report(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, dict[str, float | int]]:
    return continuation_rank_report(
        rows,
        prefix_horizons=config.get("prefix_predictor_horizons", [1]),
        top_fraction=float(config["ranking_top_fraction"]),
        predictor_screen_fraction=float(config["ranking_screen_fraction"]),
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
    identity = AffineChart.identity(
        dimension, dtype=torch.float64, compute_dtype=torch.float64
    )
    runner = CoconutChartRunner(bundle.model, identity)
    policy = CoconutContinuationAdapter(
        runner,
        temperature=float(config["sampling_temperature"]),
        stop_at_eos=bool(config.get("stop_at_eos", False)),
    )

    records: list[LatentStepRecord] = []
    complete_trace = LatentTrace()
    collection_forward_calls = 0
    for order, example in enumerate(examples):
        sample_id = f"gsm8k-train-{example['dataset_index']}"
        input_ids, attention_mask = prepare_coconut_input(
            bundle, example["query"], num_latents=int(config["num_latents"])
        )
        result = runner.run(
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

    latent_matrix = torch.stack([record.latent for record in records]).to(torch.float64)
    pair_distances = torch.cdist(latent_matrix, latent_matrix)
    pair_distances.fill_diagonal_(float("inf"))
    nearest_indices = pair_distances.argmin(dim=1)
    raw_variance = latent_matrix.var(dim=0, unbiased=True)
    positive_variance = raw_variance[raw_variance > 0]
    variance_scale = (
        float(positive_variance.median()) if positive_variance.numel() else 1.0
    )
    variance_floor = variance_scale * float(
        config["diagonal_variance_floor_fraction"]
    )
    diagonal_variance = raw_variance.clamp_min(variance_floor)

    rows: list[dict[str, Any]] = []
    prompt_summaries: list[dict[str, Any]] = []
    prompt_compute: list[dict[str, object]] = []
    horizon = int(config["continuation_horizon"])
    reported_horizons = [int(value) for value in config["reported_horizons"]]
    seeds_per_prompt = int(config["continuation_seeds_per_prompt"])
    for order, (example, record) in enumerate(zip(examples, records)):
        seeds = tuple(
            int(config["continuation_seed_base"]) + order * 100 + offset
            for offset in range(seeds_per_prompt)
        )
        evaluator = JointContinuationEvaluator(policy)
        reference = evaluator.sample_reference(
            record.prefix_state,
            record.latent,
            horizon=horizon,
            seeds=seeds,
        )
        prompt_summaries.append(
            {
                "sample_id": record.sample_id,
                "dataset_index": example["dataset_index"],
                "query": example["query"],
                "answer": example["answer"],
                "reference_token_ids": [
                    output.token_ids.tolist() for output in reference.outputs
                ],
                "reference_valid_steps": [
                    int(output.logits.shape[0]) for output in reference.outputs
                ],
                "reference_text": [
                    bundle.tokenizer.decode(output.token_ids.tolist())
                    for output in reference.outputs
                ],
            }
        )

        candidate_specs: list[dict[str, Any]] = []
        for variant, norm in enumerate(config["random_perturbation_norms"]):
            seed = int(config["random_direction_seed_base"]) + order * 100 + variant
            generator = torch.Generator(device="cpu").manual_seed(seed)
            direction = torch.randn(
                record.latent.shape,
                dtype=record.latent.dtype,
                generator=generator,
            )
            direction = direction / torch.linalg.vector_norm(direction)
            candidate_specs.append(
                {
                    "family": "isotropic",
                    "level": float(norm),
                    "seed": seed,
                    "peer_sample_id": None,
                    "latent": record.latent + float(norm) * direction,
                }
            )
        peer_index = int(nearest_indices[order])
        peer_record = records[peer_index]
        peer_delta = peer_record.latent - record.latent
        for fraction in config["peer_interpolation_fractions"]:
            candidate_specs.append(
                {
                    "family": "nearest_peer_interpolation",
                    "level": float(fraction),
                    "seed": None,
                    "peer_sample_id": peer_record.sample_id,
                    "latent": record.latent + float(fraction) * peer_delta,
                }
            )

        for variant, spec in enumerate(candidate_specs):
            candidate = spec.pop("latent")
            result = evaluator.directional_from_reference(
                record.prefix_state, candidate, reference
            )
            mean_steps = result.mean_step_kl
            cumulative = torch.cumsum(mean_steps, dim=0)
            horizon_values = {
                str(value): float(cumulative[value - 1])
                for value in reported_horizons
            }
            total_kl = float(cumulative[-1])
            first_step_kl = float(mean_steps[0])
            rows.append(
                {
                    "candidate_id": f"{record.sample_id}-candidate-{variant}",
                    "sample_id": record.sample_id,
                    "dataset_index": example["dataset_index"],
                    "target_pass": int(config["target_pass"]),
                    "family": spec["family"],
                    "level": spec["level"],
                    "candidate_seed": spec["seed"],
                    "peer_sample_id": spec["peer_sample_id"],
                    "coordinate_distances": _candidate_coordinates(
                        record.latent, candidate, diagonal_variance
                    ),
                    "first_step_kl": first_step_kl,
                    "tail_kl": total_kl - first_step_kl,
                    "tail_fraction": (
                        0.0 if total_kl <= 0.0 else (total_kl - first_step_kl) / total_kl
                    ),
                    "total_kl": total_kl,
                    "horizon_kl": horizon_values,
                    "mean_step_kl": mean_steps.tolist(),
                    "per_seed_step_kl": result.per_seed_step_kl.tolist(),
                    "per_seed_valid_steps": result.per_seed_valid_steps.tolist(),
                }
            )
        prompt_compute.append(evaluator.ledger.snapshot().to_dict())

    grouped_rank: dict[str, dict[str, dict[str, float | int]]] = {}
    for family in sorted({str(row["family"]) for row in rows}):
        family_rows = [row for row in rows if row["family"] == family]
        grouped_rank[family] = _rank_report(family_rows, config)
    for level in sorted({(str(row["family"]), float(row["level"])) for row in rows}):
        family, value = level
        level_rows = [
            row
            for row in rows
            if row["family"] == family and float(row["level"]) == value
        ]
        grouped_rank[f"{family}:{value:g}"] = _rank_report(level_rows, config)

    compute_totals = {
        key: sum(int(item[key]) for item in prompt_compute)
        for key in (
            "model_forward_calls",
            "rollout_calls",
            "prompt_tokens",
            "generated_tokens",
        )
    }
    compute_totals["collection_model_forward_calls"] = collection_forward_calls
    compute_totals["wall_time_seconds"] = time.perf_counter() - started
    first_values = [float(row["first_step_kl"]) for row in rows]
    total_values = [float(row["total_kl"]) for row in rows]
    tail_fractions = [float(row["tail_fraction"]) for row in rows]
    return {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "status": "complete",
        "scientific_evidence": False,
        "evidence_level": "exploratory calibration only",
        "interpretation": (
            "This train-split pilot selects perturbation regimes and freezes the "
            "held-out gate. Its rankings must not be reported as confirmation."
        ),
        "model": {
            "model_id": bundle.model_id,
            "checkpoint_repo": config["checkpoint_repo"],
            "checkpoint_file": config["checkpoint_file"],
            "checkpoint_sha256": bundle.checkpoint_sha256,
            "coconut_source_commit": bundle.coconut_source_commit,
            "hidden_dimension": dimension,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
        },
        "data": {
            "dataset_relative_path": config["dataset_relative_path"],
            "dataset_split": config["dataset_split"],
            "sample_count": len(examples),
            "sample_selection_seed": config["sample_selection_seed"],
            "selected_dataset_indices": [item["dataset_index"] for item in examples],
        },
        "config": {
            key: config[key]
            for key in (
                "num_latents",
                "target_pass",
                "collection_seed",
                "continuation_horizon",
                "reported_horizons",
                "continuation_seed_base",
                "continuation_seeds_per_prompt",
                "sampling_temperature",
                "stop_at_eos",
                "random_direction_seed_base",
                "random_perturbation_norms",
                "peer_interpolation_fractions",
                "diagonal_variance_floor_fraction",
                "ranking_top_fraction",
                "ranking_screen_fraction",
            )
        }
        | {
            "prefix_predictor_horizons": config.get(
                "prefix_predictor_horizons", [1]
            )
        },
        "latent_scale": {
            "norm": _quantile_summary(
                torch.linalg.vector_norm(latent_matrix, dim=1).tolist()
            ),
            "nearest_peer_distance": _quantile_summary(
                pair_distances.min(dim=1).values.tolist()
            ),
            "diagonal_variance_floor": variance_floor,
        },
        "metrics": {
            "candidate_count": len(rows),
            "first_step_kl": _quantile_summary(first_values),
            "total_kl": _quantile_summary(total_values),
            "tail_fraction": _quantile_summary(tail_fractions),
            "pooled_rank": _rank_report(rows, config),
            "grouped_rank": grouped_rank,
            "hidden_candidate_ids": _hidden_candidate_ids(
                rows,
                float(config["ranking_top_fraction"]),
                float(config["ranking_screen_fraction"]),
            ),
        },
        "compute": compute_totals,
        "prompts": prompt_summaries,
        "candidates": rows,
    }


def main() -> int:
    default_config = (
        Path(__file__).with_name("configs")
        / "public_gpt2_coconut_bcg_pilot_v2.json"
    )
    parser = argparse.ArgumentParser(
        description="Explore multi-horizon versus next-token BCG signals"
    )
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument(
        "--workspace-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--output", type=Path)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
