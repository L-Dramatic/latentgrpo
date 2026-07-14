from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch

from .charts import AffineChart
from .evaluator import ContinuationEvaluator
from .real_models.coconut import (
    CoconutChartRunner,
    CoconutContinuationAdapter,
    load_public_coconut,
    prepare_coconut_input,
)


def _load_config(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("unsupported real-model smoke config")
    return raw


def run_smoke(config: dict[str, object], workspace_root: Path) -> dict[str, object]:
    import transformers

    torch.set_num_threads(min(16, torch.get_num_threads()))
    checkpoint_path = workspace_root / str(config["checkpoint_relative_path"])
    source_path = workspace_root / str(config["coconut_source_relative_path"])
    cache_path = workspace_root / str(config["model_cache_relative_path"])
    bundle = load_public_coconut(
        checkpoint_path=checkpoint_path,
        coconut_source_directory=source_path,
        model_id=str(config["model_id"]),
        device="cpu",
        cache_directory=cache_path,
    )
    input_ids, attention_mask = prepare_coconut_input(
        bundle,
        str(config["question"]),
        num_latents=int(config["num_latents"]),
    )
    position_ids = torch.arange(
        input_ids.shape[1], dtype=torch.long, device=input_ids.device
    ).reshape(1, -1)
    with torch.no_grad():
        official = bundle.model.forward(
            input_ids,
            attention_mask,
            input_ids.clone(),
            position_ids,
        )

    dimension = int(bundle.model.embedding.weight.shape[-1])
    identity = AffineChart.identity(
        dimension, dtype=torch.float64, compute_dtype=torch.float64
    )
    affine = AffineChart.with_condition_number(
        dimension,
        float(config["affine_condition_number"]),
        seed=int(config["affine_seed"]),
        dtype=torch.float64,
        compute_dtype=torch.float64,
    )
    identity_runner = CoconutChartRunner(bundle.model, identity)
    affine_runner = CoconutChartRunner(bundle.model, affine)
    identity_result = identity_runner.run(
        input_ids,
        attention_mask,
        seed=int(config["seed"]),
        capture_trace=True,
        sample_id="smoke-question",
    )
    affine_result = affine_runner.run(
        input_ids,
        attention_mask,
        seed=int(config["seed"]),
    )

    identity_embedding_error = float(
        (identity_result.filled_inputs_embeds - official.inputs_embeds).abs().max()
    )
    identity_logit_error = float((identity_result.logits - official.logits).abs().max())
    affine_embedding_error = float(
        (affine_result.filled_inputs_embeds - official.inputs_embeds).abs().max()
    )
    affine_logit_error = float((affine_result.logits - official.logits).abs().max())
    affine_bitwise_latent_recovery = all(
        torch.equal(step.proposed_native, step.consumed_native)
        for step in affine_result.latent_steps
    )

    target_pass = int(config["num_latents"]) // 2
    record = identity_result.trace.get("smoke-question", target_pass)
    evaluator = ContinuationEvaluator(
        CoconutContinuationAdapter(identity_runner, temperature=0.8)
    )
    control = evaluator.compare_record(
        record,
        record.latent.clone(),
        horizon=int(config["continuation_horizon"]),
        seeds=[int(seed) for seed in config["continuation_seeds"]],
    )
    perturbation_generator = torch.Generator(device="cpu").manual_seed(
        int(config["seed"]) + 100
    )
    perturbation = torch.randn(
        record.latent.shape,
        generator=perturbation_generator,
        dtype=record.latent.dtype,
    )
    perturbation = perturbation / torch.linalg.vector_norm(perturbation)
    candidate = record.latent + float(config["candidate_perturbation_norm"]) * perturbation
    candidate_comparison = evaluator.compare_record(
        record,
        candidate,
        horizon=int(config["continuation_horizon"]),
        seeds=[int(seed) for seed in config["continuation_seeds"]],
    )

    metrics = {
        "identity_embedding_max_abs_error": identity_embedding_error,
        "identity_logit_max_abs_error": identity_logit_error,
        "affine_embedding_max_abs_error": affine_embedding_error,
        "affine_logit_max_abs_error": affine_logit_error,
        "affine_bitwise_latent_recovery": affine_bitwise_latent_recovery,
        "trace_record_count": len(identity_result.trace),
        "control_mean_divergence": control.mean_divergence,
        "candidate_mean_divergence": candidate_comparison.mean_divergence,
        "candidate_divergence_standard_error": candidate_comparison.standard_error,
        "continuation_rollout_calls": candidate_comparison.compute.rollout_calls,
        "continuation_generated_tokens": candidate_comparison.compute.generated_tokens,
    }
    thresholds = config["thresholds"]
    checks = {
        "identity_embedding_exactness": identity_embedding_error
        <= float(thresholds["max_identity_embedding_error"]),
        "identity_logit_exactness": identity_logit_error
        <= float(thresholds["max_identity_logit_error"]),
        "affine_embedding_exactness": affine_embedding_error
        <= float(thresholds["max_affine_embedding_error"]),
        "affine_logit_exactness": affine_logit_error
        <= float(thresholds["max_affine_logit_error"]),
        "affine_latent_bitwise_recovery": affine_bitwise_latent_recovery,
        "trace_completeness": len(identity_result.trace) == int(config["num_latents"]),
        "matched_replay_control": control.mean_divergence
        <= float(thresholds["max_control_divergence"]),
        "candidate_sensitivity": candidate_comparison.mean_divergence
        >= float(thresholds["min_candidate_divergence"]),
    }
    return {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "status": "pass" if all(checks.values()) else "fail",
        "scientific_evidence": False,
        "interpretation": (
            "This is a real-checkpoint integration contract. It validates exact chart "
            "insertion and replay, but it does not test coordinate-dependent operations "
            "over a representative task sample."
        ),
        "model": {
            "model_id": bundle.model_id,
            "checkpoint_repo": config["checkpoint_repo"],
            "checkpoint_file": config["checkpoint_file"],
            "checkpoint_sha256": bundle.checkpoint_sha256,
            "coconut_source_commit": bundle.coconut_source_commit,
            "hidden_dimension": dimension,
            "parameter_dtype": str(bundle.model.embedding.weight.dtype),
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
            "question": config["question"],
            "num_latents": config["num_latents"],
            "seed": config["seed"],
            "affine_seed": config["affine_seed"],
            "affine_condition_number": config["affine_condition_number"],
            "continuation_horizon": config["continuation_horizon"],
            "continuation_seeds": config["continuation_seeds"],
            "candidate_perturbation_norm": config["candidate_perturbation_norm"],
        },
        "thresholds": thresholds,
        "metrics": metrics,
        "checks": checks,
    }


def main() -> int:
    default_config = Path(__file__).with_name("configs") / "public_gpt2_coconut_smoke_v1.json"
    parser = argparse.ArgumentParser(description="Validate the public GPT-2 Coconut adapter")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--workspace-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    result = run_smoke(_load_config(arguments.config), arguments.workspace_root.resolve())
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

