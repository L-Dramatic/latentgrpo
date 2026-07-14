from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch

from research.coordinate_invariance.charts import AffineChart
from research.coordinate_invariance.real_models.coconut import (
    CoconutChartRunner,
    CoconutContinuationAdapter,
    load_public_coconut,
    prepare_coconut_input,
)

from .joint_kl import JointContinuationEvaluator


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("unsupported joint-KL smoke config")
    return config


def run_smoke(config: dict[str, Any], workspace_root: Path) -> dict[str, object]:
    import transformers

    torch.set_num_threads(min(16, torch.get_num_threads()))
    bundle = load_public_coconut(
        checkpoint_path=workspace_root / str(config["checkpoint_relative_path"]),
        coconut_source_directory=workspace_root
        / str(config["coconut_source_relative_path"]),
        model_id=str(config["model_id"]),
        device="cpu",
        cache_directory=workspace_root / str(config["model_cache_relative_path"]),
    )
    input_ids, attention_mask = prepare_coconut_input(
        bundle,
        str(config["question"]),
        num_latents=int(config["num_latents"]),
    )
    dimension = int(bundle.model.embedding.weight.shape[-1])
    identity = AffineChart.identity(
        dimension, dtype=torch.float64, compute_dtype=torch.float64
    )
    runner = CoconutChartRunner(bundle.model, identity)
    sample_id = "joint-kl-smoke"
    traced = runner.run(
        input_ids,
        attention_mask,
        seed=int(config["collection_seed"]),
        capture_trace=True,
        sample_id=sample_id,
    )
    record = traced.trace.get(sample_id, int(config["target_pass"]))
    policy = CoconutContinuationAdapter(
        runner,
        temperature=float(config["sampling_temperature"]),
        stop_at_eos=bool(config.get("stop_at_eos", False)),
    )
    horizon = int(config["continuation_horizon"])
    seeds = [int(seed) for seed in config["continuation_seeds"]]

    control = JointContinuationEvaluator(policy).directional(
        record.prefix_state,
        record.latent,
        record.latent.clone(),
        horizon=horizon,
        seeds=seeds,
    )
    control_logit_error = max(
        float((sampled.logits - forced.logits).abs().max())
        for sampled, forced in zip(
            control.reference_outputs, control.forced_candidate_outputs
        )
    )
    control_tokens_exact = all(
        torch.equal(sampled.token_ids, forced.token_ids)
        for sampled, forced in zip(
            control.reference_outputs, control.forced_candidate_outputs
        )
    )

    perturbation_generator = torch.Generator(device="cpu").manual_seed(
        int(config["candidate_perturbation_seed"])
    )
    direction = torch.randn(
        record.latent.shape,
        dtype=record.latent.dtype,
        generator=perturbation_generator,
    )
    direction = direction / torch.linalg.vector_norm(direction)
    candidate = record.latent + float(config["candidate_perturbation_norm"]) * direction
    candidate_result = JointContinuationEvaluator(policy).directional(
        record.prefix_state,
        record.latent,
        candidate,
        horizon=horizon,
        seeds=seeds,
    )

    metrics = {
        "control_mean_total_kl": control.mean_total_kl,
        "control_max_total_kl": float(control.per_seed_total_kl.max()),
        "control_max_logit_error": control_logit_error,
        "control_forced_tokens_exact": control_tokens_exact,
        "candidate_mean_total_kl": candidate_result.mean_total_kl,
        "candidate_mean_first_step_kl": candidate_result.mean_first_step_kl,
        "candidate_mean_tail_kl": candidate_result.mean_tail_kl,
        "candidate_per_seed_total_kl": candidate_result.per_seed_total_kl.tolist(),
        "candidate_per_seed_valid_steps": candidate_result.per_seed_valid_steps.tolist(),
        "candidate_compute": candidate_result.compute.to_dict(),
    }
    thresholds = config["thresholds"]
    checks = {
        "identical_latent_zero_joint_kl": metrics["control_max_total_kl"]
        <= float(thresholds["max_control_total_kl"]),
        "identical_latent_exact_logits": control_logit_error
        <= float(thresholds["max_control_logit_error"]),
        "forced_history_exact_tokens": control_tokens_exact,
        "candidate_sensitivity": candidate_result.mean_total_kl
        >= float(thresholds["min_candidate_total_kl"]),
    }
    return {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "status": "pass" if all(checks.values()) else "fail",
        "scientific_evidence": False,
        "interpretation": (
            "This real-checkpoint integration contract validates same-history "
            "teacher forcing and temperature-correct chain-rule joint KL. It does "
            "not establish that multi-step geometry beats next-token baselines."
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
        "config": {
            "question": config["question"],
            "num_latents": config["num_latents"],
            "target_pass": config["target_pass"],
            "collection_seed": config["collection_seed"],
            "continuation_horizon": horizon,
            "continuation_seeds": seeds,
            "sampling_temperature": config["sampling_temperature"],
            "stop_at_eos": bool(config.get("stop_at_eos", False)),
            "candidate_perturbation_seed": config["candidate_perturbation_seed"],
            "candidate_perturbation_norm": config["candidate_perturbation_norm"],
        },
        "thresholds": thresholds,
        "metrics": metrics,
        "checks": checks,
    }


def main() -> int:
    default_config = (
        Path(__file__).with_name("configs")
        / "public_gpt2_coconut_joint_kl_smoke_v2.json"
    )
    parser = argparse.ArgumentParser(
        description="Validate same-history joint KL on the public Coconut checkpoint"
    )
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument(
        "--workspace-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
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
