from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from research.simplex_policy.densities import (
    concrete_log_density,
    concrete_score,
    sample_concrete,
)

from .contracts import score_mean_diagnostics
from .lepo import apply_lepo_sampling_filters


@dataclass(frozen=True)
class PublicCheckpointConfig:
    raw: dict[str, Any]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "PublicCheckpointConfig":
        model = raw["model"]
        dataset = raw["dataset"]
        sampler = raw["sampler"]
        perturbation = raw["fixed_support_perturbation"]
        runtime = raw["runtime"]
        if not model["id"] or len(model["revision"]) != 40:
            raise ValueError("model id and full revision are required")
        if not dataset["id"] or len(dataset["revision"]) != 40:
            raise ValueError("dataset id and full revision are required")
        if int(sampler["top_k"]) < 2 or not 0 < float(sampler["top_p"]) <= 1:
            raise ValueError("sampler top_k/top_p are invalid")
        if int(sampler["gumbel_draws_per_state"]) < 8:
            raise ValueError("at least eight Gumbel draws per seed are required")
        if len(sampler["seeds"]) < 2 or not sampler["temperatures"]:
            raise ValueError("multiple seeds and at least one temperature are required")
        if float(perturbation["gaussian_logit_rms"]) <= 0:
            raise ValueError("perturbation RMS must be positive")
        if int(runtime["batch_size"]) < 1 or int(runtime["max_prompt_tokens"]) < 1:
            raise ValueError("runtime batch and prompt limits must be positive")
        return cls(raw=raw)

    @property
    def config_hash(self) -> str:
        payload = json.dumps(
            self.raw, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _quantile(values: Iterable[float], probability: float) -> float:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    if tensor.numel() == 0:
        raise ValueError("cannot summarize an empty collection")
    return float(torch.quantile(tensor, probability))


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("no records were produced")
    return {
        "record_count": len(records),
        "state_count": len({row["state_index"] for row in records}),
        "archive_mass_error_p999": _quantile(
            (row["archive_mass_error_max"] for row in records), 0.999
        ),
        "reconstruction_l1_p999": _quantile(
            (row["reconstruction_l1_max"] for row in records), 0.999
        ),
        "active_support_size_max": max(row["active_support_size"] for row in records),
        "active_support_size_min": min(row["active_support_size"] for row in records),
        "exact_score_snr_p99": _quantile(
            (row["exact_score_snr"] for row in records), 0.99
        ),
        "exact_ratio_z_p99": _quantile(
            (row["exact_ratio_z_from_one"] for row in records), 0.99
        ),
        "excluded_full_model_mass_median": _quantile(
            (row["excluded_full_model_mass"] for row in records), 0.5
        ),
        "excluded_full_model_mass_p10": _quantile(
            (row["excluded_full_model_mass"] for row in records), 0.1
        ),
        "proxy_mode_disagreement_mean": sum(
            row["proxy_mode_disagreement"] for row in records
        )
        / len(records),
        "surrogate_exact_score_cosine_median": _quantile(
            (row["surrogate_exact_score_cosine_median"] for row in records), 0.5
        ),
        "surrogate_exact_score_relative_error_median": _quantile(
            (row["surrogate_exact_score_relative_error_median"] for row in records),
            0.5,
        ),
        "filtered_support_churn_mean": sum(
            row["filtered_support_churn"] for row in records
        )
        / len(records),
        "mixture_proxy_embedding_relative_l2_median": _quantile(
            (row["mixture_proxy_embedding_relative_l2_median"] for row in records),
            0.5,
        ),
    }


def evaluate_gates(summary: dict[str, Any], gates: dict[str, float]) -> dict[str, Any]:
    controls = {
        "archive_mass": summary["archive_mass_error_p999"]
        <= gates["archive_mass_error_p999_max"],
        "action_reconstruction": summary["reconstruction_l1_p999"]
        <= gates["reconstruction_l1_p999_max"],
        "support_bound": summary["active_support_size_max"]
        <= gates["active_support_size_max"],
        "exact_score_zero_mean": summary["exact_score_snr_p99"]
        <= gates["exact_score_snr_p99_max"],
        "exact_ratio_normalization": summary["exact_ratio_z_p99"]
        <= gates["exact_ratio_z_p99_max"],
    }
    effects = {
        "excluded_model_mass": summary["excluded_full_model_mass_median"]
        >= gates["excluded_full_model_mass_median_min"],
        "proxy_mode_divergence": summary["proxy_mode_disagreement_mean"]
        >= gates["proxy_mode_disagreement_mean_min"],
        "score_cosine_gap": summary["surrogate_exact_score_cosine_median"]
        <= gates["surrogate_exact_score_cosine_median_max"],
        "score_relative_error": summary[
            "surrogate_exact_score_relative_error_median"
        ]
        >= gates["surrogate_exact_score_relative_error_median_min"],
        "dynamic_support_churn": summary["filtered_support_churn_mean"]
        >= gates["filtered_support_churn_mean_min"],
    }
    effect_pass_count = sum(effects.values())
    proceed = all(controls.values()) and effect_pass_count >= 4
    return {
        "controls": controls,
        "effects": effects,
        "effect_pass_count": effect_pass_count,
        "effect_required_count": 4,
        "proceed_to_stage_b": proceed,
    }


def _prompt_text(tokenizer: Any, problem: str, prompt_contract: dict[str, Any]) -> str:
    system_prompt = str(prompt_contract["system_prompt"])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem + "\n" + system_prompt},
    ]
    if prompt_contract["apply_model_chat_template"]:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return "\n".join(message["content"] for message in messages)


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _importance_ratio_diagnostics(
    old_log_density: Tensor, new_log_density: Tensor
) -> tuple[float, float, float]:
    ratios = torch.exp(new_log_density - old_log_density)
    mean = ratios.mean()
    standard_error = ratios.std(unbiased=True) / math.sqrt(ratios.numel())
    z_score = torch.abs(mean - 1.0) / standard_error.clamp_min(
        torch.finfo(ratios.dtype).tiny
    )
    return float(mean), float(standard_error), float(z_score)


def _analyze_state_temperature(
    *,
    state_index: int,
    raw_logits: Tensor,
    active_embeddings: Tensor,
    temperature: float,
    temperature_index: int,
    config: PublicCheckpointConfig,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    raw = config.raw
    sampler = raw["sampler"]
    top_k = int(sampler["top_k"])
    top_p = float(sampler["top_p"])
    filtered = apply_lepo_sampling_filters(raw_logits, top_k=top_k, top_p=top_p)
    active = torch.isfinite(filtered)
    active_ids = torch.nonzero(active, as_tuple=False).squeeze(-1)
    active_logits = filtered[active].to(torch.float64)
    active_size = active_logits.numel()
    if active_size < 2 or active_size > top_k:
        raise RuntimeError(f"invalid active support size {active_size}")

    sample_blocks = []
    for seed_index, base_seed in enumerate(sampler["seeds"]):
        derived_seed = (
            int(base_seed)
            + 1_000_003 * state_index
            + 10_007 * temperature_index
            + 101 * seed_index
        )
        generator = torch.Generator(device="cpu").manual_seed(derived_seed)
        samples, _ = sample_concrete(
            active_logits,
            temperature=temperature,
            sample_shape=(int(sampler["gumbel_draws_per_state"]),),
            generator=generator,
        )
        sample_blocks.append(samples)
    samples = torch.cat(sample_blocks, dim=0)

    full_probabilities = torch.softmax(raw_logits.to(torch.float64), dim=-1)
    active_model_probabilities = full_probabilities[active]
    excluded_mass = 1.0 - active_model_probabilities.sum()
    excluded_square_mass = (
        full_probabilities.square().sum() - active_model_probabilities.square().sum()
    ).clamp_min(0.0)

    exact_scores = concrete_score(
        samples,
        active_logits.expand_as(samples),
        temperature=temperature,
    )
    surrogate_active_scores = samples - active_model_probabilities
    dot = (exact_scores * surrogate_active_scores).sum(dim=-1)
    exact_norm = exact_scores.norm(dim=-1)
    surrogate_norm = torch.sqrt(
        surrogate_active_scores.square().sum(dim=-1) + excluded_square_mass
    )
    score_cosine = dot / (exact_norm * surrogate_norm).clamp_min(1e-30)
    score_gap = torch.sqrt(
        (surrogate_active_scores - exact_scores).square().sum(dim=-1)
        + excluded_square_mass
    )
    relative_error = score_gap / exact_norm.clamp_min(1e-30)
    exact_score_stats = score_mean_diagnostics(exact_scores)

    perturbation = raw["fixed_support_perturbation"]
    drift_seed = int(perturbation["seed"]) + 1_000_003 * state_index
    drift_generator = torch.Generator(device="cpu").manual_seed(drift_seed)
    drift = torch.randn(
        raw_logits.shape, dtype=torch.float64, generator=drift_generator
    )
    drift = drift - drift.mean()
    drift = drift * (
        float(perturbation["gaussian_logit_rms"])
        / drift.square().mean().sqrt().clamp_min(1e-30)
    )
    current_raw_logits = raw_logits.to(torch.float64) + drift
    current_active_logits = current_raw_logits[active]
    old_log_density = concrete_log_density(
        samples,
        active_logits.expand_as(samples),
        temperature=temperature,
    )
    new_log_density = concrete_log_density(
        samples,
        current_active_logits.expand_as(samples),
        temperature=temperature,
    )
    ratio_mean, ratio_se, ratio_z = _importance_ratio_diagnostics(
        old_log_density, new_log_density
    )

    current_filtered = apply_lepo_sampling_filters(
        current_raw_logits, top_k=top_k, top_p=top_p
    )
    current_active = torch.isfinite(current_filtered)
    support_union = active | current_active
    support_churn = (active ^ current_active).sum() / support_union.sum()

    proxy_active_index = int(torch.argmax(active_logits))
    proxy_disagreement = (
        samples.argmax(dim=-1) != proxy_active_index
    ).to(torch.float64).mean()
    archive_mass_error = torch.abs(samples.sum(dim=-1) - 1.0)
    reconstruction_l1 = torch.zeros_like(archive_mass_error)

    embeddings = active_embeddings.to(torch.float64)
    mixture_embeddings = samples @ embeddings
    proxy_embedding = embeddings[proxy_active_index]
    embedding_distance = (mixture_embeddings - proxy_embedding).norm(dim=-1)
    embedding_relative = embedding_distance / mixture_embeddings.norm(dim=-1).clamp_min(
        1e-30
    )

    return {
        **metadata,
        "state_index": state_index,
        "temperature": temperature,
        "active_support_size": active_size,
        "archive_mass_error_max": float(archive_mass_error.max()),
        "reconstruction_l1_max": float(reconstruction_l1.max()),
        "excluded_full_model_mass": float(excluded_mass),
        "proxy_mode_disagreement": float(proxy_disagreement),
        "exact_score_snr": exact_score_stats.signal_to_noise,
        "exact_ratio_mean": ratio_mean,
        "exact_ratio_standard_error": ratio_se,
        "exact_ratio_z_from_one": ratio_z,
        "surrogate_exact_score_cosine_median": float(torch.quantile(score_cosine, 0.5)),
        "surrogate_exact_score_relative_error_median": float(
            torch.quantile(relative_error, 0.5)
        ),
        "filtered_support_churn": float(support_churn),
        "mixture_proxy_embedding_relative_l2_median": float(
            torch.quantile(embedding_relative, 0.5)
        ),
        "active_token_ids_sha256": hashlib.sha256(
            active_ids.numpy().tobytes()
        ).hexdigest(),
    }


def _load_model_and_tokenizer(
    config: PublicCheckpointConfig, model_dir: Path
) -> tuple[Any, Any]:
    raw = config.raw
    runtime = raw["runtime"]
    if runtime["preferred_device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("the frozen run requires CUDA but CUDA is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_name = raw["model"]["dtype"]
    dtype = getattr(torch, dtype_name)
    max_memory = {
        0: f"{int(runtime['max_gpu_memory_mib'])}MiB",
        "cpu": f"{int(runtime['max_cpu_memory_gib'])}GiB",
    }
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory=max_memory,
        offload_state_dict=True,
    )
    model.eval()
    return model, tokenizer


def run(
    *,
    config: PublicCheckpointConfig,
    model_dir: Path,
    records_path: Path,
    limit: int | None,
    resume: bool,
) -> dict[str, Any]:
    started = time.time()
    raw = config.raw
    dataset_config = raw["dataset"]
    dataset = load_dataset(
        dataset_config["id"],
        split=dataset_config["split"],
        revision=dataset_config["revision"],
        cache_dir=str(model_dir.parent / "hf_datasets"),
    )
    total_rows = len(dataset)
    selected_rows = min(total_rows, limit) if limit is not None else total_rows
    model, tokenizer = _load_model_and_tokenizer(config, model_dir)
    input_device = model.get_input_embeddings().weight.device
    embedding_weight = model.get_input_embeddings().weight

    existing = _load_records(records_path) if resume else []
    if records_path.exists() and not resume:
        records_path.unlink()
    completed = {
        (int(row["state_index"]), float(row["temperature"])) for row in existing
    }
    records = list(existing)
    batch_size = int(raw["runtime"]["batch_size"])
    max_prompt_tokens = int(raw["runtime"]["max_prompt_tokens"])
    temperatures = [float(value) for value in raw["sampler"]["temperatures"]]

    for batch_start in range(0, selected_rows, batch_size):
        indices = list(range(batch_start, min(batch_start + batch_size, selected_rows)))
        if all((index, temperature) in completed for index in indices for temperature in temperatures):
            continue
        rows = [dataset[index] for index in indices]
        prompts = [
            _prompt_text(tokenizer, row["problem"], raw["prompt_contract"])
            for row in rows
        ]
        tokens = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_tokens,
            add_special_tokens=False,
        )
        tokens = {key: value.to(input_device) for key, value in tokens.items()}
        with torch.inference_mode():
            outputs = model(**tokens, use_cache=False, logits_to_keep=1)
        batch_logits = outputs.logits[:, -1, :].detach().to(
            device="cpu", dtype=torch.float32
        )
        del outputs, tokens

        for local_index, (state_index, row) in enumerate(zip(indices, rows)):
            raw_logits = batch_logits[local_index]
            filtered = apply_lepo_sampling_filters(
                raw_logits,
                top_k=int(raw["sampler"]["top_k"]),
                top_p=float(raw["sampler"]["top_p"]),
            )
            active_ids = torch.nonzero(
                torch.isfinite(filtered), as_tuple=False
            ).squeeze(-1)
            with torch.inference_mode():
                active_embeddings = embedding_weight[
                    active_ids.to(embedding_weight.device)
                ].detach().to(device="cpu", dtype=torch.float32)
            metadata = {
                "problem_sha256": hashlib.sha256(
                    row["problem"].encode("utf-8")
                ).hexdigest(),
                "subject": str(row.get("subject", "")),
                "level": str(row.get("level", "")),
                "prompt_token_count": int(
                    tokenizer(prompts[local_index], add_special_tokens=False)[
                        "input_ids"
                    ].__len__()
                ),
            }
            for temperature_index, temperature in enumerate(temperatures):
                key = (state_index, temperature)
                if key in completed:
                    continue
                record = _analyze_state_temperature(
                    state_index=state_index,
                    raw_logits=raw_logits,
                    active_embeddings=active_embeddings,
                    temperature=temperature,
                    temperature_index=temperature_index,
                    config=config,
                    metadata=metadata,
                )
                _append_record(records_path, record)
                records.append(record)
                completed.add(key)
        if batch_start == 0 or (batch_start // batch_size + 1) % 10 == 0:
            print(
                json.dumps(
                    {
                        "processed_states": min(batch_start + len(indices), selected_rows),
                        "selected_states": selected_rows,
                        "records": len(records),
                    }
                ),
                flush=True,
            )
        del batch_logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_records(records)
    is_complete = selected_rows == total_rows and limit is None
    gates = evaluate_gates(summary, raw["gates"]) if is_complete else None
    return {
        "experiment_name": raw["experiment_name"],
        "status": (
            "pass" if gates and gates["proceed_to_stage_b"] else
            "fail" if gates else
            "preflight"
        ),
        "config_sha256": config.config_hash,
        "model": raw["model"],
        "dataset": raw["dataset"],
        "dataset_total_rows": total_rows,
        "evaluated_rows": selected_rows,
        "summary": summary,
        "gates": gates,
        "records_path": str(records_path),
        "runtime": {
            "seconds": time.time() - started,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "datasets": __import__("datasets").__version__,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "platform": platform.platform(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    raw = json.loads(args.config.read_text(encoding="utf-8"))
    config = PublicCheckpointConfig.from_mapping(raw)
    report = run(
        config=config,
        model_dir=args.model_dir,
        records_path=args.records,
        limit=args.limit,
        resume=args.resume,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] == "fail":
        sys.exit(2)


if __name__ == "__main__":
    main()
