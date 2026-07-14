from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import Tensor

from research.simplex_policy.densities import sample_concrete

from .lepo import apply_lepo_sampling_filters
from .stage_b_common import (
    append_jsonl,
    canonical_config_hash,
    load_jsonl,
    load_model_and_tokenizer,
    prompt_text,
    quantile,
    runtime_metadata,
    select_rows,
)


@dataclass(frozen=True)
class SequentialConfig:
    raw: dict[str, Any]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "SequentialConfig":
        sampler = raw["sampler"]
        replay = raw["paired_replay"]
        if len(raw["model"]["revision"]) != 40:
            raise ValueError("a full model revision is required")
        if len(raw["dataset"]["revision"]) != 40:
            raise ValueError("a full dataset revision is required")
        if int(sampler["top_k"]) < 2:
            raise ValueError("top_k must be at least two")
        if not 0 < float(sampler["top_p"]) <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if int(sampler["latent_length"]) < 1:
            raise ValueError("latent_length must be positive")
        horizons = [int(value) for value in replay["measurement_horizons"]]
        if sorted(set(horizons)) != horizons:
            raise ValueError("measurement horizons must be unique and sorted")
        if horizons[-1] != int(sampler["latent_length"]):
            raise ValueError("the final horizon must equal latent_length")
        if not sampler["temperatures"] or not sampler["seeds"]:
            raise ValueError("temperatures and seeds cannot be empty")
        if not replay["shared_proxy_history"]:
            raise ValueError("v1 requires a shared source proxy history")
        return cls(raw=raw)

    @property
    def config_hash(self) -> str:
        return canonical_config_hash(self.raw)


def distribution_metrics(first_logits: Tensor, second_logits: Tensor) -> dict[str, float]:
    first_log_prob = F.log_softmax(first_logits.to(torch.float64), dim=-1)
    second_log_prob = F.log_softmax(second_logits.to(torch.float64), dim=-1)
    first_prob = first_log_prob.exp()
    second_prob = second_log_prob.exp()
    mixture = 0.5 * (first_prob + second_prob)
    log_mixture = mixture.clamp_min(torch.finfo(torch.float64).tiny).log()
    js = 0.5 * (
        (first_prob * (first_log_prob - log_mixture)).sum()
        + (second_prob * (second_log_prob - log_mixture)).sum()
    )
    total_variation = 0.5 * (first_prob - second_prob).abs().sum()
    return {
        "js": float(js),
        "total_variation": float(total_variation),
        "top1_disagreement": float(
            int(torch.argmax(first_logits) != torch.argmax(second_logits))
        ),
        "first_probability_sum_error": float(abs(first_prob.sum() - 1.0)),
        "second_probability_sum_error": float(abs(second_prob.sum() - 1.0)),
    }


def support_jaccard(
    first_logits: Tensor, second_logits: Tensor, *, top_k: int, top_p: float
) -> float:
    first = torch.isfinite(
        apply_lepo_sampling_filters(first_logits, top_k=top_k, top_p=top_p)
    )
    second = torch.isfinite(
        apply_lepo_sampling_filters(second_logits, top_k=top_k, top_p=top_p)
    )
    union = (first | second).sum()
    if int(union) == 0:
        raise RuntimeError("filtered supports cannot both be empty")
    return float((first & second).sum() / union)


def _trajectory_seed(
    base_seed: int, dataset_index: int, temperature_index: int, seed_index: int
) -> int:
    return (
        int(base_seed)
        + 1_000_003 * int(dataset_index)
        + 10_007 * int(temperature_index)
        + 101 * int(seed_index)
    )


def sample_filtered_action(
    active_logits: Tensor, *, temperature: float, generator: torch.Generator
) -> Tensor:
    if active_logits.numel() == 0:
        raise RuntimeError("the filtered support is empty")
    if active_logits.numel() == 1:
        return torch.ones(1, dtype=active_logits.dtype)
    action, _ = sample_concrete(
        active_logits,
        temperature=temperature,
        sample_shape=(),
        generator=generator,
    )
    return action


def _reference_nll(
    *,
    model: Any,
    embedding_weight: Tensor,
    past_key_values: Any,
    attention_mask: Tensor,
    current_logits: Tensor,
    target_ids: Tensor,
) -> Tensor:
    first_log_prob = F.log_softmax(current_logits.to(torch.float32), dim=-1)
    token_log_probs = [first_log_prob[:, target_ids[0]]]
    if target_ids.numel() > 1:
        teacher_ids = target_ids[:-1].to(embedding_weight.device)
        teacher_embeds = embedding_weight[teacher_ids]
        teacher_embeds = teacher_embeds.unsqueeze(0).expand(3, -1, -1)
        extension = torch.ones(
            (3, teacher_ids.numel()),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        extended_mask = torch.cat([attention_mask, extension], dim=1)
        outputs = model(
            inputs_embeds=teacher_embeds,
            attention_mask=extended_mask,
            past_key_values=past_key_values,
            use_cache=False,
            return_dict=True,
        )
        continuation_log_prob = F.log_softmax(
            outputs.logits.to(torch.float32), dim=-1
        )
        gather_ids = target_ids[1:].to(continuation_log_prob.device)
        gathered = continuation_log_prob.gather(
            2, gather_ids.view(1, -1, 1).expand(3, -1, 1)
        ).squeeze(-1)
        token_log_probs.append(gathered)
        del outputs
    return -torch.cat(
        [value.unsqueeze(1) if value.ndim == 1 else value for value in token_log_probs],
        dim=1,
    ).mean(dim=1)


def run_trajectory(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    solution: str,
    dataset_index: int,
    temperature: float,
    temperature_index: int,
    base_seed: int,
    seed_index: int,
    config: SequentialConfig,
) -> dict[str, Any]:
    raw = config.raw
    sampler = raw["sampler"]
    replay = raw["paired_replay"]
    top_k = int(sampler["top_k"])
    top_p = float(sampler["top_p"])
    latent_length = int(sampler["latent_length"])
    horizons = {int(value) for value in replay["measurement_horizons"]}
    max_prompt_tokens = int(raw["runtime"]["max_prompt_tokens"])

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_tokens,
        add_special_tokens=False,
    )
    input_device = model.get_input_embeddings().weight.device
    input_ids = encoded["input_ids"].to(input_device)
    branch_ids = input_ids.expand(3, -1)
    attention_mask = torch.ones_like(branch_ids)
    embedding_weight = model.get_input_embeddings().weight
    target_ids = tokenizer(
        solution,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0, : int(replay["reference_continuation_tokens"])]
    if target_ids.numel() == 0:
        raise RuntimeError("reference continuation is empty")

    generator = torch.Generator(device="cpu").manual_seed(
        _trajectory_seed(base_seed, dataset_index, temperature_index, seed_index)
    )
    action_sum_errors: list[float] = []
    identity_js_values: list[float] = []
    horizon_records: dict[str, dict[str, float]] = {}
    proxy_ids: list[int] = []
    singleton_action_steps = 0
    finite = True

    with torch.inference_mode():
        outputs = model(
            input_ids=branch_ids,
            attention_mask=attention_mask,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        current_logits = outputs.logits[:, -1, :]
        del outputs

        for latent_step in range(1, latent_length + 1):
            source_logits_cpu = current_logits[0].detach().to(
                device="cpu", dtype=torch.float32
            )
            finite = finite and bool(torch.isfinite(source_logits_cpu).all())
            filtered = apply_lepo_sampling_filters(
                source_logits_cpu, top_k=top_k, top_p=top_p
            )
            active = torch.isfinite(filtered)
            active_ids = torch.nonzero(active, as_tuple=False).squeeze(-1)
            active_logits = filtered[active].to(torch.float64)
            singleton_action_steps += int(active_logits.numel() == 1)
            action = sample_filtered_action(
                active_logits,
                temperature=temperature,
                generator=generator,
            )
            action_sum_errors.append(float(abs(action.sum() - 1.0)))
            proxy_id = int(torch.argmax(filtered))
            proxy_ids.append(proxy_id)

            active_embeddings = embedding_weight[
                active_ids.to(embedding_weight.device)
            ]
            mixture_embedding = (
                action.to(
                    device=active_embeddings.device,
                    dtype=active_embeddings.dtype,
                )
                @ active_embeddings
            )
            proxy_embedding = embedding_weight[
                torch.tensor(proxy_id, device=embedding_weight.device)
            ]
            next_embeds = torch.stack(
                [mixture_embedding, proxy_embedding, proxy_embedding], dim=0
            ).unsqueeze(1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (3, 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ],
                dim=1,
            )
            outputs = model(
                inputs_embeds=next_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            current_logits = outputs.logits[:, -1, :]
            del outputs

            finite = finite and bool(torch.isfinite(current_logits).all())
            if latent_step in horizons:
                continuous = current_logits[0].detach().to("cpu")
                projected = current_logits[1].detach().to("cpu")
                identity = current_logits[2].detach().to("cpu")
                metrics = distribution_metrics(continuous, projected)
                metrics["support_jaccard"] = support_jaccard(
                    continuous, projected, top_k=top_k, top_p=top_p
                )
                identity_metrics = distribution_metrics(projected, identity)
                metrics["identity_js"] = identity_metrics["js"]
                metrics["identity_total_variation"] = identity_metrics[
                    "total_variation"
                ]
                identity_js_values.append(identity_metrics["js"])
                horizon_records[str(latent_step)] = metrics

        nll = _reference_nll(
            model=model,
            embedding_weight=embedding_weight,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            current_logits=current_logits,
            target_ids=target_ids,
        ).detach().to(device="cpu", dtype=torch.float64)

    final = horizon_records[str(latent_length)]
    return {
        "temperature": temperature,
        "base_seed": int(base_seed),
        "latent_step_count": latent_length,
        "prompt_token_count": int(input_ids.shape[1]),
        "reference_token_count": int(target_ids.numel()),
        "singleton_action_steps": singleton_action_steps,
        "active_action_sum_error_max": max(action_sum_errors),
        "identity_branch_js_max": max(identity_js_values),
        "finite_trajectory": finite and bool(torch.isfinite(nll).all()),
        "completed_trajectory": len(proxy_ids) == latent_length,
        "proxy_ids_sha256": hashlib.sha256(
            torch.tensor(proxy_ids, dtype=torch.int64).numpy().tobytes()
        ).hexdigest(),
        "horizons": horizon_records,
        "final_js": final["js"],
        "final_total_variation": final["total_variation"],
        "final_top1_disagreement": final["top1_disagreement"],
        "final_support_jaccard": final["support_jaccard"],
        "continuous_reference_nll": float(nll[0]),
        "projected_reference_nll": float(nll[1]),
        "identity_reference_nll": float(nll[2]),
        "reference_nll_gap": float(nll[1] - nll[0]),
        "reference_nll_absolute_gap": float(abs(nll[1] - nll[0])),
        "identity_reference_nll_gap": float(abs(nll[2] - nll[1])),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("no trajectory records were produced")
    horizon_names = sorted(
        {name for record in records for name in record["horizons"]},
        key=int,
    )
    horizon_summary = {}
    for name in horizon_names:
        rows = [record["horizons"][name] for record in records]
        horizon_summary[name] = {
            "js_median": quantile((row["js"] for row in rows), 0.5),
            "total_variation_median": quantile(
                (row["total_variation"] for row in rows), 0.5
            ),
            "top1_disagreement_mean": sum(
                row["top1_disagreement"] for row in rows
            )
            / len(rows),
            "support_jaccard_median": quantile(
                (row["support_jaccard"] for row in rows), 0.5
            ),
        }
    return {
        "trajectory_count": len(records),
        "state_count": len({record["dataset_index"] for record in records}),
        "action_sum_error_p999": quantile(
            (record["active_action_sum_error_max"] for record in records), 0.999
        ),
        "identity_branch_js_max": max(
            record["identity_branch_js_max"] for record in records
        ),
        "identity_reference_nll_gap_max": max(
            record["identity_reference_nll_gap"] for record in records
        ),
        "singleton_action_step_rate": sum(
            record["singleton_action_steps"] for record in records
        )
        / sum(record["latent_step_count"] for record in records),
        "finite_trajectory_rate": sum(
            bool(record["finite_trajectory"]) for record in records
        )
        / len(records),
        "completed_trajectory_rate": sum(
            bool(record["completed_trajectory"]) for record in records
        )
        / len(records),
        "final_js_median": quantile(
            (record["final_js"] for record in records), 0.5
        ),
        "final_total_variation_median": quantile(
            (record["final_total_variation"] for record in records), 0.5
        ),
        "final_top1_disagreement_mean": sum(
            record["final_top1_disagreement"] for record in records
        )
        / len(records),
        "final_support_jaccard_median": quantile(
            (record["final_support_jaccard"] for record in records), 0.5
        ),
        "reference_nll_absolute_gap_median": quantile(
            (record["reference_nll_absolute_gap"] for record in records), 0.5
        ),
        "reference_nll_signed_gap_median": quantile(
            (record["reference_nll_gap"] for record in records), 0.5
        ),
        "horizons": horizon_summary,
    }


def evaluate_gates(summary: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    controls = {
        "action_normalization": summary["action_sum_error_p999"]
        <= float(gates["probability_sum_error_p999_max"]),
        "identity_branch": summary["identity_branch_js_max"]
        <= float(gates["identity_branch_js_max"]),
        "finite_trajectories": summary["finite_trajectory_rate"]
        >= float(gates["finite_trajectory_rate_min"]),
        "completed_trajectories": summary["completed_trajectory_rate"]
        >= float(gates["completed_trajectory_rate_min"]),
    }
    effects = {
        "distribution_js": summary["final_js_median"]
        >= float(gates["final_js_median_min"]),
        "total_variation": summary["final_total_variation_median"]
        >= float(gates["final_total_variation_median_min"]),
        "top1_decision": summary["final_top1_disagreement_mean"]
        >= float(gates["final_top1_disagreement_mean_min"]),
        "candidate_support": summary["final_support_jaccard_median"]
        <= float(gates["final_support_jaccard_median_max"]),
        "reference_continuation": summary["reference_nll_absolute_gap_median"]
        >= float(gates["reference_nll_absolute_gap_median_min"]),
    }
    effect_count = sum(effects.values())
    required = int(gates["effect_required_count"])
    return {
        "controls": controls,
        "effects": effects,
        "effect_pass_count": effect_count,
        "effect_required_count": required,
        "stage_b1_pass": all(controls.values()) and effect_count >= required,
    }


def run(
    *,
    config: SequentialConfig,
    model_dir: Path,
    records_path: Path,
    limit: int | None,
    resume: bool,
) -> dict[str, Any]:
    started = time.time()
    raw = config.raw
    dataset_raw = raw["dataset"]
    dataset = load_dataset(
        dataset_raw["id"],
        split=dataset_raw["split"],
        revision=dataset_raw["revision"],
        cache_dir=str(model_dir.parent / "hf_datasets"),
    )
    frozen_selection = select_rows(dataset, raw["selection"])
    selected = frozen_selection[:limit] if limit is not None else frozen_selection
    model, tokenizer = load_model_and_tokenizer(raw, model_dir)

    existing = load_jsonl(records_path) if resume else []
    if records_path.exists() and not resume:
        records_path.unlink()
    completed = {
        (
            int(record["dataset_index"]),
            float(record["temperature"]),
            int(record["base_seed"]),
        )
        for record in existing
    }
    records = list(existing)
    temperatures = [float(value) for value in raw["sampler"]["temperatures"]]
    seeds = [int(value) for value in raw["sampler"]["seeds"]]

    for selection_position, selected_row in enumerate(selected):
        row = selected_row.row
        prompt = prompt_text(tokenizer, str(row["problem"]), raw["prompt_contract"])
        for temperature_index, temperature in enumerate(temperatures):
            for seed_index, base_seed in enumerate(seeds):
                key = (selected_row.dataset_index, temperature, base_seed)
                if key in completed:
                    continue
                record = run_trajectory(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    solution=str(row["solution"]),
                    dataset_index=selected_row.dataset_index,
                    temperature=temperature,
                    temperature_index=temperature_index,
                    base_seed=base_seed,
                    seed_index=seed_index,
                    config=config,
                )
                record.update(
                    {
                        "dataset_index": selected_row.dataset_index,
                        "selection_position": selection_position,
                        "selection_hash": selected_row.selection_hash,
                        "unique_id": str(row[raw["selection"]["key"]]),
                        "problem_sha256": hashlib.sha256(
                            str(row["problem"]).encode("utf-8")
                        ).hexdigest(),
                        "subject": str(row.get("subject", "")),
                        "level": str(row.get("level", "")),
                    }
                )
                append_jsonl(records_path, record)
                records.append(record)
                completed.add(key)
        print(
            json.dumps(
                {
                    "processed_states": selection_position + 1,
                    "selected_states": len(selected),
                    "trajectory_records": len(records),
                }
            ),
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize_records(records)
    is_complete = limit is None and len(selected) == len(frozen_selection)
    expected_records = len(frozen_selection) * len(temperatures) * len(seeds)
    is_complete = is_complete and len(records) == expected_records
    gates = evaluate_gates(summary, raw["gates"]) if is_complete else None
    return {
        "experiment_name": raw["experiment_name"],
        "status": (
            "pass"
            if gates and gates["stage_b1_pass"]
            else "fail"
            if gates
            else "preflight"
        ),
        "config_sha256": config.config_hash,
        "model": raw["model"],
        "dataset": raw["dataset"],
        "frozen_selected_states": len(frozen_selection),
        "evaluated_states": len(selected),
        "expected_full_records": expected_records,
        "summary": summary,
        "gates": gates,
        "records_path": str(records_path),
        "runtime": runtime_metadata(time.time() - started),
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
    config = SequentialConfig.from_mapping(raw)
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
