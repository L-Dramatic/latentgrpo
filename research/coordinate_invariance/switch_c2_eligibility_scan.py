from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, TextIO

import torch

from .real_models.switch import (
    SWITCH_ADAPTER_ID,
    SWITCH_ADAPTER_REVISION,
    SWITCH_BASE_MODEL_ID,
    SWITCH_BASE_REVISION,
    SwitchAuditRun,
    SwitchAuditRunner,
    SwitchReplayPlan,
    build_first_block_replay_plan,
    build_switch_prompt,
    load_public_switch,
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
        "eligibility_runner": _sha256(Path(__file__).resolve()),
        "switch_adapter": _sha256(root / "real_models" / "switch.py"),
    }


def classify_switch_run(
    run: SwitchAuditRun,
    *,
    prompt_length: int,
    max_new_tokens: int,
    swi_start_id: int,
    swi_end_id: int,
    eos_id: int,
    visible_horizon: int,
    reject_max_token_truncation: bool,
) -> tuple[str, SwitchReplayPlan | None]:
    generated = run.output_ids[0, prompt_length:].detach().cpu()
    if generated.numel() == 0:
        return "no_visible_output", None
    max_token_truncation = (
        generated.numel() >= max_new_tokens and int(generated[-1]) != int(eos_id)
    )
    if reject_max_token_truncation and max_token_truncation:
        return "max_token_truncation", None
    if not run.latent_info:
        return "no_latent_block", None
    first = run.latent_info[0]
    if not bool(first.get("natural_exit", False)):
        return "first_block_not_natural_exit", None
    end_index = int(first["position"])
    start_index = end_index - 1
    if start_index < 0 or end_index >= generated.numel():
        return "invalid_first_block_boundary", None
    if int(generated[start_index]) != int(swi_start_id):
        return "invalid_first_block_start", None
    if int(generated[end_index]) != int(swi_end_id):
        return "invalid_first_block_end", None
    target = generated[end_index + 1 : end_index + 1 + visible_horizon]
    if target.numel() != visible_horizon:
        return "short_post_block_horizon", None
    forbidden = {int(swi_start_id), int(swi_end_id), int(eos_id)}
    if any(int(token) in forbidden for token in target):
        return "boundary_or_eos_in_post_block_horizon", None
    plan = build_first_block_replay_plan(
        run,
        prompt_length=prompt_length,
        swi_start_id=swi_start_id,
        swi_end_id=swi_end_id,
        eos_id=eos_id,
        visible_horizon=visible_horizon,
    )
    return "eligible", plan


def _verify_pass_artifact(path: Path, expected_sha256: str) -> dict[str, Any]:
    actual_hash = _sha256(path)
    if actual_hash != expected_sha256:
        raise ValueError(
            f"prerequisite artifact {path} SHA-256 is {actual_hash}, "
            f"expected {expected_sha256}"
        )
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("status") != "pass":
        raise ValueError(f"prerequisite artifact {path} did not pass")
    return artifact


def _load_journal(
    path: Path,
    *,
    config_sha256: str,
    prompt_order_sha256: str,
    implementation: dict[str, str],
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or rows[0].get("kind") != "header":
        raise ValueError("eligibility journal lacks its header")
    header = rows[0]
    if header.get("config_sha256") != config_sha256:
        raise ValueError("eligibility journal belongs to a different config")
    if header.get("prompt_order_sha256") != prompt_order_sha256:
        raise ValueError("eligibility journal belongs to a different prompt order")
    if header.get("implementation_sha256") != implementation:
        raise ValueError("eligibility journal belongs to a different implementation")
    records = rows[1:]
    for expected_rank, record in enumerate(records):
        if record.get("kind") != "prompt" or int(record["scan_rank"]) != expected_rank:
            raise ValueError("eligibility journal ranks are not contiguous")
    return records


def _open_journal(
    path: Path,
    *,
    config_sha256: str,
    prompt_order_sha256: str,
    implementation: dict[str, str],
    existing_records: int,
) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a", encoding="utf-8", newline="\n")
    if existing_records == 0 and path.stat().st_size == 0:
        header = {
            "kind": "header",
            "config_sha256": config_sha256,
            "prompt_order_sha256": prompt_order_sha256,
            "implementation_sha256": implementation,
        }
        handle.write(json.dumps(header, sort_keys=True) + "\n")
        handle.flush()
    return handle


def _plan_to_dict(plan: SwitchReplayPlan) -> dict[str, Any]:
    payload = asdict(plan)
    for key, value in list(payload.items()):
        if isinstance(value, torch.Tensor):
            payload[key] = value.detach().cpu().tolist()
    return payload


def run(
    config: dict[str, Any],
    workspace_root: Path,
    *,
    journal_path: Path,
) -> dict[str, Any]:
    import accelerate
    import peft
    import transformers

    process_started = time.perf_counter()
    workspace_root = workspace_root.resolve()
    config_sha256 = canonical_config_hash(config)
    implementation = implementation_hashes()
    prerequisites = config["prerequisites"]
    source_preflight_path = workspace_root / str(
        prerequisites["source_preflight_artifact"]
    )
    c1_path = workspace_root / str(prerequisites["coconut_c1_artifact"])
    _verify_pass_artifact(
        source_preflight_path, str(prerequisites["source_preflight_artifact_sha256"])
    )
    _verify_pass_artifact(c1_path, str(prerequisites["coconut_c1_artifact_sha256"]))
    identity_config_path = workspace_root / str(
        prerequisites["checkpoint_identity_config"]
    )
    identity_config = json.loads(identity_config_path.read_text(encoding="utf-8"))
    if canonical_config_hash(identity_config) != str(
        prerequisites["checkpoint_identity_config_sha256"]
    ):
        raise ValueError("checkpoint identity config SHA-256 mismatch")
    identity_path = workspace_root / str(
        prerequisites["checkpoint_identity_artifact"]
    )
    identity_artifact = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity_artifact.get("status") != str(
        prerequisites["checkpoint_identity_required_status"]
    ):
        raise ValueError("paper-final checkpoint identity smoke did not pass")
    if identity_artifact.get("config_sha256") != str(
        prerequisites["checkpoint_identity_config_sha256"]
    ):
        raise ValueError("checkpoint identity artifact used a different config")

    dataset_config = config["dataset"]
    dataset_path = workspace_root / str(dataset_config["relative_path"])
    dataset_bytes = dataset_path.read_bytes()
    dataset_sha256 = hashlib.sha256(dataset_bytes).hexdigest()
    if dataset_sha256 != str(dataset_config["sha256"]):
        raise ValueError("MATH-500 dataset SHA-256 mismatch")
    dataset_rows = [
        json.loads(line)
        for line in dataset_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    if len(dataset_rows) != int(dataset_config["expected_rows"]):
        raise ValueError("MATH-500 row count mismatch")
    prompt_order_path = workspace_root / str(dataset_config["prompt_order_artifact"])
    prompt_order_sha256 = _sha256(prompt_order_path)
    if prompt_order_sha256 != str(dataset_config["prompt_order_artifact_sha256"]):
        raise ValueError("prompt-order artifact SHA-256 mismatch")
    prompt_order = json.loads(prompt_order_path.read_text(encoding="utf-8"))
    if prompt_order.get("config_sha256") != str(
        dataset_config["prompt_order_config_sha256"]
    ):
        raise ValueError("prompt-order artifact used a different config")

    selection = config["selection"]
    scan_limit = int(selection["scan_limit"])
    candidates = prompt_order["ordered_candidates"][:scan_limit]
    if len(candidates) != scan_limit:
        raise ValueError("prompt order contains fewer rows than the scan limit")
    journal_path = journal_path.resolve()
    records = _load_journal(
        journal_path,
        config_sha256=config_sha256,
        prompt_order_sha256=prompt_order_sha256,
        implementation=implementation,
    )
    if len(records) > scan_limit:
        raise ValueError("eligibility journal exceeds the frozen scan limit")
    for record, candidate in zip(records, candidates):
        if (
            int(record["dataset_index"]) != int(candidate["dataset_index"])
            or str(record["unique_id"]) != str(candidate["unique_id"])
        ):
            raise ValueError("eligibility journal does not follow the frozen order")

    model_config = config["model"]
    expected_model_fields = {
        "base_id": SWITCH_BASE_MODEL_ID,
        "base_revision": SWITCH_BASE_REVISION,
        "adapter_id": SWITCH_ADAPTER_ID,
        "adapter_revision": SWITCH_ADAPTER_REVISION,
    }
    for key, expected in expected_model_fields.items():
        if str(model_config[key]) != expected:
            raise ValueError(f"C2 model field {key} does not match the pinned release")
    device = torch.device(str(config["runtime"]["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("the frozen SWITCH eligibility scan requires CUDA")
    seed = int(config["runtime"]["seed"])
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        if bool(config["runtime"].get("disable_tf32", True)):
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    model_load_seconds = 0.0
    if len(records) < scan_limit:
        load_started = time.perf_counter()
        bundle = load_public_switch(
            source_directory=workspace_root / str(model_config["source_relative_path"]),
            cache_directory=workspace_root / str(model_config["cache_relative_path"]),
            device=device,
            attention_implementation=str(model_config["attention_implementation"]),
        )
        model_load_seconds = time.perf_counter() - load_started
        if bundle.source_commit != str(model_config["source_commit"]):
            raise ValueError("loaded SWITCH source differs from the frozen commit")
        runner = SwitchAuditRunner(bundle.model)
        existing_eligible = sum(record["eligibility"] == "eligible" for record in records)
        calibration_count = int(selection["calibration_eligible_prompts"])
        test_count = int(selection["test_eligible_prompts"])
        selected_limit = calibration_count + test_count
        max_new_tokens = int(selection["maximum_new_visible_tokens"])
        visible_horizon = int(selection["required_post_block_visible_tokens"])
        journal = _open_journal(
            journal_path,
            config_sha256=config_sha256,
            prompt_order_sha256=prompt_order_sha256,
            implementation=implementation,
            existing_records=len(records),
        )
        try:
            for candidate in candidates[len(records) :]:
                prompt_started = time.perf_counter()
                dataset_index = int(candidate["dataset_index"])
                row = dataset_rows[dataset_index]
                problem_hash = hashlib.sha256(
                    str(row["problem"]).encode("utf-8")
                ).hexdigest()
                if problem_hash != str(candidate["problem_sha256"]):
                    raise ValueError("ordered problem SHA-256 mismatch")
                if str(row["unique_id"]) != str(candidate["unique_id"]):
                    raise ValueError("ordered unique id mismatch")
                prompt = build_switch_prompt(
                    bundle.tokenizer,
                    str(row["problem"]),
                    suffix=str(config["prompt"]["suffix"]),
                )
                encoded = bundle.tokenizer(prompt, return_tensors="pt")
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                audit = runner.run(
                    input_ids,
                    attention_mask,
                    max_new_tokens=max_new_tokens,
                    min_latent_steps=int(selection["minimum_latent_dwell"]),
                    capture_trace=False,
                )
                eligibility, plan = classify_switch_run(
                    audit,
                    prompt_length=int(input_ids.shape[1]),
                    max_new_tokens=max_new_tokens,
                    swi_start_id=int(bundle.token_config.swi_start_id),
                    swi_end_id=int(bundle.token_config.swi_end_id),
                    eos_id=int(bundle.token_config.eos_token_id),
                    visible_horizon=visible_horizon,
                    reject_max_token_truncation=bool(
                        selection["reject_max_token_truncation"]
                    ),
                )
                selected_split = None
                selected_index = None
                replay_plan = None
                if eligibility == "eligible":
                    if existing_eligible < calibration_count:
                        selected_split = "calibration"
                        selected_index = existing_eligible
                    elif existing_eligible < selected_limit:
                        selected_split = "test"
                        selected_index = existing_eligible - calibration_count
                    if selected_split is not None:
                        if plan is None:
                            raise AssertionError("eligible prompt lacks a replay plan")
                        replay_plan = _plan_to_dict(plan)
                    existing_eligible += 1
                generated = audit.output_ids[0, input_ids.shape[1] :]
                record = {
                    "kind": "prompt",
                    "scan_rank": int(candidate["scan_rank"]),
                    "dataset_index": dataset_index,
                    "unique_id": str(candidate["unique_id"]),
                    "problem_sha256": problem_hash,
                    "subject": str(candidate["subject"]),
                    "level": int(candidate["level"]),
                    "eligibility": eligibility,
                    "selected_split": selected_split,
                    "selected_index": selected_index,
                    "prompt_token_count": int(input_ids.shape[1]),
                    "generated_visible_token_count": int(generated.numel()),
                    "generated_ids_sha256": hashlib.sha256(
                        json.dumps(
                            generated.detach().cpu().tolist(), separators=(",", ":")
                        ).encode("utf-8")
                    ).hexdigest(),
                    "latent_block_count": len(audit.latent_info),
                    "latent_steps": [
                        int(item["n_latent_steps"]) for item in audit.latent_info
                    ],
                    "natural_exits": [
                        bool(item["natural_exit"]) for item in audit.latent_info
                    ],
                    "model_forward_calls": int(audit.model_forward_calls),
                    "prompt_seconds": time.perf_counter() - prompt_started,
                    "replay_plan": replay_plan,
                }
                records.append(record)
                journal.write(json.dumps(record, sort_keys=True) + "\n")
                journal.flush()
        finally:
            journal.close()

    selected_calibration = [
        record for record in records if record["selected_split"] == "calibration"
    ]
    selected_test = [record for record in records if record["selected_split"] == "test"]
    test_subjects = sorted({str(record["subject"]) for record in selected_test})
    test_levels = sorted({int(record["level"]) for record in selected_test})
    reason_counts: dict[str, int] = {}
    for record in records:
        reason = str(record["eligibility"])
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    gates = {
        "complete_frozen_scan": len(records) == scan_limit,
        "calibration_count": len(selected_calibration)
        == int(selection["calibration_eligible_prompts"]),
        "test_count": len(selected_test) == int(selection["test_eligible_prompts"]),
        "test_subject_diversity": len(test_subjects)
        >= int(selection["minimum_test_subjects"]),
        "test_level_diversity": len(test_levels)
        >= int(selection["minimum_test_levels"]),
        "selected_plans_present": all(
            record["replay_plan"] is not None
            for record in selected_calibration + selected_test
        ),
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "experiment_name": config["experiment_name"] + "-eligibility",
        "status": "pass" if all(gates.values()) else "fail",
        "evidence_level": "frozen checkpoint-dependent sample assignment only",
        "config_sha256": config_sha256,
        "implementation_sha256": implementation,
        "prerequisites": {
            "source_preflight_sha256": _sha256(source_preflight_path),
            "coconut_c1_sha256": _sha256(c1_path),
            "checkpoint_identity_sha256": _sha256(identity_path),
        },
        "checkpoint": expected_model_fields | {
            "source_commit": str(model_config["source_commit"])
        },
        "dataset_sha256": dataset_sha256,
        "prompt_order_sha256": prompt_order_sha256,
        "scan": {
            "rows": len(records),
            "reason_counts": reason_counts,
            "eligible_total": reason_counts.get("eligible", 0),
            "selected_calibration": selected_calibration,
            "selected_test": selected_test,
            "test_subjects": test_subjects,
            "test_levels": test_levels,
        },
        "gates": gates,
        "runtime": {
            "current_process_seconds": time.perf_counter() - process_started,
            "summed_prompt_seconds": sum(float(row["prompt_seconds"]) for row in records),
            "model_load_seconds_current_process": model_load_seconds,
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
            "visible_tokens": sum(
                int(row["generated_visible_token_count"]) for row in records
            ),
            "latent_steps": sum(
                sum(int(value) for value in row["latent_steps"]) for row in records
            ),
            "model_forward_calls": sum(
                int(row["model_forward_calls"]) for row in records
            ),
        },
        "interpretation": (
            "A pass fixes the calibration and test prompts without using any "
            "geometry or effect outcome. It is not evidence for FCTR."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--journal", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = run(
        config,
        args.workspace_root,
        journal_path=args.journal,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "scan"}, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
