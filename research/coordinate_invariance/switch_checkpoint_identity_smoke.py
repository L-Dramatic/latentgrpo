from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch

from .real_models.switch import (
    SWITCH_ADAPTER_ID,
    SWITCH_ADAPTER_REVISION,
    SWITCH_BASE_MODEL_ID,
    SWITCH_BASE_REVISION,
    SwitchAuditRunner,
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
        "identity_runner": _sha256(Path(__file__).resolve()),
        "switch_adapter": _sha256(root / "real_models" / "switch.py"),
    }


def _ids_hash(ids: torch.Tensor) -> str:
    payload = json.dumps(ids.detach().cpu().tolist(), separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _require_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{name} is {actual!r}, expected {expected!r}")


def _reset_cuda_peak_memory_stats(device: torch.device) -> None:
    # PyTorch 2.8 memory-stat APIs do not initialize the CUDA allocator.
    torch.cuda.init()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def run(config: dict[str, Any], workspace_root: Path) -> dict[str, Any]:
    import accelerate
    import peft
    import transformers

    started = time.perf_counter()
    workspace_root = workspace_root.resolve()
    model_config = config["model"]
    _require_equal("base model id", model_config["base_id"], SWITCH_BASE_MODEL_ID)
    _require_equal(
        "base model revision", model_config["base_revision"], SWITCH_BASE_REVISION
    )
    _require_equal("adapter id", model_config["adapter_id"], SWITCH_ADAPTER_ID)
    _require_equal(
        "adapter revision", model_config["adapter_revision"], SWITCH_ADAPTER_REVISION
    )

    dataset_config = config["dataset"]
    dataset_path = workspace_root / str(dataset_config["relative_path"])
    dataset_bytes = dataset_path.read_bytes()
    dataset_sha256 = hashlib.sha256(dataset_bytes).hexdigest()
    _require_equal("dataset SHA-256", dataset_sha256, dataset_config["sha256"])
    rows = [
        json.loads(line)
        for line in dataset_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    _require_equal("dataset row count", len(rows), int(dataset_config["expected_rows"]))

    order_path = workspace_root / str(dataset_config["prompt_order_relative_path"])
    _require_equal(
        "prompt-order artifact SHA-256",
        _sha256(order_path),
        dataset_config["prompt_order_sha256"],
    )
    prompt_order = json.loads(order_path.read_text(encoding="utf-8"))
    _require_equal(
        "prompt-order config SHA-256",
        prompt_order["config_sha256"],
        dataset_config["prompt_order_config_sha256"],
    )
    prompt_count = int(config["selection"]["ordered_prompt_count"])
    candidates = prompt_order["ordered_candidates"][:prompt_count]
    if len(candidates) != prompt_count:
        raise ValueError("prompt-order artifact has too few candidates")
    for candidate in candidates:
        row = rows[int(candidate["dataset_index"])]
        problem_hash = hashlib.sha256(str(row["problem"]).encode("utf-8")).hexdigest()
        _require_equal("ordered problem SHA-256", problem_hash, candidate["problem_sha256"])
        _require_equal("ordered unique id", str(row["unique_id"]), candidate["unique_id"])

    device = torch.device(str(config["runtime"]["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("the frozen SWITCH identity smoke requires CUDA")
    seed = int(config["runtime"]["seed"])
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        _reset_cuda_peak_memory_stats(device)
        if bool(config["runtime"].get("disable_tf32", True)):
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    bundle = load_public_switch(
        source_directory=workspace_root / str(config["source"]["relative_path"]),
        cache_directory=workspace_root / str(model_config["cache_relative_path"]),
        device=device,
        attention_implementation=str(model_config["attention_implementation"]),
    )
    _require_equal("source commit", bundle.source_commit, config["source"]["commit"])
    runner = SwitchAuditRunner(bundle.model)
    generation = config["generation"]
    max_new_tokens = int(generation["max_new_tokens"])
    minimum_latent_dwell = int(generation["minimum_latent_dwell"])
    records: list[dict[str, Any]] = []
    token_mismatches = 0
    latent_info_mismatches = 0
    forward_count_mismatches = 0
    identity_hidden_mismatches = 0
    prompts_with_latent_blocks = 0
    total_model_forward_calls = 0

    for candidate in candidates:
        row = rows[int(candidate["dataset_index"])]
        prompt = build_switch_prompt(
            bundle.tokenizer,
            str(row["problem"]),
            suffix=str(config["prompt"]["suffix"]),
        )
        encoded = bundle.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            official_ids, official_info = bundle.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                min_latent_steps=minimum_latent_dwell,
            )
        official_forward_calls = int(bundle.model.gen_forward_cnt)
        audited = runner.run(
            input_ids,
            attention_mask,
            max_new_tokens=max_new_tokens,
            min_latent_steps=minimum_latent_dwell,
            capture_trace=True,
        )
        identity = runner.run(
            input_ids,
            attention_mask,
            max_new_tokens=max_new_tokens,
            min_latent_steps=minimum_latent_dwell,
            operation=lambda latent, _block, _step: latent,
            capture_trace=True,
        )
        tokens_exact = bool(
            torch.equal(official_ids, audited.output_ids)
            and torch.equal(audited.output_ids, identity.output_ids)
        )
        info_exact = bool(
            official_info == list(audited.latent_info)
            and audited.latent_info == identity.latent_info
        )
        forwards_exact = bool(
            official_forward_calls
            == audited.model_forward_calls
            == identity.model_forward_calls
        )
        hidden_exact = bool(
            len(audited.latent_steps) == len(identity.latent_steps)
            and all(
                torch.equal(left.proposed_native, right.proposed_native)
                and torch.equal(right.proposed_native, right.consumed_native)
                and torch.equal(left.exit_logits, right.exit_logits)
                for left, right in zip(audited.latent_steps, identity.latent_steps)
            )
        )
        token_mismatches += int(not tokens_exact)
        latent_info_mismatches += int(not info_exact)
        forward_count_mismatches += int(not forwards_exact)
        identity_hidden_mismatches += int(not hidden_exact)
        prompts_with_latent_blocks += int(bool(official_info))
        total_model_forward_calls += (
            official_forward_calls
            + audited.model_forward_calls
            + identity.model_forward_calls
        )
        records.append(
            {
                "scan_rank": int(candidate["scan_rank"]),
                "dataset_index": int(candidate["dataset_index"]),
                "unique_id": str(candidate["unique_id"]),
                "problem_sha256": str(candidate["problem_sha256"]),
                "prompt_token_count": int(input_ids.shape[1]),
                "generated_visible_token_count": int(
                    official_ids.shape[1] - input_ids.shape[1]
                ),
                "output_ids_sha256": _ids_hash(official_ids),
                "latent_blocks": len(official_info),
                "latent_steps": [int(item["n_latent_steps"]) for item in official_info],
                "natural_exits": [bool(item["natural_exit"]) for item in official_info],
                "model_forward_calls_per_path": official_forward_calls,
                "tokens_exact": tokens_exact,
                "latent_info_exact": info_exact,
                "forward_counts_exact": forwards_exact,
                "identity_hidden_exact": hidden_exact,
            }
        )

    thresholds = config["thresholds"]
    gates = {
        "ordered_prompt_count": len(records) == prompt_count,
        "latent_block_coverage": prompts_with_latent_blocks
        >= int(thresholds["minimum_prompts_with_latent_blocks"]),
        "official_audit_tokens_exact": token_mismatches
        <= int(thresholds["maximum_token_mismatches"]),
        "official_audit_latent_info_exact": latent_info_mismatches
        <= int(thresholds["maximum_latent_info_mismatches"]),
        "official_audit_forward_counts_exact": forward_count_mismatches
        <= int(thresholds["maximum_forward_count_mismatches"]),
        "identity_hook_hidden_exact": identity_hidden_mismatches
        <= int(thresholds["maximum_identity_hidden_mismatches"]),
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "experiment_name": config["experiment_name"],
        "status": "pass" if all(gates.values()) else "fail",
        "evidence_level": "paper-final checkpoint execution identity only",
        "config_sha256": canonical_config_hash(config),
        "runner_sha256": _sha256(Path(__file__)),
        "implementation_sha256": implementation_hashes(),
        "checkpoint": {
            "base_id": SWITCH_BASE_MODEL_ID,
            "base_revision": bundle.base_revision,
            "adapter_id": SWITCH_ADAPTER_ID,
            "adapter_revision": bundle.adapter_revision,
            "source_commit": bundle.source_commit,
            "attention_implementation": model_config["attention_implementation"],
        },
        "dataset_sha256": dataset_sha256,
        "prompts": records,
        "summary": {
            "prompt_count": len(records),
            "prompts_with_latent_blocks": prompts_with_latent_blocks,
            "token_mismatches": token_mismatches,
            "latent_info_mismatches": latent_info_mismatches,
            "forward_count_mismatches": forward_count_mismatches,
            "identity_hidden_mismatches": identity_hidden_mismatches,
        },
        "gates": gates,
        "runtime": {
            "seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "accelerate": accelerate.__version__,
            "device": str(device),
            "cuda_device": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "cuda_peak_allocated_mib": (
                float(torch.cuda.max_memory_allocated(device) / 1024**2)
                if device.type == "cuda"
                else None
            ),
            "model_forward_calls": total_model_forward_calls,
        },
        "interpretation": (
            "A pass establishes that the source-equivalent audit loop and an "
            "identity latent hook reproduce the released checkpoint exactly. "
            "It is not a scientific effect and does not authorize training."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = run(config, args.workspace_root)
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
