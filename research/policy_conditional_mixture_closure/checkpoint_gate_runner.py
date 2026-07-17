"""GPU runner for the frozen PCMC A0 checkpoint gate.

The engineering preflight never writes scientific records. The A0 mode is
resumable at one immutable prompt record and refuses to run without a matching
asset audit and successful preflight from the same protocol and checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import torch

from .asset_preflight import verify_assets
from .checkpoint_adapters import (
    ConditionedAction,
    SourceAction,
    action_fingerprint,
    condition_on_content,
    sample_source_action,
    weighted_embedding,
)
from .checkpoint_protocol import (
    CheckpointProfile,
    checkpoint_profiles,
    load_protocol,
    validate_protocol,
)
from .gate_a_analysis import a0_record_key, write_json_atomic
from .task_manifest import RecordStore, build_a0_manifest


def _progress(message: str) -> None:
    print(f"[pcmc-a0] {message}", file=sys.stderr, flush=True)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _profile(protocol: dict[str, Any], checkpoint_id: str) -> CheckpointProfile:
    matches = [
        profile
        for profile in checkpoint_profiles(protocol)
        if profile.checkpoint_id == checkpoint_id
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown checkpoint id: {checkpoint_id}")
    return matches[0]


def _load_math500(path: Path) -> list[dict[str, Any]]:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    with pa.memory_map(str(path), "r") as source:
        table = ipc.open_stream(source).read_all()
    required = {"problem", "answer", "unique_id"}
    if table.num_rows != 500 or not required.issubset(table.column_names):
        raise ValueError("pinned MATH-500 Arrow schema or row count changed")
    return table.to_pylist()


def _render_prompt(
    tokenizer: Any,
    profile: CheckpointProfile,
    problem: str,
    maximum_prompt_tokens: int,
) -> tuple[str, tuple[int, ...]]:
    messages = [
        {"role": "system", "content": profile.prompt.system_prompt},
        {"role": "user", "content": problem},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=profile.prompt.add_generation_prompt,
    )
    if not rendered.rstrip().endswith(profile.prompt.required_rendered_suffix):
        raise RuntimeError("rendered prompt does not enter the thinking region")
    prompt_ids = tuple(
        int(value)
        for value in tokenizer(rendered, add_special_tokens=False)["input_ids"]
    )
    if not prompt_ids or len(prompt_ids) > maximum_prompt_tokens:
        raise RuntimeError("rendered prompt violates the frozen token-length cap")
    return rendered, prompt_ids


def _validate_tokenizer_contract(tokenizer: Any, profile: CheckpointProfile) -> None:
    encoded_end = [
        int(value)
        for value in tokenizer.encode("</think>", add_special_tokens=False)
    ]
    if not encoded_end:
        raise RuntimeError("tokenizer cannot encode the structural end marker")
    expected = int(profile.structural_end_token_id)
    observed = encoded_end[0] if profile.adapter == "official_latent_grpo_llama" else encoded_end[-1]
    if observed != expected:
        raise RuntimeError(
            f"structural end token mismatch: observed {observed}, expected {expected}"
        )


def _cache_tensors(value: Any) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    seen: set[int] = set()

    def visit(item: Any) -> None:
        identity = id(item)
        if identity in seen:
            return
        seen.add(identity)
        if isinstance(item, torch.Tensor):
            tensors.append(item)
        elif isinstance(item, dict):
            for key, child in item.items():
                visit(key)
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        elif hasattr(item, "__dict__"):
            for child in vars(item).values():
                visit(child)

    visit(value)
    return tensors


def _cache_ptrs(value: Any) -> set[int]:
    return {int(tensor.data_ptr()) for tensor in _cache_tensors(value)}


def _clone_cache(value: Any) -> Any:
    cloned = copy.deepcopy(value)
    source_ptrs = _cache_ptrs(value)
    cloned_ptrs = _cache_ptrs(cloned)
    if not source_ptrs or not cloned_ptrs or source_ptrs & cloned_ptrs:
        raise RuntimeError("KV cache deep-copy isolation failed")
    return cloned


def _repeat_cache(value: Any, repeats: int) -> Any:
    result = _clone_cache(value)
    if hasattr(result, "batch_repeat_interleave"):
        result.batch_repeat_interleave(repeats)
        return result
    if isinstance(result, (tuple, list)):
        layers = []
        for layer in result:
            layers.append(
                tuple(tensor.repeat_interleave(repeats, dim=0) for tensor in layer)
            )
        return tuple(layers)
    raise TypeError(f"unsupported transformers cache type: {type(result).__name__}")


def _prefill(model: Any, prompt_ids: tuple[int, ...]) -> Any:
    device = model.get_input_embeddings().weight.device
    input_ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    return model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=True,
        return_dict=True,
    )


def _step(
    model: Any,
    cache: Any,
    embeddings: torch.Tensor,
    position: int,
) -> Any:
    if embeddings.ndim == 1:
        embeddings = embeddings.unsqueeze(0)
    batch_size = embeddings.shape[0]
    device = embeddings.device
    return model(
        inputs_embeds=embeddings.unsqueeze(1),
        attention_mask=torch.ones(
            (batch_size, position + 1), dtype=torch.long, device=device
        ),
        position_ids=torch.full(
            (batch_size, 1), position, dtype=torch.long, device=device
        ),
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )


def _seeded_source_action(
    logits: torch.Tensor, profile: CheckpointProfile, seed: int
) -> SourceAction:
    device_index = logits.device.index
    devices = [] if device_index is None else [device_index]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if logits.is_cuda:
            torch.cuda.manual_seed_all(seed)
        return sample_source_action(logits, profile)


def _soft_official_replay_matches(
    logits: torch.Tensor,
    profile: CheckpointProfile,
    seed: int,
    action: SourceAction,
) -> bool:
    if profile.adapter != "soft_grpo_qwen":
        return True
    from .soft_official_sampler_replay import replay_pinned_soft_sampler

    sampler = profile.sampler
    device_index = logits.device.index
    devices = [] if device_index is None else [device_index]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if logits.is_cuda:
            torch.cuda.manual_seed_all(seed)
        official = replay_pinned_soft_sampler(
            logits,
            top_p=sampler.top_p,
            top_k=sampler.top_k,
            max_topk=sampler.max_topk,
            temperature=sampler.temperature,
            gumbel_softmax_temperature=sampler.gumbel_softmax_temperature,
            noise_scale=sampler.noise_scale,
        )
    return bool(
        action.proxy_token_id == official.next_token_id
        and torch.equal(action.token_ids, official.token_ids)
        and torch.equal(action.weights, official.weights)
    )


def _probability_hash(probabilities: torch.Tensor) -> str:
    values = probabilities.detach().float().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


def _jensen_shannon(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("JS divergence requires matching probability vectors")
    left = left.float()
    right = right.float()
    midpoint = 0.5 * (left + right)
    tiny = torch.finfo(torch.float32).tiny
    left_kl = torch.sum(left * (torch.log(left.clamp_min(tiny)) - torch.log(midpoint.clamp_min(tiny))))
    right_kl = torch.sum(right * (torch.log(right.clamp_min(tiny)) - torch.log(midpoint.clamp_min(tiny))))
    result = float((0.5 * (left_kl + right_kl)).item())
    if not math.isfinite(result) or result < -1e-7 or result > math.log(2.0) + 1e-6:
        raise RuntimeError(f"invalid JS divergence: {result}")
    return max(0.0, result)


def _one_step_closure(
    model: Any,
    prefill: Any,
    action: ConditionedAction,
    *,
    position: int,
    distribution_temperature: float,
) -> dict[str, Any]:
    embedding_table = model.get_input_embeddings().weight
    arithmetic_embedding = weighted_embedding(
        embedding_table, action.token_ids, action.weights
    )
    arithmetic_cache = _clone_cache(prefill.past_key_values)
    branch_cache = _repeat_cache(prefill.past_key_values, action.token_ids.numel())
    if _cache_ptrs(arithmetic_cache) & _cache_ptrs(branch_cache):
        raise RuntimeError("arithmetic and branch KV caches overlap")
    arithmetic_output = _step(
        model, arithmetic_cache, arithmetic_embedding, position
    )
    branch_embeddings = embedding_table[action.token_ids]
    branch_output = _step(model, branch_cache, branch_embeddings, position)
    arithmetic_distribution = torch.softmax(
        arithmetic_output.logits[0, -1].float() / distribution_temperature,
        dim=-1,
    )
    branch_distributions = torch.softmax(
        branch_output.logits[:, -1].float() / distribution_temperature,
        dim=-1,
    )
    branch_teacher = torch.sum(
        action.weights.float().unsqueeze(-1) * branch_distributions, dim=0
    )
    if abs(float(arithmetic_distribution.sum()) - 1.0) > 1e-5 or abs(
        float(branch_teacher.sum()) - 1.0
    ) > 1e-5:
        raise RuntimeError("one-step distributions failed normalization")
    return {
        "js_branch_arithmetic_nats": _jensen_shannon(
            arithmetic_distribution, branch_teacher
        ),
        "arithmetic_distribution_sha256": _probability_hash(
            arithmetic_distribution
        ),
        "branch_teacher_distribution_sha256": _probability_hash(branch_teacher),
        "arithmetic_top_token_id": int(arithmetic_distribution.argmax().item()),
        "branch_teacher_top_token_id": int(branch_teacher.argmax().item()),
    }


def _action_record(
    protocol: dict[str, Any],
    profile: CheckpointProfile,
    task: dict[str, Any],
    prompt_ids: tuple[int, ...],
    action: SourceAction,
    conditioned: ConditionedAction,
    closure: dict[str, Any] | None,
) -> dict[str, Any]:
    audit = validate_protocol(protocol)
    record: dict[str, Any] = {
        "record_type": "a0_event",
        "key": a0_record_key(profile.checkpoint_id, str(task["example_id"])),
        "status": conditioned.status,
        "protocol_id": audit.protocol_id,
        "protocol_sha256": audit.protocol_sha256,
        "checkpoint_id": profile.checkpoint_id,
        "example_id": str(task["example_id"]),
        "dataset_partition": str(task["dataset_partition"]),
        "action_seed": int(task["action_seed"]),
        "prompt_token_count": len(prompt_ids),
        "latent_step_index": 0,
        "structural_end_mass": conditioned.structural_end_mass,
        "content_support_size": int(conditioned.token_ids.numel()),
        "source_action_fingerprint": action_fingerprint(action),
        "source_proxy_token_id": action.proxy_token_id,
        "source_token_ids": [int(value) for value in action.token_ids.tolist()],
        "source_weights": [float(value) for value in action.weights.tolist()],
        "content_token_ids": [int(value) for value in conditioned.token_ids.tolist()],
        "content_weights": [float(value) for value in conditioned.weights.tolist()],
    }
    if conditioned.status == "INELIGIBLE_CONTENT":
        record["ineligible_reason"] = conditioned.reason
        return record
    weights = conditioned.weights.float()
    support_size = int(weights.numel())
    entropy = -torch.sum(weights * torch.log(weights.clamp_min(torch.finfo(torch.float32).tiny)))
    record.update(
        {
            "weight_entropy_normalized": float(entropy.item() / math.log(support_size)),
            "maximum_weight": float(weights.max().item()),
            "effective_support": float(1.0 / torch.sum(weights.square()).item()),
            **(closure or {}),
        }
    )
    return record


def _validate_asset_report(
    report: dict[str, Any],
    protocol: dict[str, Any],
    profile: CheckpointProfile,
) -> None:
    audit = validate_protocol(protocol)
    if report.get("status") != "PASS" or report.get("protocol_sha256") != audit.protocol_sha256:
        raise RuntimeError("asset report is not a PASS for the frozen protocol")
    matches = [
        value
        for value in report.get("checkpoints", [])
        if value.get("checkpoint_id") == profile.checkpoint_id
    ]
    if len(matches) != 1 or matches[0].get("status") != "PASS":
        raise RuntimeError("checkpoint is not covered by the passing asset report")


def _load_runtime(
    protocol: dict[str, Any], profile: CheckpointProfile, workspace_root: Path
) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for checkpoint execution")
    checkpoint_root = workspace_root / profile.checkpoint_path
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_root, local_files_only=True, use_fast=True
    )
    _validate_tokenizer_contract(tokenizer, profile)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_root,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to("cuda").eval()
    model.config.use_cache = True
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if model.get_input_embeddings().weight.shape[0] != model.config.vocab_size:
        raise RuntimeError("checkpoint embedding rows do not match model vocabulary")
    return model, tokenizer


def _task_rows(
    protocol: dict[str, Any], checkpoint_id: str
) -> list[dict[str, Any]]:
    manifest = build_a0_manifest(protocol)
    return [
        task
        for task in manifest["tasks"]
        if task["checkpoint_id"] == checkpoint_id
    ]


def run_engineering_preflight(
    protocol: dict[str, Any],
    profile: CheckpointProfile,
    workspace_root: Path,
    *,
    task_count: int,
) -> dict[str, Any]:
    if task_count < 1 or task_count > 16:
        raise ValueError("engineering preflight task count must be in [1, 16]")
    started = time.perf_counter()
    model = tokenizer = None
    task_results: list[dict[str, Any]] = []
    try:
        model, tokenizer = _load_runtime(protocol, profile, workspace_root)
        dataset = _load_math500(workspace_root / protocol["dataset"]["arrow_path"])
        tasks = _task_rows(protocol, profile.checkpoint_id)[:task_count]
        maximum_prompt_tokens = int(protocol["natural_action"]["maximum_prompt_tokens"])
        minimum_support = int(protocol["natural_action"]["minimum_content_support"])
        with torch.inference_mode():
            for task in tasks:
                index = int(str(task["example_id"]).split(":")[1])
                _, prompt_ids = _render_prompt(
                    tokenizer,
                    profile,
                    str(dataset[index]["problem"]),
                    maximum_prompt_tokens,
                )
                prefill = _prefill(model, prompt_ids)
                first = _seeded_source_action(
                    prefill.logits[0, -1], profile, int(task["action_seed"])
                )
                second = _seeded_source_action(
                    prefill.logits[0, -1], profile, int(task["action_seed"])
                )
                if action_fingerprint(first) != action_fingerprint(second):
                    raise RuntimeError("source action is not reproducible at a fixed seed")
                official_replay_match = _soft_official_replay_matches(
                    prefill.logits[0, -1],
                    profile,
                    int(task["action_seed"]),
                    first,
                )
                if not official_replay_match:
                    raise RuntimeError(
                        "SofT adapter differs from the unmodified pinned sampler"
                    )
                conditioned = condition_on_content(
                    first, profile, minimum_content_support=minimum_support
                )
                closure = None
                if conditioned.status == "COMPLETE":
                    closure = _one_step_closure(
                        model,
                        prefill,
                        conditioned,
                        position=len(prompt_ids),
                        distribution_temperature=float(
                            protocol["gate_a0"]["distribution_temperature"]
                        ),
                    )
                task_results.append(
                    {
                        "task_key": task["task_key"],
                        "status": conditioned.status,
                        "prompt_token_count": len(prompt_ids),
                        "content_support_size": int(conditioned.token_ids.numel()),
                        "structural_end_mass": conditioned.structural_end_mass,
                        "action_fingerprint": action_fingerprint(first),
                        "official_sampler_replay_match": official_replay_match,
                        "closure_executed": closure is not None,
                    }
                )
        complete_count = sum(value["closure_executed"] for value in task_results)
        status = (
            "PASS_ENGINEERING_PREFLIGHT"
            if complete_count > 0
            else "BLOCKED_NO_ELIGIBLE_ENGINEERING_ACTION"
        )
        return {
            "status": status,
            "scientific_evidence": False,
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": validate_protocol(protocol).protocol_sha256,
            "checkpoint_id": profile.checkpoint_id,
            "task_count": len(task_results),
            "closure_task_count": complete_count,
            "task_results": task_results,
            "runtime": {
                "torch": torch.__version__,
                "cuda_device": torch.cuda.get_device_name(0),
            },
            "elapsed_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _validate_preflight(
    result: dict[str, Any], protocol: dict[str, Any], profile: CheckpointProfile
) -> None:
    audit = validate_protocol(protocol)
    if result.get("status") != "PASS_ENGINEERING_PREFLIGHT":
        raise RuntimeError("A0 requires a passing engineering preflight")
    if result.get("scientific_evidence") is not False:
        raise RuntimeError("engineering preflight evidence boundary is invalid")
    if result.get("protocol_sha256") != audit.protocol_sha256:
        raise RuntimeError("engineering preflight protocol hash mismatch")
    if result.get("checkpoint_id") != profile.checkpoint_id:
        raise RuntimeError("engineering preflight checkpoint mismatch")


def run_a0(
    protocol: dict[str, Any],
    profile: CheckpointProfile,
    workspace_root: Path,
    output_root: Path,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    _validate_preflight(preflight, protocol, profile)
    audit = validate_protocol(protocol)
    store = RecordStore(output_root / "a0_store")
    existing = {str(record["key"]): record for record in store.records()}
    for record in existing.values():
        if record.get("protocol_sha256") != audit.protocol_sha256:
            raise RuntimeError("record store contains a different frozen protocol")
    tasks = _task_rows(protocol, profile.checkpoint_id)
    expected_keys = {str(task["task_key"]) for task in tasks}
    unexpected = sorted(
        key
        for key, record in existing.items()
        if record.get("checkpoint_id") == profile.checkpoint_id
        and key not in expected_keys
    )
    if unexpected:
        raise RuntimeError(f"record store contains unexpected checkpoint work: {unexpected[0]}")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / f"a0_runtime_{profile.checkpoint_id}.json"
    started_wall = time.time()
    started = time.perf_counter()
    wall_cap = int(
        protocol["execution"]["maximum_a0_wall_clock_seconds_per_checkpoint"]
    )
    model = tokenizer = None
    manifest: dict[str, Any] = {
        "status": "STARTING",
        "protocol_id": audit.protocol_id,
        "protocol_sha256": audit.protocol_sha256,
        "checkpoint_id": profile.checkpoint_id,
        "started_unix": started_wall,
        "expected_checkpoint_records": len(tasks),
        "resumed_checkpoint_records": sum(key in existing for key in expected_keys),
    }

    def update(status: str, **extra: Any) -> None:
        checkpoint_count = sum(key in existing for key in expected_keys)
        manifest.update(
            {
                "status": status,
                "elapsed_seconds": time.perf_counter() - started,
                "completed_checkpoint_records": checkpoint_count,
                **extra,
            }
        )
        if torch.cuda.is_available():
            manifest["gpu"] = {
                "device": torch.cuda.get_device_name(0),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "free_bytes": int(torch.cuda.mem_get_info()[0]),
            }
        write_json_atomic(manifest_path, manifest)

    update("STARTING")
    try:
        if all(key in existing for key in expected_keys):
            compacted = store.compact(output_root / "a0_records.jsonl")
            update("COMPLETE_CHECKPOINT", compaction=compacted, resumed_without_gpu=True)
            return manifest
        model, tokenizer = _load_runtime(protocol, profile, workspace_root)
        dataset = _load_math500(workspace_root / protocol["dataset"]["arrow_path"])
        maximum_prompt_tokens = int(protocol["natural_action"]["maximum_prompt_tokens"])
        minimum_support = int(protocol["natural_action"]["minimum_content_support"])
        update("RUNNING")
        with torch.inference_mode():
            for task in tasks:
                task_key = str(task["task_key"])
                if task_key in existing:
                    continue
                if time.perf_counter() - started >= wall_cap:
                    update("STOPPED_WALL_CLOCK_CAP")
                    return manifest
                index = int(str(task["example_id"]).split(":")[1])
                _, prompt_ids = _render_prompt(
                    tokenizer,
                    profile,
                    str(dataset[index]["problem"]),
                    maximum_prompt_tokens,
                )
                prefill = _prefill(model, prompt_ids)
                action = _seeded_source_action(
                    prefill.logits[0, -1], profile, int(task["action_seed"])
                )
                conditioned = condition_on_content(
                    action, profile, minimum_content_support=minimum_support
                )
                closure = None
                if conditioned.status == "COMPLETE":
                    closure = _one_step_closure(
                        model,
                        prefill,
                        conditioned,
                        position=len(prompt_ids),
                        distribution_temperature=float(
                            protocol["gate_a0"]["distribution_temperature"]
                        ),
                    )
                record = _action_record(
                    protocol,
                    profile,
                    task,
                    prompt_ids,
                    action,
                    conditioned,
                    closure,
                )
                store.put(record)
                existing[task_key] = record
                update("RUNNING", last_completed_key=task_key)
        compacted = store.compact(output_root / "a0_records.jsonl")
        update("COMPLETE_CHECKPOINT", compaction=compacted)
        return manifest
    except torch.OutOfMemoryError as exc:
        update("OOM", error=f"{type(exc).__name__}: {exc}"[:1000])
        raise
    except Exception as exc:
        update("ERROR", error=f"{type(exc).__name__}: {exc}"[:1000])
        raise
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("lint", "asset-preflight", "engineering-preflight", "run-a0"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--checkpoint-id")
    parser.add_argument("--asset-report", type=Path)
    parser.add_argument("--preflight-result", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--preflight-tasks", type=int, default=4)
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    audit = validate_protocol(protocol)
    workspace_root = args.workspace_root.resolve()
    if args.mode == "lint":
        print(json.dumps(asdict(audit), indent=2, sort_keys=True))
        return
    if args.mode == "asset-preflight":
        if args.output is None:
            parser.error("--output is required for asset-preflight")
        result = verify_assets(protocol, workspace_root)
        write_json_atomic(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["status"] != "PASS":
            raise SystemExit(2)
        return
    if not args.checkpoint_id:
        parser.error("--checkpoint-id is required for checkpoint execution")
    profile = _profile(protocol, args.checkpoint_id)
    if args.asset_report is None:
        parser.error("--asset-report is required for checkpoint execution")
    _validate_asset_report(_read_json(args.asset_report), protocol, profile)
    if args.mode == "engineering-preflight":
        if args.output is None:
            parser.error("--output is required for engineering-preflight")
        result = run_engineering_preflight(
            protocol,
            profile,
            workspace_root,
            task_count=args.preflight_tasks,
        )
        write_json_atomic(args.output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["status"] != "PASS_ENGINEERING_PREFLIGHT":
            raise SystemExit(3)
        return
    if args.preflight_result is None or args.output_root is None:
        parser.error("run-a0 requires --preflight-result and --output-root")
    result = run_a0(
        protocol,
        profile,
        workspace_root,
        args.output_root,
        _read_json(args.preflight_result),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "COMPLETE_CHECKPOINT":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
