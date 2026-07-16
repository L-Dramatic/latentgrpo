"""Resumable real-checkpoint runner for sacrificial forward-KL discovery v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from research.behavioral_geometry.p1_sacrificial_protocol import (
    action_record_key,
    action_seed,
    history_seed,
    load_protocol,
    path_record_key,
    summarize_discovery,
    validate_protocol,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = (
    ROOT / "research" / "behavioral_geometry" / "configs" / "p1_sacrificial_discovery_v1.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "research" / "behavioral_geometry" / "results" / "p1_sacrificial_discovery_v1"
)
SOURCE_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def _progress(message: str) -> None:
    print(f"[p1-sacrificial-discovery] {message}", file=sys.stderr, flush=True)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_record_map(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            key = str(record.get("key", ""))
            if not key:
                raise ValueError(f"record line {line_number} has no key")
            if key in result:
                raise ValueError(f"duplicate record key {key!r} at line {line_number}")
            result[key] = record
    return result


def append_record(path: Path, record_map: dict[str, dict[str, Any]], record: dict[str, Any]) -> None:
    key = str(record.get("key", ""))
    if not key:
        raise ValueError("record has no key")
    if key in record_map:
        raise ValueError(f"refusing to overwrite completed record {key}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    record_map[key] = record


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _proposal_fingerprint(proposal) -> str:
    digest = hashlib.sha256()
    digest.update(int(proposal.next_token_id).to_bytes(8, "little", signed=True))
    for tensor in (
        proposal.topk_indices,
        proposal.topk_probs,
        proposal.topk_gumbels,
        proposal.topk_original_indices,
        proposal.topk_original_probs,
    ):
        digest.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def _sample_visible_history(
    model,
    embedding_weight,
    endpoint_output,
    position: int,
    *,
    terminal_token_ids: set[int],
    seed: int,
    horizon: int,
    temperature: float,
):
    import torch

    from research.behavioral_geometry.p1_forward_action_preflight import _step

    generator = torch.Generator(device="cuda").manual_seed(seed)
    current = endpoint_output
    token_ids: list[int] = []
    logits: list[torch.Tensor] = []
    termination_token_id: int | None = None
    for _ in range(horizon):
        raw_logits = current.logits[0, -1]
        logits.append(raw_logits.detach())
        token_id = int(
            torch.multinomial(
                torch.softmax(raw_logits.float() / temperature, dim=-1),
                1,
                generator=generator,
            ).item()
        )
        if not 0 <= token_id < embedding_weight.shape[0]:
            raise RuntimeError("visible sampler selected a token outside model rows")
        token_ids.append(token_id)
        if token_id in terminal_token_ids:
            termination_token_id = token_id
            break
        current = _step(model, current.past_key_values, embedding_weight[token_id], position)
        position += 1
    return token_ids, torch.stack(logits), termination_token_id


def _force_visible_history(
    model,
    embedding_weight,
    endpoint_output,
    position: int,
    *,
    token_ids: list[int],
    terminal_token_ids: set[int],
):
    import torch

    from research.behavioral_geometry.p1_forward_action_preflight import _step

    current = endpoint_output
    logits: list[torch.Tensor] = []
    for token_id in token_ids:
        logits.append(current.logits[0, -1].detach())
        if token_id in terminal_token_ids:
            break
        current = _step(model, current.past_key_values, embedding_weight[token_id], position)
        position += 1
    return torch.stack(logits)


def _path_result(sampled_logits, forced_logits, temperature: float) -> list[float]:
    from research.behavioral_geometry.p1_forward_kl_contract import paired_path_kl

    result = paired_path_kl(sampled_logits.float() / temperature, forced_logits.float() / temperature)
    return result.per_step.detach().cpu().tolist()


def lint_protocol(protocol_path: Path) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    audit = validate_protocol(protocol)
    return {"status": "PASS-PROTOCOL-LINT", **asdict(audit)}


def run_discovery(protocol_path: Path, output_dir: Path) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

    from research.behavioral_geometry.p1_forward_action_preflight import (
        FORBIDDEN_TOKENIZER_ONLY_ID,
        MODEL_VOCAB_SIZE,
        SOURCE_END_ID,
        _cache_ptrs,
        _close_source_action,
        _execute_under_actual_request,
        _prefill,
        _sample_source_action,
        _sampler_config,
        _source_weighted_embedding,
    )
    from research.behavioral_geometry.p1_official_sampler_replay import (
        replay_pinned_sampler_on_fake_request,
    )

    protocol = load_protocol(protocol_path)
    audit = validate_protocol(protocol)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    manifest_path = output_dir / "manifest.json"
    summary_path = output_dir / "summary.json"
    record_map = load_record_map(records_path)

    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("protocol_sha256") != audit.protocol_sha256:
            raise RuntimeError("resume manifest protocol hash does not match the frozen config")

    model_dir = (ROOT / str(protocol["checkpoint"])).resolve()
    model_file = model_dir / "model.safetensors"
    checkpoint_sha256 = _file_sha256(model_file)
    source_dir = ROOT / "_external" / "official_latent_grpo"
    actual_source_commit = subprocess.check_output(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_source_commit != str(protocol["source_commit"]):
        raise RuntimeError(
            f"official source commit mismatch: {actual_source_commit} != {protocol['source_commit']}"
        )
    started_wall = time.time()
    started = time.perf_counter()
    cap = int(protocol["wall_clock_cap_seconds"])
    model = None

    manifest: dict[str, Any] = {
        "protocol_id": audit.protocol_id,
        "protocol_sha256": audit.protocol_sha256,
        "checkpoint": str(model_dir),
        "checkpoint_sha256": checkpoint_sha256,
        "source_commit": actual_source_commit,
        "status": "STARTING",
        "started_unix": started_wall,
        "resumed_record_count": len(record_map),
        "completed_record_count": len(record_map),
        "expected_record_count": audit.expected_action_records + audit.expected_path_records,
    }
    write_json_atomic(manifest_path, manifest)

    def elapsed() -> float:
        return time.perf_counter() - started

    def enforce_budget() -> None:
        if elapsed() >= cap:
            raise TimeoutError("frozen one-hour wall-clock cap reached")

    def update_manifest(status: str, **extra: Any) -> None:
        manifest.update(
            {
                "status": status,
                "elapsed_seconds": elapsed(),
                "completed_record_count": len(record_map),
                **extra,
            }
        )
        if torch.cuda.is_available():
            manifest["gpu"] = {
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "free_bytes": int(torch.cuda.mem_get_info()[0]),
            }
        write_json_atomic(manifest_path, manifest)

    try:
        if not torch.cuda.is_available():
            update_manifest("BLOCKED_NO_CUDA")
            return manifest
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        _progress("loading frozen checkpoint once")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).to("cuda").eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        embedding_weight = model.get_input_embeddings().weight
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir, local_files_only=True, use_fast=True
        )
        from sglang.srt.hf_transformers_utils import attach_additional_stop_token_ids

        attach_additional_stop_token_ids(tokenizer)
        additional_stops = set(tokenizer.additional_stop_token_ids or [])
        if (
            SOURCE_END_ID == int(tokenizer.eos_token_id)
            or SOURCE_END_ID in additional_stops
            or SOURCE_END_ID in tokenizer.all_special_ids
        ):
            raise RuntimeError("source 524 is captured by a terminal tokenizer set")
        if len(tokenizer) <= FORBIDDEN_TOKENIZER_ONLY_ID:
            raise RuntimeError("expected tokenizer-only compress id is missing")
        if embedding_weight.shape[0] != MODEL_VOCAB_SIZE:
            raise RuntimeError("checkpoint model-row count changed")

        manifest["runtime"] = {
            "torch": torch.__version__,
            "transformers": transformers_version,
            "cuda_device": torch.cuda.get_device_name(0),
        }
        update_manifest("RUNNING")

        with torch.inference_mode():
            for prompt_index, prompt_item in enumerate(protocol["prompts"]):
                enforce_budget()
                prompt_id = str(prompt_item["prompt_id"])
                user_text = SOURCE_INSTRUCTION + "\n" + str(prompt_item["problem"])
                prompt_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if not prompt_text.rstrip().endswith("<think>"):
                    raise RuntimeError(f"{prompt_id} does not end with <think>")
                prompt_ids = tuple(
                    int(value)
                    for value in tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
                )
                if any(value < 0 or value >= MODEL_VOCAB_SIZE for value in prompt_ids):
                    raise RuntimeError(f"{prompt_id} contains a tokenizer-only model input")

                prefill = _prefill(model, prompt_ids)
                reference = replay_pinned_sampler_on_fake_request(
                    prefill.logits[0, -1], _sampler_config(noise=False)
                )
                reference_execution = _execute_under_actual_request(
                    tokenizer, reference, f"{prompt_id}-reference-admissibility"
                )
                if not reference_execution["continuous_action_consumed"]:
                    raise RuntimeError(f"{prompt_id} reference is not a continuous latent action")
                reference_embedding = _source_weighted_embedding(
                    embedding_weight, reference.topk_indices, reference.topk_probs
                )

                candidates = []
                for action_index in range(audit.candidate_count):
                    seed = action_seed(protocol, prompt_index, action_index)
                    proposal = _sample_source_action(prefill.logits[0, -1], seed)
                    execution = _execute_under_actual_request(
                        tokenizer, proposal, f"{prompt_id}-a{action_index}-admissibility"
                    )
                    embedding = _source_weighted_embedding(
                        embedding_weight, proposal.topk_indices, proposal.topk_probs
                    )
                    candidates.append((action_index, seed, proposal, execution, embedding))

                def fresh_closure(proposal, rid: str):
                    return _close_source_action(
                        model,
                        tokenizer,
                        embedding_weight,
                        _prefill(model, prompt_ids),
                        len(prompt_ids),
                        proposal,
                        rid,
                        latent_cap=int(protocol["natural_latent_cap"]),
                        force_visible_control=False,
                    )

                def assert_cache_disjoint(left_output, right_output, label: str) -> None:
                    left_ptrs = set(_cache_ptrs(left_output.past_key_values))
                    right_ptrs = set(_cache_ptrs(right_output.past_key_values))
                    if not left_ptrs or not right_ptrs or left_ptrs & right_ptrs:
                        raise RuntimeError(f"cache isolation failed for {label}")

                action_entries: dict[int | None, dict[str, Any]] = {}
                for action_index, seed, proposal, execution, embedding in [
                    (None, None, reference, reference_execution, reference_embedding),
                    *candidates,
                ]:
                    enforce_budget()
                    role = "reference" if action_index is None else "candidate"
                    key = action_record_key(prompt_id, action_index)
                    endpoint_output, _, closure = fresh_closure(
                        proposal, f"{prompt_id}-{role}-{action_index}-endpoint"
                    )
                    cache_ptrs = _cache_ptrs(endpoint_output.past_key_values)
                    record = {
                        "record_type": "action",
                        "key": key,
                        "status": "COMPLETE",
                        "protocol_id": audit.protocol_id,
                        "prompt_id": prompt_id,
                        "prompt_index": prompt_index,
                        "role": role,
                        "action_index": action_index,
                        "seed": seed,
                        "proxy": int(proposal.next_token_id),
                        "proposal_fingerprint": _proposal_fingerprint(proposal),
                        "continuous_action_consumed": execution[
                            "continuous_action_consumed"
                        ],
                        "structural_end_consumed": execution["structural_end_consumed"],
                        "endpoint": closure["endpoint"],
                        "latent_steps": len(closure["records"]),
                        "proxy_trace": [
                            int(item["proxy"]) for item in closure["records"]
                        ],
                        "cache_tensor_count": len(cache_ptrs),
                        "embedding_norm": float(embedding.float().norm().item()),
                        "embedding_l2_to_reference": float(
                            (embedding.float() - reference_embedding.float()).norm().item()
                        ),
                    }
                    if key in record_map:
                        previous = record_map[key]
                        for field in ("proposal_fingerprint", "endpoint", "proxy_trace"):
                            if previous.get(field) != record[field]:
                                raise RuntimeError(f"resume action mismatch for {key}: {field}")
                        action_entries[action_index] = previous
                    else:
                        append_record(records_path, record_map, record)
                        action_entries[action_index] = record
                        update_manifest("RUNNING", last_completed_key=key)

                temperature = float(protocol["visible_temperature"])
                horizon = int(protocol["max_visible_horizon"])
                eos_token_id = int(tokenizer.eos_token_id)
                terminal_token_ids = {eos_token_id, *additional_stops}

                # Forward histories are sampled once and reused across all four candidates.
                for history_index in range(audit.histories_per_direction):
                    missing_actions = [
                        action_index
                        for action_index in range(audit.candidate_count)
                        if path_record_key(prompt_id, action_index, "forward", history_index)
                        not in record_map
                    ]
                    if missing_actions:
                        enforce_budget()
                        seed = history_seed(
                            protocol,
                            direction="forward",
                            prompt_index=prompt_index,
                            history_index=history_index,
                        )
                        if action_entries[None]["endpoint"] == "NATURAL_VISIBLE":
                            ref_output, ref_position, ref_meta = fresh_closure(
                                reference, f"{prompt_id}-forward-h{history_index}-sample"
                            )
                            if ref_meta["endpoint"] != "NATURAL_VISIBLE":
                                raise RuntimeError("fresh forward reference endpoint changed")
                            token_ids, reference_logits, termination_token_id = _sample_visible_history(
                                model,
                                embedding_weight,
                                ref_output,
                                ref_position,
                                terminal_token_ids=terminal_token_ids,
                                seed=seed,
                                horizon=horizon,
                                temperature=temperature,
                            )
                        else:
                            token_ids, reference_logits, termination_token_id = [], None, None

                        for action_index in missing_actions:
                            enforce_budget()
                            key = path_record_key(
                                prompt_id, action_index, "forward", history_index
                            )
                            endpoint_eligible = bool(
                                action_entries[None]["endpoint"] == "NATURAL_VISIBLE"
                                and action_entries[action_index]["endpoint"] == "NATURAL_VISIBLE"
                            )
                            if endpoint_eligible:
                                proposal = candidates[action_index][2]
                                cand_output, cand_position, cand_meta = fresh_closure(
                                    proposal,
                                    f"{prompt_id}-a{action_index}-forward-h{history_index}-force",
                                )
                                if cand_meta["endpoint"] != "NATURAL_VISIBLE":
                                    raise RuntimeError("fresh forward candidate endpoint changed")
                                assert_cache_disjoint(
                                    ref_output,
                                    cand_output,
                                    f"{prompt_id}-a{action_index}-forward-h{history_index}",
                                )
                                candidate_logits = _force_visible_history(
                                    model,
                                    embedding_weight,
                                    cand_output,
                                    cand_position,
                                    token_ids=token_ids,
                                    terminal_token_ids=terminal_token_ids,
                                )
                                per_step = _path_result(
                                    reference_logits, candidate_logits, temperature
                                )
                                status = "COMPLETE"
                            else:
                                per_step = []
                                status = "INELIGIBLE_ENDPOINT"
                            record = {
                                "record_type": "path",
                                "key": key,
                                "status": status,
                                "protocol_id": audit.protocol_id,
                                "prompt_id": prompt_id,
                                "prompt_index": prompt_index,
                                "action_index": action_index,
                                "direction": "forward",
                                "history_index": history_index,
                                "history_seed": seed,
                                "endpoint_eligible": endpoint_eligible,
                                "visible_temperature": temperature,
                                "requested_horizon": horizon,
                                "valid_steps": len(per_step),
                                "terminated": bool(
                                    endpoint_eligible and termination_token_id is not None
                                ),
                                "termination_token_id": (
                                    termination_token_id if endpoint_eligible else None
                                ),
                                "eos_terminated": bool(
                                    endpoint_eligible
                                    and termination_token_id == eos_token_id
                                ),
                                "token_ids": token_ids if endpoint_eligible else [],
                                "per_step_kl": per_step,
                                "elapsed_seconds": elapsed(),
                            }
                            append_record(records_path, record_map, record)
                            update_manifest("RUNNING", last_completed_key=key)

                # Reverse histories are candidate-owned and cannot be shared across actions.
                for action_index in range(audit.candidate_count):
                    proposal = candidates[action_index][2]
                    for history_index in range(audit.histories_per_direction):
                        key = path_record_key(
                            prompt_id, action_index, "reverse", history_index
                        )
                        if key in record_map:
                            continue
                        enforce_budget()
                        seed = history_seed(
                            protocol,
                            direction="reverse",
                            prompt_index=prompt_index,
                            action_index=action_index,
                            history_index=history_index,
                        )
                        endpoint_eligible = bool(
                            action_entries[None]["endpoint"] == "NATURAL_VISIBLE"
                            and action_entries[action_index]["endpoint"] == "NATURAL_VISIBLE"
                        )
                        if endpoint_eligible:
                            cand_output, cand_position, cand_meta = fresh_closure(
                                proposal,
                                f"{prompt_id}-a{action_index}-reverse-h{history_index}-sample",
                            )
                            ref_output, ref_position, ref_meta = fresh_closure(
                                reference,
                                f"{prompt_id}-a{action_index}-reverse-h{history_index}-force",
                            )
                            if not (
                                cand_meta["endpoint"] == "NATURAL_VISIBLE"
                                and ref_meta["endpoint"] == "NATURAL_VISIBLE"
                            ):
                                raise RuntimeError("fresh reverse endpoint changed")
                            assert_cache_disjoint(
                                cand_output,
                                ref_output,
                                f"{prompt_id}-a{action_index}-reverse-h{history_index}",
                            )
                            token_ids, candidate_logits, termination_token_id = _sample_visible_history(
                                model,
                                embedding_weight,
                                cand_output,
                                cand_position,
                                terminal_token_ids=terminal_token_ids,
                                seed=seed,
                                horizon=horizon,
                                temperature=temperature,
                            )
                            reference_logits = _force_visible_history(
                                model,
                                embedding_weight,
                                ref_output,
                                ref_position,
                                token_ids=token_ids,
                                terminal_token_ids=terminal_token_ids,
                            )
                            per_step = _path_result(
                                candidate_logits, reference_logits, temperature
                            )
                            status = "COMPLETE"
                        else:
                            token_ids, per_step, termination_token_id = [], [], None
                            status = "INELIGIBLE_ENDPOINT"
                        record = {
                            "record_type": "path",
                            "key": key,
                            "status": status,
                            "protocol_id": audit.protocol_id,
                            "prompt_id": prompt_id,
                            "prompt_index": prompt_index,
                            "action_index": action_index,
                            "direction": "reverse",
                            "history_index": history_index,
                            "history_seed": seed,
                            "endpoint_eligible": endpoint_eligible,
                            "visible_temperature": temperature,
                            "requested_horizon": horizon,
                            "valid_steps": len(per_step),
                            "terminated": termination_token_id is not None,
                            "termination_token_id": termination_token_id,
                            "eos_terminated": termination_token_id == eos_token_id,
                            "token_ids": token_ids,
                            "per_step_kl": per_step,
                            "elapsed_seconds": elapsed(),
                        }
                        append_record(records_path, record_map, record)
                        update_manifest("RUNNING", last_completed_key=key)

                _progress(
                    f"prompt {prompt_index + 1}/{audit.prompt_count} complete; "
                    f"records={len(record_map)}/{audit.expected_action_records + audit.expected_path_records}; "
                    f"elapsed={elapsed():.1f}s"
                )
                torch.cuda.empty_cache()

        all_records = list(record_map.values())
        summary = summarize_discovery(protocol, all_records, run_complete=True)
        write_json_atomic(summary_path, summary)
        update_manifest("COMPLETE", decision=summary["decision"])
        return {"manifest": manifest, "summary": summary}
    except TimeoutError as exc:
        summary = summarize_discovery(protocol, list(record_map.values()), run_complete=False)
        write_json_atomic(summary_path, summary)
        update_manifest("BUDGET_STOP", error=str(exc), decision=summary["decision"])
        return {"manifest": manifest, "summary": summary}
    except KeyboardInterrupt:
        update_manifest("INTERRUPTED")
        raise
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        update_manifest("OOM", error=str(exc)[:1000])
        raise
    except Exception as exc:
        update_manifest("ERROR", error=f"{type(exc).__name__}: {exc}"[:2000])
        raise
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("lint", "run"), default="lint")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = (
        lint_protocol(args.protocol)
        if args.mode == "lint"
        else run_discovery(args.protocol, args.output_dir)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
