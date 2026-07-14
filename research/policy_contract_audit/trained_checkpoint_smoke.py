from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .stage_b_common import canonical_config_hash


def validate_config(raw: dict[str, Any]) -> None:
    if not raw.get("experiment_name"):
        raise ValueError("experiment_name is required")
    checkpoints = raw.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("at least one checkpoint is required")
    for checkpoint in checkpoints:
        if not checkpoint.get("name") or not checkpoint.get("id"):
            raise ValueError("every checkpoint requires name and id")
        if len(str(checkpoint.get("revision", ""))) != 40:
            raise ValueError("every checkpoint requires a full revision")
        if not checkpoint.get("local_dir") or not checkpoint.get("weight_file"):
            raise ValueError("every checkpoint requires local_dir and weight_file")
        if int(checkpoint.get("expected_weight_bytes", 0)) < 1:
            raise ValueError("expected_weight_bytes must be positive")
        if len(str(checkpoint.get("expected_weight_sha256", ""))) != 64:
            raise ValueError("expected_weight_sha256 must be complete")
    runtime = raw.get("runtime", {})
    if runtime.get("preferred_device") not in {"cuda", "cpu"}:
        raise ValueError("preferred_device must be cuda or cpu")
    if int(runtime.get("max_prompt_tokens", 0)) < 1:
        raise ValueError("max_prompt_tokens must be positive")
    if float(raw.get("gates", {}).get("repeat_max_abs_error_max", -1)) < 0:
        raise ValueError("repeatability tolerance must be nonnegative")


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_id_diagnostics(tokenizer: Any, embedding_rows: int) -> dict[str, Any]:
    token_ids = [int(value) for value in tokenizer.get_vocab().values()]
    if not token_ids:
        raise ValueError("tokenizer vocabulary is empty")
    out_of_range = sorted({value for value in token_ids if value >= embedding_rows})
    return {
        "tokenizer_length": len(tokenizer),
        "tokenizer_min_id": min(token_ids),
        "tokenizer_max_id": max(token_ids),
        "out_of_range_token_id_count": len(out_of_range),
        "out_of_range_token_ids": out_of_range,
    }


def summarize_gate_status(gates: dict[str, bool]) -> str:
    if not gates:
        raise ValueError("at least one gate is required")
    return "pass" if all(gates.values()) else "fail"


def render_prompt(tokenizer: Any, prompt_config: dict[str, Any]) -> tuple[str, bool]:
    messages = [
        {"role": "system", "content": str(prompt_config["system"])},
        {"role": "user", "content": str(prompt_config["user"])},
    ]
    if getattr(tokenizer, "chat_template", None):
        return (
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            ),
            False,
        )
    return "\n".join(message["content"] for message in messages), True


def load_and_check_checkpoint(
    checkpoint: dict[str, Any],
    prompt_config: dict[str, Any],
    runtime: dict[str, Any],
    gate_config: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    local_dir = (root / str(checkpoint["local_dir"])).resolve()
    weight_path = local_dir / str(checkpoint["weight_file"])
    if not weight_path.is_file():
        raise FileNotFoundError(f"missing checkpoint weight: {weight_path}")

    started = time.time()
    actual_bytes = weight_path.stat().st_size
    actual_sha256 = sha256_file(weight_path)

    tokenizer = AutoTokenizer.from_pretrained(
        local_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if runtime["preferred_device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("the smoke protocol requires CUDA but CUDA is unavailable")
    dtype = getattr(torch, str(checkpoint["dtype"]))
    max_memory: dict[Any, str] | None = None
    if runtime["preferred_device"] == "cuda":
        max_memory = {
            0: f"{int(runtime['max_gpu_memory_mib'])}MiB",
            "cpu": f"{int(runtime['max_cpu_memory_gib'])}GiB",
        }
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    load_started = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if runtime["preferred_device"] == "cuda" else "cpu",
        max_memory=max_memory,
        offload_state_dict=True,
    )
    model.eval()
    load_seconds = time.time() - load_started

    input_embedding = model.get_input_embeddings()
    output_embedding = model.get_output_embeddings()
    embedding_rows, embedding_dim = map(int, input_embedding.weight.shape)
    output_rows = int(output_embedding.weight.shape[0])
    tokenizer_diagnostics = tokenizer_id_diagnostics(tokenizer, embedding_rows)

    prompt_text, add_special_tokens = render_prompt(tokenizer, prompt_config)
    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=int(runtime["max_prompt_tokens"]),
        add_special_tokens=add_special_tokens,
    )
    input_ids = encoded["input_ids"]
    prompt_max_id = int(input_ids.max())
    prompt_min_id = int(input_ids.min())
    prompt_ids_in_range = prompt_min_id >= 0 and prompt_max_id < embedding_rows
    if not prompt_ids_in_range:
        raise ValueError(
            f"prompt id range [{prompt_min_id}, {prompt_max_id}] exceeds "
            f"embedding rows {embedding_rows}"
        )

    input_device = input_embedding.weight.device
    model_inputs = {
        key: value.to(input_device)
        for key, value in encoded.items()
        if isinstance(value, torch.Tensor)
    }
    forward_started = time.time()
    with torch.inference_mode():
        logits_a = model(**model_inputs, use_cache=False).logits[:, -1, :]
        logits_b = model(**model_inputs, use_cache=False).logits[:, -1, :]
        logits_a_float = logits_a.float()
        logits_b_float = logits_b.float()
        finite_logits = bool(torch.isfinite(logits_a_float).all().item())
        repeat_max_abs_error = float(
            (logits_a_float - logits_b_float).abs().max().item()
        )
        top_values, top_ids = torch.topk(logits_a_float[0], k=5)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    forward_seconds = time.time() - forward_started

    top_ids_list = [int(value) for value in top_ids.cpu().tolist()]
    top_values_list = [float(value) for value in top_values.cpu().tolist()]
    config_vocab_size = int(model.config.vocab_size)
    gates = {
        "weight_bytes_match": actual_bytes
        == int(checkpoint["expected_weight_bytes"]),
        "weight_sha256_match": actual_sha256
        == str(checkpoint["expected_weight_sha256"]),
        "config_vocab_matches_input_embedding": config_vocab_size
        == embedding_rows,
        "output_vocab_matches_input_embedding": output_rows == embedding_rows,
        "prompt_ids_in_range": prompt_ids_in_range,
        "finite_logits": finite_logits,
        "repeatability": repeat_max_abs_error
        <= float(gate_config["repeat_max_abs_error_max"]),
    }
    result = {
        "name": checkpoint["name"],
        "id": checkpoint["id"],
        "revision": checkpoint["revision"],
        "status": summarize_gate_status(gates),
        "local_dir": str(local_dir),
        "weight_file": str(weight_path),
        "weight_bytes": actual_bytes,
        "weight_sha256": actual_sha256,
        "architecture": list(getattr(model.config, "architectures", []) or []),
        "dtype": str(checkpoint["dtype"]),
        "config_vocab_size": config_vocab_size,
        "embedding_rows": embedding_rows,
        "embedding_dim": embedding_dim,
        "output_rows": output_rows,
        "tokenizer": tokenizer_diagnostics,
        "prompt": {
            "sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
            "token_count": int(input_ids.shape[1]),
            "min_id": prompt_min_id,
            "max_id": prompt_max_id,
        },
        "forward": {
            "finite_logits": finite_logits,
            "repeat_max_abs_error": repeat_max_abs_error,
            "top_token_ids": top_ids_list,
            "top_tokens": tokenizer.convert_ids_to_tokens(top_ids_list),
            "top_logits": top_values_list,
        },
        "gates": gates,
        "runtime": {
            "load_seconds": load_seconds,
            "forward_seconds": forward_seconds,
            "total_seconds": time.time() - started,
            "input_device": str(input_device),
            "hf_device_map": {
                str(key): str(value)
                for key, value in getattr(model, "hf_device_map", {}).items()
            },
            "cuda_peak_allocated_mib": (
                float(torch.cuda.max_memory_allocated() / 1024**2)
                if torch.cuda.is_available()
                else None
            ),
        },
    }

    del logits_a, logits_b, logits_a_float, logits_b_float
    del model_inputs, encoded, input_ids, model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run(raw: dict[str, Any], root: Path) -> dict[str, Any]:
    validate_config(raw)
    started = time.time()
    results: list[dict[str, Any]] = []
    for checkpoint in raw["checkpoints"]:
        try:
            result = load_and_check_checkpoint(
                checkpoint=checkpoint,
                prompt_config=raw["prompt"],
                runtime=raw["runtime"],
                gate_config=raw["gates"],
                root=root,
            )
        except Exception as exc:
            result = {
                "name": checkpoint["name"],
                "id": checkpoint["id"],
                "revision": checkpoint["revision"],
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        results.append(result)

    status = "pass" if all(row["status"] == "pass" for row in results) else "fail"
    return {
        "experiment_name": raw["experiment_name"],
        "status": status,
        "config_sha256": canonical_config_hash(raw),
        "checkpoints": results,
        "runtime": {
            "seconds": time.time() - started,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "platform": platform.platform(),
        },
        "interpretation": (
            "A pass establishes only that pinned native checkpoints load without "
            "repair, accept the frozen prompt, and produce finite repeatable logits."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    raw = json.loads(args.config.read_text(encoding="utf-8"))
    report = run(raw, args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        sys.exit(2)


if __name__ == "__main__":
    main()
