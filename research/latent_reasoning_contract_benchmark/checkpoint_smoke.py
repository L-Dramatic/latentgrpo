"""Pinned four-method checkpoint readiness smoke with one new CODI run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("configs") / "checkpoint_smoke_v1.json"
SOURCE_MANIFEST = Path(__file__).with_name("SOURCE_MANIFEST.json")


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"path escaped workspace: {relative}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    payload = value.detach().float().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), *args], text=True, encoding="utf-8"
    ).strip()


def _canonical_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_config(root: Path, config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("checkpoint smoke schema_version must be 1")
    if _sha256(SOURCE_MANIFEST) != config.get("source_manifest_sha256"):
        raise ValueError("checkpoint smoke source manifest hash drift")
    linked_methods = [
        method
        for evidence in config.get("linked_evidence", [])
        for method in evidence.get("methods", [])
    ]
    codi_id = config.get("codi", {}).get("method_id")
    if sorted(linked_methods + [codi_id]) != sorted(
        ["coconut_gpt2", "codi_gpt2", "latent_grpo_llama_1b", "soft_grpo_qwen_1_5b"]
    ):
        raise ValueError("checkpoint smoke must cover exactly four methods")
    codi = config["codi"]
    if int(codi["num_latent"]) != 6 or int(codi["inf_latent_iterations"]) != 6:
        raise ValueError("CODI checkpoint smoke must retain six latent iterations")
    if codi["required_versions"].get("peft") != "0.15.2":
        raise ValueError("CODI checkpoint smoke must pin peft 0.15.2")
    gates = config["gates"]
    required_true = (
        "require_zero_missing_keys",
        "require_zero_unexpected_keys",
        "require_finite_logits",
        "require_exact_repeatability",
        "require_prompt_ids_in_range",
        "require_exact_latent_step_count",
    )
    if any(gates.get(name) is not True for name in required_true):
        raise ValueError("checkpoint smoke may not disable mandatory gates")


def validate_linked_evidence(root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    for entry in config["linked_evidence"]:
        path = _inside(root, entry["path"])
        observed_sha = _sha256(path) if path.is_file() else None
        failures = []
        payload = None
        if observed_sha != entry["sha256"]:
            failures.append("artifact hash mismatch")
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") != entry["required_status"]:
                failures.append("artifact status mismatch")
            if entry.get("require_scientific_evidence_false") and payload.get(
                "scientific_evidence"
            ) is not False:
                failures.append("linked artifact is not engineering-only")
            checkpoint_ids = {
                checkpoint.get("id") for checkpoint in payload.get("checkpoints", [])
            }
            expected_checkpoint_ids = {
                "DJCheng/LLaMA3.2-1B-Instruct-Latent-GRPO-Top10",
                "zz1358m/SofT-GRPO-master",
            }
            if len(entry["methods"]) == 2 and checkpoint_ids != expected_checkpoint_ids:
                failures.append("linked trained-checkpoint coverage mismatch")
        results.append(
            {
                "methods": entry["methods"],
                "path": entry["path"],
                "sha256_expected": entry["sha256"],
                "sha256_observed": observed_sha,
                "pass": not failures,
                "failures": failures,
            }
        )
    return results


def _load_codi_module(source_root: Path) -> Any:
    path = source_root / "src" / "model.py"
    spec = importlib.util.spec_from_file_location("lrc_checkpoint_codi_source", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load pinned CODI model source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_codi_recurrence(model: Any, tokenizer: Any, prompt: str, latent_steps: int) -> dict[str, Any]:
    device = next(model.parameters()).device
    batch = tokenizer([prompt], return_tensors="pt", padding="longest")
    bot = torch.tensor([[model.bot_id]], dtype=torch.long)
    input_ids = torch.cat((batch["input_ids"], bot), dim=1).to(device)
    attention_mask = torch.cat((batch["attention_mask"], torch.ones_like(bot)), dim=1).to(device)
    latent_hashes = []
    with torch.inference_mode():
        outputs = model.codi(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            attention_mask=attention_mask,
        )
        cache = outputs.past_key_values
        latent = model.prj(outputs.hidden_states[-1][:, -1, :].unsqueeze(1))
        for _ in range(latent_steps):
            latent_hashes.append(_tensor_sha256(latent))
            outputs = model.codi(
                inputs_embeds=latent,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=cache,
            )
            cache = outputs.past_key_values
            latent = model.prj(outputs.hidden_states[-1][:, -1, :].unsqueeze(1))
        embedding = model.get_embd(model.codi, model.model_name)
        eot = embedding(torch.tensor([model.eot_id], device=device)).unsqueeze(0)
        visible = model.codi(inputs_embeds=eot, use_cache=True, past_key_values=cache)
        logits = visible.logits[:, -1, : model.codi.config.vocab_size - 1]
    return {
        "input_ids": input_ids.detach().cpu(),
        "latent_hashes": latent_hashes,
        "logits": logits.detach().cpu(),
    }


def run_codi_smoke(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    import peft
    import transformers
    from peft import LoraConfig, TaskType

    codi = config["codi"]
    source_root = _inside(root, codi["source_path"])
    checkpoint_path = _inside(root, codi["checkpoint_path"])
    base_root = _inside(root, codi["base_model_path"])
    failures = []
    source_commit = _git(source_root, "rev-parse", "HEAD")
    if source_commit != codi["source_commit"]:
        failures.append("source commit mismatch")
    checkpoint_sha = _sha256(checkpoint_path)
    if checkpoint_sha != codi["checkpoint_sha256"]:
        failures.append("checkpoint hash mismatch")
    if checkpoint_path.stat().st_size != int(codi["checkpoint_bytes"]):
        failures.append("checkpoint byte count mismatch")
    base_files = []
    for entry in codi["base_files"]:
        path = _inside(base_root, entry["path"])
        observed = _sha256(path) if path.is_file() else None
        if observed != entry["sha256"]:
            failures.append(f"base file mismatch: {entry['path']}")
        base_files.append(
            {"path": entry["path"], "sha256_expected": entry["sha256"], "sha256_observed": observed}
        )
    if peft.__version__ != codi["required_versions"]["peft"]:
        failures.append("peft version mismatch")
    if failures:
        return {"method_id": codi["method_id"], "status": "hold", "failures": failures}
    if not torch.cuda.is_available():
        return {
            "method_id": codi["method_id"],
            "status": "hold",
            "failures": ["CUDA is unavailable"],
        }

    torch.manual_seed(int(codi["seed"]))
    torch.cuda.manual_seed_all(int(codi["seed"]))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    module = _load_codi_module(source_root)
    model_config = codi["model"]
    model_args = module.ModelArguments(
        model_name_or_path=str(base_root),
        full_precision=True,
        train=False,
        lora_r=int(model_config["lora_r"]),
        lora_alpha=int(model_config["lora_alpha"]),
        lora_init=True,
        ckpt_dir=str(checkpoint_path.parent),
    )
    training_args = module.TrainingArguments(
        output_dir=str(root / "artifacts" / "latent_reasoning_contract_benchmark" / "_codi_smoke_unused"),
        report_to=[],
        bf16=bool(model_config["bf16"]),
        seed=int(codi["seed"]),
        num_latent=int(codi["num_latent"]),
        inf_latent_iterations=int(codi["inf_latent_iterations"]),
        use_lora=True,
        use_prj=bool(model_config["use_projection"]),
        prj_dim=int(model_config["projection_dim"]),
        prj_no_ln=not bool(model_config["projection_layer_norm"]),
        prj_dropout=0.0,
        greedy=True,
        remove_eos=True,
        disable_tqdm=True,
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=int(model_config["lora_r"]),
        lora_alpha=int(model_config["lora_alpha"]),
        lora_dropout=float(model_config["lora_dropout"]),
        target_modules=list(model_config["target_modules"]),
        init_lora_weights=True,
    )
    model = module.CODI(model_args, training_args, lora_config)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(state_dict, strict=False)
    model.codi.tie_weights()
    # Match official test.py lines 99-100: move first, then cast the complete
    # wrapper so the projection and BF16 backbone share one dtype.
    model = model.to("cuda:0")
    model.to(torch.bfloat16)
    model.eval()
    load_seconds = time.perf_counter() - start
    tokenizer = model.tokenizer
    first = _run_codi_recurrence(
        model, tokenizer, codi["prompt"], int(codi["inf_latent_iterations"])
    )
    second = _run_codi_recurrence(
        model, tokenizer, codi["prompt"], int(codi["inf_latent_iterations"])
    )
    torch.cuda.synchronize()
    peak_mib = torch.cuda.max_memory_allocated() / (1024**2)
    embedding_rows = model.get_embd(model.codi, model.model_name).weight.shape[0]
    gates = config["gates"]
    gate_results = {
        "zero_missing_keys": len(incompatible.missing_keys) == 0,
        "zero_unexpected_keys": len(incompatible.unexpected_keys) == 0,
        "finite_logits": bool(torch.isfinite(first["logits"]).all()),
        "exact_repeatability": first["latent_hashes"] == second["latent_hashes"]
        and torch.equal(first["logits"], second["logits"]),
        "prompt_ids_in_range": int(first["input_ids"].max()) < embedding_rows,
        "exact_latent_step_count": len(first["latent_hashes"])
        == int(codi["inf_latent_iterations"]),
        "cuda_memory_bounded": peak_mib <= float(gates["maximum_cuda_peak_allocated_mib"]),
    }
    top_values, top_ids = torch.topk(first["logits"][0], k=5)
    result = {
        "method_id": codi["method_id"],
        "status": "pass" if all(gate_results.values()) else "hold",
        "failures": [name for name, passed in gate_results.items() if not passed],
        "source_commit": source_commit,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_revision": codi["checkpoint_revision"],
        "base_model_revision": codi["base_model_revision"],
        "base_files": base_files,
        "versions": {
            "peft": peft.__version__,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
        },
        "load": {
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "seconds": load_seconds,
        },
        "prompt": {
            "token_count": int(first["input_ids"].numel()),
            "minimum_id": int(first["input_ids"].min()),
            "maximum_id": int(first["input_ids"].max()),
            "embedding_rows": int(embedding_rows),
            "ids_sha256": _tensor_sha256(first["input_ids"]),
        },
        "forward": {
            "latent_step_count": len(first["latent_hashes"]),
            "latent_hashes": first["latent_hashes"],
            "logits_sha256": _tensor_sha256(first["logits"]),
            "top_token_ids": [int(value) for value in top_ids.tolist()],
            "top_logits": [float(value) for value in top_values.tolist()],
        },
        "runtime": {
            "cuda_device": torch.cuda.get_device_name(0),
            "cuda_peak_allocated_mib": peak_mib,
            "total_seconds": time.perf_counter() - start,
        },
        "gates": gate_results,
    }
    del model, state_dict
    torch.cuda.empty_cache()
    return result


def run_checkpoint_smoke(root: Path, config_path: Path) -> dict[str, Any]:
    root = root.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(root, config)
    linked = validate_linked_evidence(root, config)
    try:
        codi = run_codi_smoke(root, config)
    except Exception as exc:  # Fail closed while preserving an auditable artifact.
        codi = {
            "method_id": config["codi"]["method_id"],
            "status": "hold",
            "failures": [f"{type(exc).__name__}: {exc}"],
        }
    controls = {
        "all_linked_evidence_valid": all(entry["pass"] for entry in linked),
        "codi_checkpoint_smoke_pass": codi["status"] == "pass",
        "four_methods_covered": len(
            {method for entry in linked for method in entry["methods"]}
            | {codi["method_id"]}
        )
        == 4,
    }
    passed = all(controls.values())
    return {
        "protocol_id": config["protocol_id"],
        "config_file_sha256": _sha256(config_path),
        "config_canonical_sha256": _canonical_sha256(config),
        "implementation_sha256": _sha256(Path(__file__)),
        "stage": "four_method_checkpoint_smoke",
        "gpu_used": True,
        "training_used": False,
        "status": "pass" if passed else "hold",
        "decision": "PASS_CHECKPOINT_SMOKE" if passed else "HOLD_CHECKPOINT_SMOKE",
        "controls": controls,
        "linked_evidence": linked,
        "codi": codi,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "scientific_evidence": False,
        "next_required_action": (
            "freeze tiny checkpoint-state intervention preflight"
            if passed
            else "do not inspect effects; version any required packaging repair"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_checkpoint_smoke(args.workspace, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "controls": report["controls"]}))
    return 0 if report["decision"] == "PASS_CHECKPOINT_SMOKE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
