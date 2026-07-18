"""Frozen deterministic checkpoint-intervention calibration for LRC-Bench."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("configs") / "deterministic_intervention_a0_v1.json"
SOURCE_MANIFEST = Path(__file__).with_name("SOURCE_MANIFEST.json")
EXPECTED_METHODS = ("coconut_gpt2", "codi_gpt2")


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


def _canonical_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    payload = value.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), *args], text=True, encoding="utf-8"
    ).strip()


def validate_config(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != 1:
        raise ValueError("deterministic A0 schema_version must be 1")
    if _sha256(SOURCE_MANIFEST) != config.get("source_manifest_sha256"):
        raise ValueError("source manifest hash drift")
    if tuple(config.get("methods", {}).keys()) != EXPECTED_METHODS:
        raise ValueError("deterministic A0 must cover Coconut and CODI in frozen order")
    required_conditions = {
        "native_live",
        "native_replay",
        "repeat_first",
        "reverse_steps",
        "zero_matched_depth",
        "norm_matched_random",
        "no_latent",
    }
    if set(config.get("conditions", [])) != required_conditions:
        raise ValueError("deterministic A0 condition set drift")
    equal_depth = set(config.get("equal_depth_effect_conditions", []))
    if equal_depth != required_conditions - {"native_live", "native_replay", "no_latent"}:
        raise ValueError("deterministic A0 equal-depth condition set drift")
    if any(int(method.get("latent_steps", 0)) != 6 for method in config["methods"].values()):
        raise ValueError("both deterministic methods must execute six latent steps")
    if config["methods"]["coconut_gpt2"].get("target_template") != "### {answer}":
        raise ValueError("Coconut target format drift")
    if config["methods"]["codi_gpt2"].get("target_template") != "The answer is: {answer}":
        raise ValueError("CODI target format drift")

    smoke_entry = config["checkpoint_smoke"]
    smoke_path = _inside(root, smoke_entry["path"])
    if _sha256(smoke_path) != smoke_entry["sha256"]:
        raise ValueError("checkpoint smoke artifact hash drift")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if smoke.get("decision") != smoke_entry["required_decision"]:
        raise ValueError("checkpoint smoke decision drift")

    prompt_entry = config["prompt_manifest"]
    prompt_path = _inside(root, prompt_entry["path"])
    if _sha256(prompt_path) != prompt_entry["sha256"]:
        raise ValueError("prompt manifest hash drift")
    prompt_manifest = json.loads(prompt_path.read_text(encoding="utf-8"))
    selected = [
        row for row in prompt_manifest.get("records", []) if row.get("split") == prompt_entry["split"]
    ]
    if len(selected) != int(prompt_entry["count"]):
        raise ValueError("prompt split count drift")
    if len({row["question_sha256"] for row in selected}) != len(selected):
        raise ValueError("prompt manifest contains duplicate questions")

    controls = config["controls"]
    if int(controls.get("require_exact_equal_depth_steps", 0)) != 6:
        raise ValueError("equal-depth control may not be relaxed")
    if float(controls.get("require_finite_rate", 0.0)) != 1.0:
        raise ValueError("finite-rate control may not be relaxed")
    if controls.get("require_target_token_identity") is not True:
        raise ValueError("target-token identity control may not be relaxed")
    signal = config["calibration_signal_gates"]
    if signal.get("require_signal_in_both_methods") is not True:
        raise ValueError("both-method signal gate may not be relaxed")
    return prompt_manifest


def _target_scores(logits: torch.Tensor, target_ids: torch.Tensor) -> tuple[float, torch.Tensor]:
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("scored logits must have shape [1, tokens, vocabulary]")
    if target_ids.ndim != 1 or logits.shape[1] != target_ids.numel():
        raise ValueError("scored logits and target ids must align")
    log_probs = F.log_softmax(logits.float(), dim=-1)
    nll = F.nll_loss(log_probs[0], target_ids.to(log_probs.device), reduction="mean")
    return float(nll), log_probs.detach().cpu()


def _mean_kl(native_log_probs: torch.Tensor, candidate_log_probs: torch.Tensor) -> float:
    if native_log_probs.shape != candidate_log_probs.shape:
        raise ValueError("KL inputs must be shape matched")
    values = (native_log_probs.exp() * (native_log_probs - candidate_log_probs)).sum(dim=-1)
    return float(values.mean())


def _norm_matched_random(states: list[torch.Tensor], seed: int) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    controls = []
    for state in states:
        sample = torch.randn(state.shape, generator=generator, dtype=torch.float32)
        sample_norm = torch.linalg.vector_norm(sample).clamp_min(1e-12)
        target_norm = torch.linalg.vector_norm(state.detach().float().cpu())
        controls.append((sample * (target_norm / sample_norm)).to(device=state.device, dtype=state.dtype))
    return controls


def _state_features(states: list[torch.Tensor]) -> dict[str, float]:
    float_states = [state.detach().float().cpu().reshape(-1) for state in states]
    norms = [float(torch.linalg.vector_norm(state)) for state in float_states]
    changes = [
        float(torch.linalg.vector_norm(right - left))
        for left, right in zip(float_states, float_states[1:])
    ]
    cosine = float(
        F.cosine_similarity(float_states[0].unsqueeze(0), float_states[-1].unsqueeze(0)).item()
    )
    return {
        "mean_norm": statistics.fmean(norms),
        "mean_step_change": statistics.fmean(changes) if changes else 0.0,
        "first_last_cosine": cosine,
    }


def _relative_displacement(native: list[torch.Tensor], candidate: list[torch.Tensor]) -> float:
    numerator = torch.sqrt(
        sum(torch.sum((left.detach().float().cpu() - right.detach().float().cpu()) ** 2) for left, right in zip(native, candidate))
    )
    denominator = torch.sqrt(
        sum(torch.sum(left.detach().float().cpu() ** 2) for left in native)
    ).clamp_min(1e-12)
    return float(numerator / denominator)


def _condition_payload(
    *, name: str, logits: torch.Tensor, target_ids: torch.Tensor, native_log_probs: torch.Tensor,
    latent_steps: int, displacement: float | None,
) -> tuple[dict[str, Any], torch.Tensor]:
    nll, log_probs = _target_scores(logits, target_ids)
    payload = {
        "nll": nll,
        "teacher_forced_kl_from_native": 0.0 if name == "native_live" else _mean_kl(native_log_probs, log_probs),
        "scored_logits_sha256": _tensor_sha256(logits.detach().float().cpu()),
        "latent_steps": latent_steps,
        "relative_state_displacement": displacement,
        "finite": bool(torch.isfinite(logits).all()) and math.isfinite(nll),
    }
    return payload, log_probs


def _run_coconut_prompt(bundle: Any, runner: Any, row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    method = config["methods"]["coconut_gpt2"]
    tokenizer = bundle.tokenizer
    question_ids = tokenizer.encode(row["question"] + "\n", add_special_tokens=True)
    target_ids_list = tokenizer.encode(
        method["target_template"].format(answer=row["final_answer"]), add_special_tokens=False
    ) + [tokenizer.eos_token_id]
    prefix = question_ids + [bundle.model.start_latent_id] + [bundle.model.latent_token_id] * 6 + [bundle.model.end_latent_id]
    full_ids = torch.tensor(prefix + target_ids_list, dtype=torch.long, device=bundle.device).unsqueeze(0)
    mask = torch.ones_like(full_ids)
    target_ids = full_ids[0, len(prefix) :]

    native = runner.run(full_ids, mask, seed=int(method["model_seed"]))
    native_states = [step.proposed_native.detach().clone() for step in native.latent_steps]
    native_logits = native.logits[:, len(prefix) - 1 : -1, :]
    native_nll, native_log_probs = _target_scores(native_logits, target_ids)
    conditions: dict[str, Any] = {
        "native_live": {
            "nll": native_nll,
            "teacher_forced_kl_from_native": 0.0,
            "scored_logits_sha256": _tensor_sha256(native_logits.detach().float().cpu()),
            "latent_steps": 6,
            "relative_state_displacement": 0.0,
            "finite": bool(torch.isfinite(native_logits).all()) and math.isfinite(native_nll),
        }
    }
    condition_states = {
        "native_replay": native_states,
        "repeat_first": [native_states[0]] * 6,
        "reverse_steps": list(reversed(native_states)),
        "zero_matched_depth": [torch.zeros_like(state) for state in native_states],
        "norm_matched_random": _norm_matched_random(
            native_states, int(config["random_seed"]) + int(row["dataset_index"])
        ),
    }
    native_replay_logits = None
    for name, states in condition_states.items():
        replay = runner.run(
            full_ids,
            mask,
            seed=int(method["model_seed"]),
            native_overrides={index: state for index, state in enumerate(states)},
        )
        logits = replay.logits[:, len(prefix) - 1 : -1, :]
        payload, _ = _condition_payload(
            name=name,
            logits=logits,
            target_ids=target_ids,
            native_log_probs=native_log_probs,
            latent_steps=len(states),
            displacement=_relative_displacement(native_states, states),
        )
        conditions[name] = payload
        if name == "native_replay":
            native_replay_logits = logits

    no_latent_ids = torch.tensor(
        question_ids + target_ids_list, dtype=torch.long, device=bundle.device
    ).unsqueeze(0)
    with torch.inference_mode():
        no_latent_output = bundle.model.base_causallm(input_ids=no_latent_ids)
    no_latent_logits = no_latent_output.logits[:, len(question_ids) - 1 : -1, :]
    conditions["no_latent"], _ = _condition_payload(
        name="no_latent",
        logits=no_latent_logits,
        target_ids=target_ids,
        native_log_probs=native_log_probs,
        latent_steps=0,
        displacement=None,
    )
    assert native_replay_logits is not None
    return {
        "method_id": "coconut_gpt2",
        "selection_rank": row["selection_rank"],
        "dataset_index": row["dataset_index"],
        "question_sha256": row["question_sha256"],
        "target_sha256": row["target_sha256"],
        "target_token_sha256": _tensor_sha256(target_ids.detach().cpu()),
        "target_token_count": int(target_ids.numel()),
        "native_features": _state_features(native_states),
        "conditions": conditions,
        "controls": {
            "native_replay_nll_absolute_error": abs(conditions["native_replay"]["nll"] - native_nll),
            "native_replay_logit_absolute_error": float((native_replay_logits - native_logits).abs().max()),
        },
    }


def _load_codi(root: Path, config: dict[str, Any]) -> Any:
    from peft import LoraConfig, TaskType

    from .checkpoint_smoke import _load_codi_module

    method = config["methods"]["codi_gpt2"]
    source_root = _inside(root, method["source_path"])
    if _git(source_root, "rev-parse", "HEAD") != method["source_commit"]:
        raise ValueError("CODI source commit mismatch")
    checkpoint = _inside(root, method["checkpoint_path"])
    if _sha256(checkpoint) != method["checkpoint_sha256"]:
        raise ValueError("CODI checkpoint hash mismatch")
    module = _load_codi_module(source_root)
    model_args = module.ModelArguments(
        model_name_or_path=str(_inside(root, method["base_model_path"])),
        full_precision=True,
        train=False,
        lora_r=int(method["lora_r"]),
        lora_alpha=int(method["lora_alpha"]),
        lora_init=True,
        ckpt_dir=str(checkpoint.parent),
    )
    training_args = module.TrainingArguments(
        output_dir=str(root / "artifacts" / "latent_reasoning_contract_benchmark" / "_a0_unused"),
        report_to=[],
        bf16=True,
        seed=int(method["model_seed"]),
        num_latent=6,
        inf_latent_iterations=6,
        use_lora=True,
        use_prj=True,
        prj_dim=int(method["projection_dim"]),
        prj_no_ln=False,
        prj_dropout=0.0,
        greedy=True,
        remove_eos=True,
        disable_tqdm=True,
    )
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=int(method["lora_r"]),
        lora_alpha=int(method["lora_alpha"]),
        lora_dropout=float(method["lora_dropout"]),
        target_modules=list(method["target_modules"]),
        init_lora_weights=True,
    )
    torch.manual_seed(int(method["model_seed"]))
    model = module.CODI(model_args, training_args, lora)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("CODI checkpoint key mismatch")
    model.codi.tie_weights()
    model = model.to("cuda:0")
    model.to(torch.bfloat16)
    model.eval()
    return model


def _codi_prefix(model: Any, question: str) -> tuple[Any, torch.Tensor, torch.Tensor]:
    batch = model.tokenizer([question], return_tensors="pt", padding="longest")
    bot = torch.tensor([[model.bot_id]], dtype=torch.long)
    input_ids = torch.cat((batch["input_ids"], bot), dim=1).to("cuda:0")
    attention = torch.cat((batch["attention_mask"], torch.ones_like(bot)), dim=1).to("cuda:0")
    outputs = model.codi(
        input_ids=input_ids,
        use_cache=True,
        output_hidden_states=True,
        attention_mask=attention,
    )
    latent = model.prj(outputs.hidden_states[-1][:, -1, :].unsqueeze(1))
    return outputs.past_key_values, latent, input_ids


def _codi_execute(
    model: Any, question: str, *, forced_states: list[torch.Tensor] | None, latent_steps: int
) -> tuple[Any, list[torch.Tensor], torch.Tensor]:
    cache, latent, input_ids = _codi_prefix(model, question)
    executed = []
    with torch.inference_mode():
        for index in range(latent_steps):
            consumed = latent if forced_states is None else forced_states[index].to(latent)
            executed.append(consumed.detach().clone())
            outputs = model.codi(
                inputs_embeds=consumed,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=cache,
            )
            cache = outputs.past_key_values
            latent = model.prj(outputs.hidden_states[-1][:, -1, :].unsqueeze(1))
    return cache, executed, input_ids


def _codi_score(model: Any, cache: Any, target_ids: torch.Tensor) -> torch.Tensor:
    decoder = torch.cat(
        [torch.tensor([model.eot_id], device="cuda:0"), target_ids.to("cuda:0")]
    ).unsqueeze(0)
    embedding = model.get_embd(model.codi, model.model_name)
    with torch.inference_mode():
        output = model.codi(inputs_embeds=embedding(decoder), use_cache=True, past_key_values=cache)
    return output.logits[:, :-1, :]


def _run_codi_prompt(model: Any, row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    method = config["methods"]["codi_gpt2"]
    answer = method["target_template"].format(answer=row["final_answer"])
    target_ids = torch.tensor(
        model.tokenizer.encode(answer, add_special_tokens=False) + [model.tokenizer.eos_token_id],
        dtype=torch.long,
        device="cuda:0",
    )
    native_cache, native_states, _ = _codi_execute(
        model, row["question"], forced_states=None, latent_steps=6
    )
    native_logits = _codi_score(model, native_cache, target_ids)
    native_nll, native_log_probs = _target_scores(native_logits, target_ids)
    conditions: dict[str, Any] = {
        "native_live": {
            "nll": native_nll,
            "teacher_forced_kl_from_native": 0.0,
            "scored_logits_sha256": _tensor_sha256(native_logits.detach().float().cpu()),
            "latent_steps": 6,
            "relative_state_displacement": 0.0,
            "finite": bool(torch.isfinite(native_logits).all()) and math.isfinite(native_nll),
        }
    }
    condition_states = {
        "native_replay": native_states,
        "repeat_first": [native_states[0]] * 6,
        "reverse_steps": list(reversed(native_states)),
        "zero_matched_depth": [torch.zeros_like(state) for state in native_states],
        "norm_matched_random": _norm_matched_random(
            native_states, int(config["random_seed"]) + int(row["dataset_index"])
        ),
    }
    native_replay_logits = None
    for name, states in condition_states.items():
        cache, _, _ = _codi_execute(model, row["question"], forced_states=states, latent_steps=6)
        logits = _codi_score(model, cache, target_ids)
        conditions[name], _ = _condition_payload(
            name=name,
            logits=logits,
            target_ids=target_ids,
            native_log_probs=native_log_probs,
            latent_steps=6,
            displacement=_relative_displacement(native_states, states),
        )
        if name == "native_replay":
            native_replay_logits = logits
    no_latent_cache, _, _ = _codi_execute(
        model, row["question"], forced_states=[], latent_steps=0
    )
    no_latent_logits = _codi_score(model, no_latent_cache, target_ids)
    conditions["no_latent"], _ = _condition_payload(
        name="no_latent",
        logits=no_latent_logits,
        target_ids=target_ids,
        native_log_probs=native_log_probs,
        latent_steps=0,
        displacement=None,
    )
    assert native_replay_logits is not None
    return {
        "method_id": "codi_gpt2",
        "selection_rank": row["selection_rank"],
        "dataset_index": row["dataset_index"],
        "question_sha256": row["question_sha256"],
        "target_sha256": row["target_sha256"],
        "target_token_sha256": _tensor_sha256(target_ids.detach().cpu()),
        "target_token_count": int(target_ids.numel()),
        "native_features": _state_features(native_states),
        "conditions": conditions,
        "controls": {
            "native_replay_nll_absolute_error": abs(conditions["native_replay"]["nll"] - native_nll),
            "native_replay_logit_absolute_error": float((native_replay_logits - native_logits).abs().max()),
        },
    }


def _bootstrap_mean(values: list[float], *, replicates: int, seed: int) -> list[float]:
    generator = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        estimates.append(statistics.fmean(generator.choice(values) for _ in values))
    estimates.sort()
    return [
        estimates[int(0.025 * (replicates - 1))],
        estimates[int(0.975 * (replicates - 1))],
    ]


def summarize_records(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    controls_cfg = config["controls"]
    signal_cfg = config["calibration_signal_gates"]
    expected_count = int(config["prompt_manifest"]["count"])
    methods: dict[str, Any] = {}
    all_controls = True
    for method_index, method_id in enumerate(EXPECTED_METHODS):
        method_records = sorted(
            [record for record in records if record["method_id"] == method_id],
            key=lambda record: record["selection_rank"],
        )
        finite_rate = statistics.fmean(
            float(condition["finite"])
            for record in method_records
            for condition in record["conditions"].values()
        ) if method_records else 0.0
        max_nll_error = max(
            (record["controls"]["native_replay_nll_absolute_error"] for record in method_records),
            default=float("inf"),
        )
        max_logit_error = max(
            (record["controls"]["native_replay_logit_absolute_error"] for record in method_records),
            default=float("inf"),
        )
        condition_complete = all(
            set(record["conditions"]) == set(config["conditions"]) for record in method_records
        )
        exact_depth = all(
            record["conditions"][name]["latent_steps"] == int(controls_cfg["require_exact_equal_depth_steps"])
            for record in method_records
            for name in config["equal_depth_effect_conditions"]
        )
        method_controls = {
            "expected_prompt_count": len(method_records) == expected_count,
            "condition_complete": condition_complete,
            "finite_rate": finite_rate,
            "finite_rate_pass": finite_rate == float(controls_cfg["require_finite_rate"]),
            "native_replay_nll_absolute_error_max": max_nll_error,
            "native_replay_nll_pass": max_nll_error <= float(controls_cfg["maximum_native_replay_nll_absolute_error"]),
            "native_replay_logit_absolute_error_max": max_logit_error,
            "native_replay_logit_pass": max_logit_error <= float(controls_cfg["maximum_native_replay_logit_absolute_error"]),
            "exact_equal_depth_steps": exact_depth,
        }
        controls_pass = all(
            value for key, value in method_controls.items() if key.endswith("_pass") or key in {"expected_prompt_count", "condition_complete", "exact_equal_depth_steps"}
        )
        all_controls = all_controls and controls_pass
        effects = {}
        signal_conditions = []
        for condition_index, name in enumerate(config["conditions"]):
            if name == "native_live":
                continue
            deltas = [
                record["conditions"][name]["nll"] - record["conditions"]["native_live"]["nll"]
                for record in method_records
            ]
            kls = [record["conditions"][name]["teacher_forced_kl_from_native"] for record in method_records]
            mean_delta = statistics.fmean(deltas) if deltas else float("nan")
            mean_kl = statistics.fmean(kls) if kls else float("nan")
            nonzero_fraction = (
                statistics.fmean(
                    float(abs(value) > float(signal_cfg["nonzero_nll_delta_epsilon"])) for value in deltas
                )
                if deltas
                else 0.0
            )
            effect = {
                "mean_nll_delta": mean_delta,
                "median_nll_delta": statistics.median(deltas) if deltas else float("nan"),
                "mean_absolute_nll_delta": statistics.fmean(abs(value) for value in deltas) if deltas else float("nan"),
                "mean_nll_delta_bootstrap_95": _bootstrap_mean(
                    deltas,
                    replicates=int(config["bootstrap_replicates"]),
                    seed=int(config["bootstrap_seed"]) + 100 * method_index + condition_index,
                ) if deltas else [float("nan"), float("nan")],
                "mean_teacher_forced_kl": mean_kl,
                "nonzero_nll_delta_fraction": nonzero_fraction,
            }
            if name in config["equal_depth_effect_conditions"]:
                effect["clears_calibration_signal"] = (
                    abs(mean_delta) >= float(signal_cfg["minimum_absolute_mean_nll_delta"])
                    and mean_kl >= float(signal_cfg["minimum_mean_teacher_forced_kl"])
                    and nonzero_fraction >= float(signal_cfg["minimum_nonzero_nll_delta_fraction"])
                )
                if effect["clears_calibration_signal"]:
                    signal_conditions.append(name)
            effects[name] = effect
        methods[method_id] = {
            "controls": method_controls,
            "controls_pass": controls_pass,
            "signal_conditions": signal_conditions,
            "signal_pass": bool(signal_conditions),
            "effects": effects,
        }
    signal_pass = all(methods[method]["signal_pass"] for method in EXPECTED_METHODS)
    decision = (
        "HOLD_A0_CONTROLS"
        if not all_controls
        else "PASS_A0_SIGNAL"
        if signal_pass
        else "KILL_DETERMINISTIC_EFFECT_BRANCH"
    )
    return {
        "decision": decision,
        "controls_pass": all_controls,
        "signal_pass": signal_pass,
        "methods": methods,
    }


def _load_existing(path: Path, *, protocol_id: str, config_sha: str, implementation_sha: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if (
            record.get("protocol_id") != protocol_id
            or record.get("config_file_sha256") != config_sha
            or record.get("implementation_sha256") != implementation_sha
        ):
            raise ValueError("existing A0 record provenance mismatch")
        records.append(record)
    keys = [(record["method_id"], record["selection_rank"]) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("existing A0 records contain duplicate method/prompt keys")
    return records


def run_a0(root: Path, config_path: Path, records_path: Path) -> dict[str, Any]:
    root = root.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    prompt_manifest = validate_config(root, config)
    config_sha = _sha256(config_path)
    implementation_sha = _sha256(Path(__file__))
    records = _load_existing(
        records_path,
        protocol_id=config["protocol_id"],
        config_sha=config_sha,
        implementation_sha=implementation_sha,
    )
    completed = {(record["method_id"], record["selection_rank"]) for record in records}
    prompts = [
        row
        for row in prompt_manifest["records"]
        if row["split"] == config["prompt_manifest"]["split"]
    ]
    records_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    runtime: dict[str, Any] = {}

    for method_id in EXPECTED_METHODS:
        pending = [row for row in prompts if (method_id, row["selection_rank"]) not in completed]
        if not pending:
            continue
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        method_started = time.perf_counter()
        if method_id == "coconut_gpt2":
            from research.coordinate_invariance.charts import AffineChart
            from research.coordinate_invariance.real_models.coconut import (
                CoconutChartRunner,
                load_public_coconut,
            )

            method = config["methods"][method_id]
            source = _inside(root, method["source_path"])
            if _git(source, "rev-parse", "HEAD") != method["source_commit"]:
                raise ValueError("Coconut source commit mismatch")
            bundle = load_public_coconut(
                checkpoint_path=_inside(root, method["checkpoint_path"]),
                coconut_source_directory=source,
                model_id=method["base_model_id"],
                model_revision=method["base_model_revision"],
                expected_checkpoint_sha256=method["checkpoint_sha256"],
                cache_directory=_inside(root, method["base_model_cache"]),
                device="cuda:0",
            )
            chart = AffineChart.identity(
                int(bundle.model.embedding.weight.shape[-1]),
                dtype=torch.float64,
                compute_dtype=torch.float64,
            )
            runner = CoconutChartRunner(bundle.model, chart)
            prompt_runner = lambda row: _run_coconut_prompt(bundle, runner, row, config)
            owned = bundle
        else:
            owned = _load_codi(root, config)
            prompt_runner = lambda row: _run_codi_prompt(owned, row, config)

        with records_path.open("a", encoding="utf-8", newline="\n") as handle:
            for row in pending:
                record = prompt_runner(row)
                record.update(
                    {
                        "protocol_id": config["protocol_id"],
                        "config_file_sha256": config_sha,
                        "implementation_sha256": implementation_sha,
                    }
                )
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                handle.flush()
                records.append(record)
        torch.cuda.synchronize()
        runtime[method_id] = {
            "seconds": time.perf_counter() - method_started,
            "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        }
        del owned
        torch.cuda.empty_cache()

    summary = summarize_records(records, config)
    summary.update(
        {
            "protocol_id": config["protocol_id"],
            "config_file_sha256": config_sha,
            "config_canonical_sha256": _canonical_sha256(config),
            "implementation_sha256": implementation_sha,
            "prompt_manifest_sha256": config["prompt_manifest"]["sha256"],
            "stage": "deterministic_intervention_a0_calibration",
            "split": config["prompt_manifest"]["split"],
            "record_count": len(records),
            "gpu_used": True,
            "training_used": False,
            "runtime": {
                "methods": runtime,
                "total_seconds_this_invocation": time.perf_counter() - started,
                "cuda_device": torch.cuda.get_device_name(0),
            },
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "platform": platform.platform(),
            },
            "scientific_evidence": "calibration_only",
            "next_required_action": (
                "freeze untouched confirmation and stochastic policy-contract preflight"
                if summary["decision"] == "PASS_A0_SIGNAL"
                else "repair controls without inspecting effects"
                if summary["decision"] == "HOLD_A0_CONTROLS"
                else "stop deterministic effect branch and prioritize stochastic contract evidence"
            ),
        }
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run_a0(args.workspace, args.config, args.records)
    except Exception as exc:
        report = {
            "protocol_id": json.loads(args.config.read_text(encoding="utf-8")).get("protocol_id"),
            "stage": "deterministic_intervention_a0_calibration",
            "decision": "HOLD_A0_CONTROLS",
            "failures": [f"{type(exc).__name__}: {exc}"],
            "training_used": False,
            "scientific_evidence": False,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "failures": report.get("failures", [])}))
    return 0 if report["decision"] == "PASS_A0_SIGNAL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
