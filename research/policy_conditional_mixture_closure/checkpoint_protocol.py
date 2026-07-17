"""Frozen protocol validation and deterministic work allocation for PCMC gates."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


SHA256_PATTERN = re.compile(r"^[0-9A-F]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
STAGES = (
    "A0_ONE_STEP_CLOSURE",
    "A1_CAUSAL_CONTINUATION",
    "B0_CONSTRAINED_ORACLE",
    "B1_POSTERIOR_AWARE_SEQUENCE",
)
METHODS = ("arithmetic", "randomized_hard", "top1", "sharpened")


@dataclass(frozen=True)
class PromptProfile:
    message_layout: str
    system_prompt: str
    add_generation_prompt: bool
    required_rendered_suffix: str


@dataclass(frozen=True)
class SamplerProfile:
    top_p: float
    top_k: int
    max_topk: int
    temperature: float
    gumbel_softmax_temperature: float
    noise_kind: str
    noise_scale: float
    gumbel_clip_min: float
    gumbel_clip_max: float


@dataclass(frozen=True)
class CheckpointProfile:
    checkpoint_id: str
    checkpoint_path: str
    checkpoint_sha256: str
    source_path: str
    source_commit: str
    adapter: str
    model_type: str
    structural_end_token_id: int | None
    maximum_structural_end_mass: float
    prompt: PromptProfile
    sampler: SamplerProfile


@dataclass(frozen=True)
class ProtocolAudit:
    protocol_id: str
    protocol_sha256: str
    checkpoint_ids: tuple[str, ...]
    example_count: int
    calibration_count: int
    confirmation_count: int
    continuation_replicates: int
    maximum_a0_records: int
    maximum_a1_records: int


def load_protocol(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    data = json.loads(payload.decode("utf-8"))
    if not isinstance(data, dict):
        raise TypeError("protocol root must be a JSON object")
    data["_protocol_sha256"] = hashlib.sha256(payload).hexdigest()
    return data


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    return value


def _require_relative_path(value: Any, name: str) -> str:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must stay inside the workspace")
    return path.as_posix()


def _require_probability(value: Any, name: str, *, allow_zero: bool = True) -> float:
    result = float(value)
    lower_ok = result >= 0.0 if allow_zero else result > 0.0
    if not lower_ok or result > 1.0:
        raise ValueError(f"{name} must be a probability")
    return result


def checkpoint_profiles(protocol: dict[str, Any]) -> tuple[CheckpointProfile, ...]:
    raw_profiles = protocol.get("checkpoints")
    if not isinstance(raw_profiles, list) or len(raw_profiles) != 2:
        raise ValueError("v1 requires exactly two checkpoint profiles")
    profiles: list[CheckpointProfile] = []
    for raw_value in raw_profiles:
        raw = _require_mapping(raw_value, "checkpoint profile")
        prompt_raw = _require_mapping(raw.get("prompt"), "checkpoint prompt")
        sampler_raw = _require_mapping(raw.get("sampler"), "checkpoint sampler")
        checkpoint_sha256 = str(raw["checkpoint_sha256"])
        source_commit = str(raw["source_commit"])
        if not SHA256_PATTERN.fullmatch(checkpoint_sha256):
            raise ValueError("checkpoint_sha256 must be uppercase SHA-256")
        if not COMMIT_PATTERN.fullmatch(source_commit):
            raise ValueError("source_commit must be a full lowercase git hash")
        structural = raw.get("structural_end_token_id")
        if structural is not None and int(structural) < 0:
            raise ValueError("structural_end_token_id must be nonnegative or null")
        prompt = PromptProfile(
            message_layout=str(prompt_raw["message_layout"]),
            system_prompt=str(prompt_raw["system_prompt"]),
            add_generation_prompt=bool(prompt_raw["add_generation_prompt"]),
            required_rendered_suffix=str(prompt_raw["required_rendered_suffix"]),
        )
        if prompt.message_layout != "system_then_user":
            raise ValueError("v1 prompt layout must remain system_then_user")
        if not prompt.system_prompt or not prompt.add_generation_prompt:
            raise ValueError("checkpoint prompt contract is incomplete")
        if prompt.required_rendered_suffix != "<think>":
            raise ValueError("checkpoint prompt must enter the thinking region")
        sampler = SamplerProfile(
            top_p=_require_probability(
                sampler_raw["top_p"], "sampler top_p", allow_zero=False
            ),
            top_k=int(sampler_raw["top_k"]),
            max_topk=int(sampler_raw["max_topk"]),
            temperature=float(sampler_raw["temperature"]),
            gumbel_softmax_temperature=float(
                sampler_raw["gumbel_softmax_temperature"]
            ),
            noise_kind=str(sampler_raw["noise_kind"]),
            noise_scale=float(sampler_raw["noise_scale"]),
            gumbel_clip_min=float(sampler_raw["gumbel_clip_min"]),
            gumbel_clip_max=float(sampler_raw["gumbel_clip_max"]),
        )
        if sampler.top_k < 1 or sampler.max_topk < 2:
            raise ValueError("sampler top-k support is invalid")
        if sampler.max_topk > sampler.top_k:
            raise ValueError("sampler max_topk cannot exceed top_k")
        if sampler.temperature <= 0.0 or sampler.gumbel_softmax_temperature <= 0.0:
            raise ValueError("sampler temperatures must be positive")
        if sampler.noise_scale < 0.0:
            raise ValueError("sampler noise scale must be nonnegative")
        if not sampler.gumbel_clip_min < sampler.gumbel_clip_max:
            raise ValueError("sampler Gumbel clipping interval is invalid")
        profiles.append(
            CheckpointProfile(
                checkpoint_id=str(raw["checkpoint_id"]),
                checkpoint_path=_require_relative_path(
                    raw["checkpoint_path"], "checkpoint_path"
                ),
                checkpoint_sha256=checkpoint_sha256,
                source_path=_require_relative_path(raw["source_path"], "source_path"),
                source_commit=source_commit,
                adapter=str(raw["adapter"]),
                model_type=str(raw["model_type"]),
                structural_end_token_id=(
                    None if structural is None else int(structural)
                ),
                maximum_structural_end_mass=_require_probability(
                    raw["maximum_structural_end_mass"],
                    "maximum_structural_end_mass",
                ),
                prompt=prompt,
                sampler=sampler,
            )
        )
    ids = [profile.checkpoint_id for profile in profiles]
    if len(set(ids)) != len(ids) or any(not value for value in ids):
        raise ValueError("checkpoint ids must be unique and non-empty")
    return tuple(profiles)


def validate_protocol(protocol: dict[str, Any]) -> ProtocolAudit:
    required = {
        "protocol_id",
        "scientific_status",
        "stage_order",
        "dataset",
        "checkpoints",
        "natural_action",
        "gate_a0",
        "gate_a1",
        "gate_b0",
        "gate_b1",
        "execution",
    }
    missing = sorted(required - protocol.keys())
    if missing:
        raise ValueError(f"protocol is missing fields: {missing}")
    if tuple(protocol["stage_order"]) != STAGES:
        raise ValueError("stage order must remain fail-closed")

    dataset = _require_mapping(protocol["dataset"], "dataset")
    example_count = int(dataset["example_count"])
    calibration_count = int(dataset["calibration_count"])
    if example_count != 500 or calibration_count != 250:
        raise ValueError("v1 requires a frozen 250/250 split of MATH-500")
    if str(dataset["revision"]) not in str(dataset["arrow_path"]):
        raise ValueError("dataset path must contain the pinned revision")
    _require_relative_path(dataset["arrow_path"], "dataset arrow_path")
    if not SHA256_PATTERN.fullmatch(str(dataset["arrow_sha256"])):
        raise ValueError("dataset arrow_sha256 must be uppercase SHA-256")
    if not str(dataset["split_salt"]):
        raise ValueError("split_salt must be non-empty")

    profiles = checkpoint_profiles(protocol)
    natural = _require_mapping(protocol["natural_action"], "natural_action")
    if int(natural["events_per_prompt"]) != 1:
        raise ValueError("v1 uses exactly one label-blind natural event per prompt")
    if int(natural["minimum_content_support"]) < 2:
        raise ValueError("PCMC requires at least two content components")
    if int(natural.get("maximum_prompt_tokens", 0)) != 1024:
        raise ValueError("v1 prompt cap must remain source-native at 1024 tokens")
    if str(natural.get("structural_end_handling", "")) != (
        "measure original mass; exclude structural component; renormalize content "
        "weights; mark ineligible above cap"
    ):
        raise ValueError("structural end handling must remain content-conditioned")

    expected_adapters = {
        "official_latent_grpo_llama": {
            "structural_end_token_id": 524,
            "noise_kind": "one_sided_gumbel_on_full_log_probs",
            "top_p": 0.95,
            "top_k": 30,
            "max_topk": 10,
            "temperature": 0.6,
            "gumbel_softmax_temperature": 1.0,
        },
        "soft_grpo_qwen": {
            "structural_end_token_id": 151649,
            "noise_kind": "gumbel_on_truncated_log_probs",
            "top_p": 0.95,
            "top_k": 5,
            "max_topk": 5,
            "temperature": 1.0,
            "gumbel_softmax_temperature": 0.1,
        },
    }
    for profile in profiles:
        expected = expected_adapters.get(profile.adapter)
        if expected is None:
            raise ValueError(f"unsupported checkpoint adapter: {profile.adapter}")
        observed = {
            "structural_end_token_id": profile.structural_end_token_id,
            "noise_kind": profile.sampler.noise_kind,
            "top_p": profile.sampler.top_p,
            "top_k": profile.sampler.top_k,
            "max_topk": profile.sampler.max_topk,
            "temperature": profile.sampler.temperature,
            "gumbel_softmax_temperature": profile.sampler.gumbel_softmax_temperature,
        }
        if observed != expected:
            raise ValueError(f"source-native sampler drift for {profile.checkpoint_id}")

    gate_a0 = _require_mapping(protocol["gate_a0"], "gate_a0")
    high_quantile = _require_probability(
        gate_a0["high_gap_quantile"], "high_gap_quantile", allow_zero=False
    )
    low_quantile = _require_probability(
        gate_a0["low_gap_quantile"], "low_gap_quantile", allow_zero=False
    )
    if not low_quantile < high_quantile:
        raise ValueError("low gap quantile must be below high gap quantile")
    _require_probability(
        gate_a0["minimum_calibration_eligible_rate"],
        "minimum_calibration_eligible_rate",
    )
    if float(gate_a0["minimum_calibration_high_gap_js_nats"]) <= 0.0:
        raise ValueError("minimum JS threshold must be positive")
    if int(gate_a0["minimum_confirmation_high_gap_prompts"]) < 32:
        raise ValueError("confirmation high-gap floor is underpowered")
    high_count = int(gate_a0["maximum_selected_high_gap_prompts"])
    low_count = int(gate_a0["maximum_selected_low_gap_prompts"])
    if high_count < int(gate_a0["minimum_confirmation_high_gap_prompts"]):
        raise ValueError("selected high-gap cap is below the PASS floor")

    gate_a1 = _require_mapping(protocol["gate_a1"], "gate_a1")
    if tuple(gate_a1["methods"]) != METHODS:
        raise ValueError("all four frozen intervention methods are required")
    continuation_seeds = [int(value) for value in gate_a1["continuation_seed_bases"]]
    if len(continuation_seeds) != 8 or len(set(continuation_seeds)) != 8:
        raise ValueError("v1 requires eight unique continuation seeds")
    if int(gate_a1["bootstrap_replicates"]) < 10000:
        raise ValueError("scientific analysis requires at least 10000 bootstraps")
    for name in (
        "minimum_high_gap_accuracy_delta",
        "minimum_gap_reward_spearman",
        "minimum_partial_spearman",
        "minimum_randomized_hard_over_top1",
        "minimum_randomized_hard_over_sharpened",
    ):
        if float(gate_a1[name]) <= 0.0:
            raise ValueError(f"{name} must be positive")

    gate_b0 = _require_mapping(protocol["gate_b0"], "gate_b0")
    if int(gate_b0["events_per_checkpoint"]) < 50:
        raise ValueError("B0 requires at least 50 events per checkpoint")
    if str(gate_b0["optimizer"]) != "projected_adam":
        raise ValueError("B0 optimizer must remain projected Adam")
    _require_probability(
        gate_b0["minimum_median_kl_reduction_fraction"],
        "minimum_median_kl_reduction_fraction",
    )
    _require_probability(
        gate_b0["minimum_improved_event_fraction"],
        "minimum_improved_event_fraction",
    )

    gate_b1 = _require_mapping(protocol["gate_b1"], "gate_b1")
    if "posterior-updated" not in str(gate_b1["target"]):
        raise ValueError("B1 must use posterior-updated branch weights")
    if int(gate_b1["maximum_visible_horizon"]) < 8:
        raise ValueError("B1 horizon is too short")

    execution = _require_mapping(protocol["execution"], "execution")
    if execution.get("fail_closed") is not True:
        raise ValueError("execution must fail closed")
    if execution.get("shutdown_on_success_or_failure") is not True:
        raise ValueError("managed execution must always request shutdown")
    if int(execution["minimum_free_disk_bytes"]) < 50 * 1024**3:
        raise ValueError("minimum free disk must remain at least 50 GiB")

    confirmation_count = example_count - calibration_count
    maximum_a0_records = len(profiles) * example_count
    maximum_a1_records = (
        len(profiles)
        * (high_count + low_count)
        * len(METHODS)
        * len(continuation_seeds)
    )
    return ProtocolAudit(
        protocol_id=str(protocol["protocol_id"]),
        protocol_sha256=str(protocol.get("_protocol_sha256", "UNHASHED")),
        checkpoint_ids=tuple(profile.checkpoint_id for profile in profiles),
        example_count=example_count,
        calibration_count=calibration_count,
        confirmation_count=confirmation_count,
        continuation_replicates=len(continuation_seeds),
        maximum_a0_records=maximum_a0_records,
        maximum_a1_records=maximum_a1_records,
    )


def example_ids(protocol: dict[str, Any]) -> tuple[str, ...]:
    count = int(protocol["dataset"]["example_count"])
    return tuple(f"math500:{index:03d}" for index in range(count))


def _stable_order(values: Iterable[str], salt: str) -> list[str]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(f"{salt}|{value}".encode("utf-8")).digest(),
    )


def split_example_ids(protocol: dict[str, Any]) -> dict[str, str]:
    validate_protocol(protocol)
    dataset = protocol["dataset"]
    ordered = _stable_order(example_ids(protocol), str(dataset["split_salt"]))
    calibration_count = int(dataset["calibration_count"])
    calibration = set(ordered[:calibration_count])
    return {
        example_id: (
            "calibration" if example_id in calibration else "confirmation"
        )
        for example_id in example_ids(protocol)
    }


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("quantile values must be finite")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def selected_confirmation_ids(
    protocol: dict[str, Any],
    a0_records: Sequence[dict[str, Any]],
    *,
    checkpoint_id: str,
) -> dict[str, Any]:
    """Select confirmation work using A0 geometry only, never reward labels."""

    splits = split_example_ids(protocol)
    records = [
        record
        for record in a0_records
        if record.get("checkpoint_id") == checkpoint_id
        and record.get("record_type") == "a0_event"
        and record.get("status") == "COMPLETE"
    ]
    by_id = {str(record["example_id"]): record for record in records}
    if len(by_id) != len(records):
        raise ValueError("duplicate complete A0 records")
    calibration = [
        record
        for example_id, record in by_id.items()
        if splits.get(example_id) == "calibration"
    ]
    if not calibration:
        raise ValueError("no complete calibration A0 records")
    gaps = [float(record["js_branch_arithmetic_nats"]) for record in calibration]
    gate_a0 = protocol["gate_a0"]
    high_threshold = quantile(gaps, float(gate_a0["high_gap_quantile"]))
    low_threshold = quantile(gaps, float(gate_a0["low_gap_quantile"]))
    confirmation = [
        record
        for example_id, record in by_id.items()
        if splits.get(example_id) == "confirmation"
    ]
    high_candidates = [
        str(record["example_id"])
        for record in confirmation
        if float(record["js_branch_arithmetic_nats"]) >= high_threshold
    ]
    low_candidates = [
        str(record["example_id"])
        for record in confirmation
        if float(record["js_branch_arithmetic_nats"]) <= low_threshold
    ]
    salt = str(gate_a0["selection_salt"])
    selected_high = _stable_order(high_candidates, f"{salt}|{checkpoint_id}|high")[
        : int(gate_a0["maximum_selected_high_gap_prompts"])
    ]
    selected_low = _stable_order(low_candidates, f"{salt}|{checkpoint_id}|low")[
        : int(gate_a0["maximum_selected_low_gap_prompts"])
    ]
    return {
        "high_threshold": high_threshold,
        "low_threshold": low_threshold,
        "calibration_complete_count": len(calibration),
        "confirmation_complete_count": len(confirmation),
        "high_candidate_count": len(high_candidates),
        "low_candidate_count": len(low_candidates),
        "selected_high": selected_high,
        "selected_low": selected_low,
    }


def action_seed(
    protocol: dict[str, Any], checkpoint_index: int, example_index: int
) -> int:
    config = protocol["natural_action"]
    return (
        int(config["action_seed_base"])
        + checkpoint_index * int(config["checkpoint_seed_stride"])
        + example_index * int(config["example_seed_stride"])
    )


def continuation_seed(
    protocol: dict[str, Any],
    checkpoint_index: int,
    example_index: int,
    replicate_index: int,
) -> int:
    config = protocol["gate_a1"]
    bases = [int(value) for value in config["continuation_seed_bases"]]
    return (
        bases[replicate_index]
        + checkpoint_index * int(config["checkpoint_seed_stride"])
        + example_index * int(config["example_seed_stride"])
    )


def audit_as_dict(protocol: dict[str, Any]) -> dict[str, Any]:
    return asdict(validate_protocol(protocol))
