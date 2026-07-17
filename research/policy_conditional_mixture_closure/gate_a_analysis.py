"""Fail-closed offline analysis for the frozen PCMC checkpoint Gate A."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Sequence

import numpy as np

from .checkpoint_protocol import (
    METHODS,
    action_seed,
    checkpoint_profiles,
    continuation_seed,
    example_ids,
    load_protocol,
    quantile,
    selected_confirmation_ids,
    split_example_ids,
    validate_protocol,
)


def a0_record_key(checkpoint_id: str, example_id: str) -> str:
    return f"a0|{checkpoint_id}|{example_id}"


def a1_record_key(
    checkpoint_id: str, example_id: str, method: str, replicate_index: int
) -> str:
    if method not in METHODS:
        raise ValueError("unknown intervention method")
    return f"a1|{checkpoint_id}|{example_id}|{method}|r{replicate_index}"


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"JSONL line {line_number} is not an object")
            records.append(value)
    return records


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_record_set(
    protocol: dict[str, Any], records: Sequence[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    audit = validate_protocol(protocol)
    profile_ids = set(audit.checkpoint_ids)
    profiles = checkpoint_profiles(protocol)
    profile_by_id = {profile.checkpoint_id: profile for profile in profiles}
    checkpoint_indices = {
        profile.checkpoint_id: index for index, profile in enumerate(profiles)
    }
    valid_examples = set(example_ids(protocol))
    example_indices = {
        example_id: index for index, example_id in enumerate(example_ids(protocol))
    }
    frozen_splits = split_example_ids(protocol)
    protocol_hash = audit.protocol_sha256
    by_key: dict[str, dict[str, Any]] = {}
    for record in records:
        key = str(record.get("key", ""))
        if not key:
            raise ValueError("every record requires a key")
        if key in by_key:
            raise ValueError(f"duplicate record key: {key}")
        if record.get("protocol_id") != audit.protocol_id:
            raise ValueError(f"record protocol id mismatch: {key}")
        if record.get("protocol_sha256") != protocol_hash:
            raise ValueError(f"record protocol hash mismatch: {key}")
        checkpoint_id = str(record.get("checkpoint_id", ""))
        example_id = str(record.get("example_id", ""))
        if checkpoint_id not in profile_ids or example_id not in valid_examples:
            raise ValueError(f"record identity is outside the frozen protocol: {key}")
        if record.get("dataset_partition") != frozen_splits[example_id]:
            raise ValueError(f"record dataset partition mismatch: {key}")

        record_type = record.get("record_type")
        if record_type == "a0_event":
            expected_key = a0_record_key(checkpoint_id, example_id)
            if key != expected_key:
                raise ValueError(f"A0 key mismatch: {key}")
            expected_seed = action_seed(
                protocol,
                checkpoint_indices[checkpoint_id],
                example_indices[example_id],
            )
            if int(record.get("action_seed", -1)) != expected_seed:
                raise ValueError(f"A0 action seed mismatch: {key}")
            status = record.get("status")
            if status not in {"COMPLETE", "INELIGIBLE_CONTENT"}:
                raise ValueError(f"invalid A0 status: {key}")
            if status == "COMPLETE":
                js = _finite(record.get("js_branch_arithmetic_nats"), "A0 JS")
                if not 0.0 <= js <= math.log(2.0) + 1e-9:
                    raise ValueError(f"A0 JS is outside [0, log(2)]: {key}")
                entropy = _finite(
                    record.get("weight_entropy_normalized"), "weight entropy"
                )
                max_weight = _finite(record.get("maximum_weight"), "maximum weight")
                effective_support = _finite(
                    record.get("effective_support"), "effective support"
                )
                structural_mass = _finite(
                    record.get("structural_end_mass"), "structural end mass"
                )
                if not 0.0 <= entropy <= 1.0 + 1e-9:
                    raise ValueError(f"normalized entropy is invalid: {key}")
                if not 0.0 <= max_weight <= 1.0:
                    raise ValueError(f"maximum weight is invalid: {key}")
                if effective_support < 1.0 or not 0.0 <= structural_mass <= 1.0:
                    raise ValueError(f"support diagnostics are invalid: {key}")
                if structural_mass > profile_by_id[
                    checkpoint_id
                ].maximum_structural_end_mass + 1e-12:
                    raise ValueError(f"complete A0 record exceeds end-mass cap: {key}")
                if int(record.get("content_support_size", 0)) < 2:
                    raise ValueError(f"complete A0 record has singleton content: {key}")
                if int(record.get("prompt_token_count", 0)) < 1:
                    raise ValueError(f"prompt token count is invalid: {key}")
                if int(record.get("latent_step_index", -1)) < 0:
                    raise ValueError(f"latent step index is invalid: {key}")
        elif record_type == "a1_continuation":
            if record.get("status") != "COMPLETE":
                raise ValueError(f"A1 records must be complete or absent: {key}")
            method = str(record.get("method", ""))
            replicate_index = int(record.get("replicate_index", -1))
            if method not in METHODS or not 0 <= replicate_index < audit.continuation_replicates:
                raise ValueError(f"A1 method or replicate is invalid: {key}")
            if key != a1_record_key(
                checkpoint_id, example_id, method, replicate_index
            ):
                raise ValueError(f"A1 key mismatch: {key}")
            expected_seed = continuation_seed(
                protocol,
                checkpoint_indices[checkpoint_id],
                example_indices[example_id],
                replicate_index,
            )
            if int(record.get("continuation_seed", -1)) != expected_seed:
                raise ValueError(f"A1 continuation seed mismatch: {key}")
            correct = record.get("correct")
            if correct not in {0, 1, False, True}:
                raise ValueError(f"A1 correctness must be binary: {key}")
            response_hash = str(record.get("response_sha256", ""))
            if len(response_hash) != 64 or any(
                character not in "0123456789abcdef" for character in response_hash
            ):
                raise ValueError(f"A1 response hash is invalid: {key}")
        else:
            raise ValueError(f"unknown record type: {key}")
        by_key[key] = record
    return by_key


def _rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("rankdata requires a finite non-empty vector")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.size, dtype=np.float64)
    position = 0
    while position < array.size:
        end = position + 1
        while end < array.size and array[order[end]] == array[order[position]]:
            end += 1
        average_rank = (position + end - 1) / 2.0 + 1.0
        ranks[order[position:end]] = average_rank
        position = end
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 3:
        raise ValueError("spearman requires equal vectors with at least three items")
    return _pearson(_rankdata(left), _rankdata(right))


def partial_spearman(
    left: Sequence[float],
    right: Sequence[float],
    controls: Sequence[Sequence[float]],
) -> float:
    if len(left) != len(right) or len(left) < 5:
        raise ValueError("partial spearman requires at least five paired items")
    if any(len(control) != len(left) for control in controls):
        raise ValueError("all controls must match the outcome length")
    ranked_left = _rankdata(left)
    ranked_right = _rankdata(right)
    columns = [np.ones(len(left), dtype=np.float64)]
    columns.extend(_rankdata(control) for control in controls)
    design = np.column_stack(columns)
    left_fit = design @ np.linalg.lstsq(design, ranked_left, rcond=None)[0]
    right_fit = design @ np.linalg.lstsq(design, ranked_right, rcond=None)[0]
    return _pearson(ranked_left - left_fit, ranked_right - right_fit)


def _bootstrap_interval(
    item_count: int,
    metric: Callable[[np.ndarray], float],
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    if item_count < 2 or replicates < 100:
        raise ValueError("bootstrap requires at least two items and 100 replicates")
    generator = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(replicates):
        indices = generator.integers(0, item_count, size=item_count)
        value = float(metric(indices))
        if math.isfinite(value):
            values.append(value)
    if len(values) < int(0.8 * replicates):
        raise ValueError("too many invalid bootstrap replicates")
    return quantile(values, 0.025), quantile(values, 0.975)


def _checkpoint_a0(
    protocol: dict[str, Any],
    by_key: dict[str, dict[str, Any]],
    checkpoint_id: str,
) -> dict[str, Any]:
    splits = split_example_ids(protocol)
    expected = [a0_record_key(checkpoint_id, example_id) for example_id in example_ids(protocol)]
    missing = [key for key in expected if key not in by_key]
    if missing:
        return {
            "decision": "BLOCKED_INCOMPLETE_A0",
            "missing_record_count": len(missing),
            "first_missing_key": missing[0],
        }
    records = [by_key[key] for key in expected]
    calibration = [
        record
        for record in records
        if splits[str(record["example_id"])] == "calibration"
    ]
    complete_calibration = [
        record for record in calibration if record["status"] == "COMPLETE"
    ]
    eligible_rate = len(complete_calibration) / len(calibration)
    if not complete_calibration:
        return {
            "decision": "KILL_A0_NO_NATURAL_CONTENT_MIXTURES",
            "calibration_eligible_rate": eligible_rate,
        }
    selection = selected_confirmation_ids(
        protocol, records, checkpoint_id=checkpoint_id
    )
    gate = protocol["gate_a0"]
    reasons: list[str] = []
    if eligible_rate < float(gate["minimum_calibration_eligible_rate"]):
        reasons.append("calibration eligible rate below frozen minimum")
    if selection["high_threshold"] < float(
        gate["minimum_calibration_high_gap_js_nats"]
    ):
        reasons.append("calibration high-gap JS below frozen minimum")
    if selection["high_candidate_count"] < int(
        gate["minimum_confirmation_high_gap_prompts"]
    ):
        reasons.append("too few confirmation high-gap prompts")
    decision = "KILL_A0" if reasons else "ADVANCE_A1"
    return {
        "decision": decision,
        "decision_reasons": reasons,
        "calibration_eligible_rate": eligible_rate,
        **selection,
    }


def _method_means(
    by_key: dict[str, dict[str, Any]],
    checkpoint_id: str,
    selected_ids: Sequence[str],
    replicates: int,
) -> tuple[dict[str, dict[str, float]], list[str]]:
    missing: list[str] = []
    result: dict[str, dict[str, float]] = {}
    for example_id in selected_ids:
        result[example_id] = {}
        for method in METHODS:
            values: list[float] = []
            for replicate_index in range(replicates):
                key = a1_record_key(
                    checkpoint_id, example_id, method, replicate_index
                )
                record = by_key.get(key)
                if record is None:
                    missing.append(key)
                else:
                    values.append(float(bool(record["correct"])))
            if len(values) == replicates:
                result[example_id][method] = mean(values)
    return result, missing


def _checkpoint_a1(
    protocol: dict[str, Any],
    by_key: dict[str, dict[str, Any]],
    checkpoint_id: str,
    a0_summary: dict[str, Any],
    *,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    selected_high = list(a0_summary["selected_high"])
    selected_low = list(a0_summary["selected_low"])
    selected = selected_high + selected_low
    replicates = validate_protocol(protocol).continuation_replicates
    method_means, missing = _method_means(
        by_key, checkpoint_id, selected, replicates
    )
    if missing:
        return {
            "decision": "BLOCKED_INCOMPLETE_A1",
            "missing_record_count": len(missing),
            "first_missing_key": missing[0],
            "selected_high_count": len(selected_high),
            "selected_low_count": len(selected_low),
        }

    high_hard_delta = np.asarray(
        [
            method_means[example_id]["randomized_hard"]
            - method_means[example_id]["arithmetic"]
            for example_id in selected_high
        ],
        dtype=np.float64,
    )
    high_hard_top1 = np.asarray(
        [
            method_means[example_id]["randomized_hard"]
            - method_means[example_id]["top1"]
            for example_id in selected_high
        ],
        dtype=np.float64,
    )
    high_hard_sharpened = np.asarray(
        [
            method_means[example_id]["randomized_hard"]
            - method_means[example_id]["sharpened"]
            for example_id in selected_high
        ],
        dtype=np.float64,
    )
    a0_records = {
        example_id: by_key[a0_record_key(checkpoint_id, example_id)]
        for example_id in selected
    }
    gaps = np.asarray(
        [float(a0_records[example_id]["js_branch_arithmetic_nats"]) for example_id in selected]
    )
    reward_losses = np.asarray(
        [
            method_means[example_id]["randomized_hard"]
            - method_means[example_id]["arithmetic"]
            for example_id in selected
        ]
    )
    controls = [
        np.asarray(
            [float(a0_records[example_id][field]) for example_id in selected]
        )
        for field in (
            "weight_entropy_normalized",
            "maximum_weight",
            "effective_support",
            "prompt_token_count",
        )
    ]
    gap_reward_spearman = spearman(gaps, reward_losses)
    controlled_spearman = partial_spearman(gaps, reward_losses, controls)

    seed = int(protocol["gate_a1"]["bootstrap_seed"]) + int(
        hashlib.sha256(checkpoint_id.encode("utf-8")).hexdigest()[:8], 16
    )
    delta_ci = _bootstrap_interval(
        len(high_hard_delta),
        lambda indices: float(high_hard_delta[indices].mean()),
        replicates=bootstrap_replicates,
        seed=seed,
    )
    correlation_ci = _bootstrap_interval(
        len(selected),
        lambda indices: spearman(gaps[indices], reward_losses[indices]),
        replicates=bootstrap_replicates,
        seed=seed + 1,
    )
    partial_ci = _bootstrap_interval(
        len(selected),
        lambda indices: partial_spearman(
            gaps[indices],
            reward_losses[indices],
            [control[indices] for control in controls],
        ),
        replicates=bootstrap_replicates,
        seed=seed + 2,
    )

    gate = protocol["gate_a1"]
    metrics = {
        "high_gap_accuracy_delta": float(high_hard_delta.mean()),
        "high_gap_accuracy_delta_ci95": list(delta_ci),
        "gap_reward_spearman": gap_reward_spearman,
        "gap_reward_spearman_ci95": list(correlation_ci),
        "partial_spearman": controlled_spearman,
        "partial_spearman_ci95": list(partial_ci),
        "randomized_hard_over_top1": float(high_hard_top1.mean()),
        "randomized_hard_over_sharpened": float(high_hard_sharpened.mean()),
    }
    checks = {
        "accuracy_effect": metrics["high_gap_accuracy_delta"]
        >= float(gate["minimum_high_gap_accuracy_delta"]),
        "accuracy_ci": delta_ci[0]
        > float(gate["minimum_high_gap_delta_ci_lower"]),
        "gap_reward_association": gap_reward_spearman
        >= float(gate["minimum_gap_reward_spearman"]),
        "gap_reward_ci": correlation_ci[0]
        > float(gate["minimum_gap_reward_spearman_ci_lower"]),
        "controlled_association": controlled_spearman
        >= float(gate["minimum_partial_spearman"]),
        "controlled_ci": partial_ci[0]
        > float(gate["minimum_partial_spearman_ci_lower"]),
        "beats_top1": metrics["randomized_hard_over_top1"]
        >= float(gate["minimum_randomized_hard_over_top1"]),
        "beats_sharpened": metrics["randomized_hard_over_sharpened"]
        >= float(gate["minimum_randomized_hard_over_sharpened"]),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "decision": "PASS_A1" if not failed else "KILL_A1",
        "failed_checks": failed,
        "selected_high_count": len(selected_high),
        "selected_low_count": len(selected_low),
        "metrics": metrics,
        "checks": checks,
    }


def summarize_gate_a(
    protocol: dict[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    completed_stage: str,
    bootstrap_replicates: int | None = None,
) -> dict[str, Any]:
    if completed_stage not in {"A0", "A1"}:
        raise ValueError("completed_stage must be A0 or A1")
    audit = validate_protocol(protocol)
    by_key = _validate_record_set(protocol, records)
    allowed_a1_keys: set[str] = set()
    checkpoint_results: dict[str, Any] = {}
    for profile in checkpoint_profiles(protocol):
        a0 = _checkpoint_a0(protocol, by_key, profile.checkpoint_id)
        result: dict[str, Any] = {"a0": a0}
        if a0["decision"] == "ADVANCE_A1":
            for example_id in a0["selected_high"] + a0["selected_low"]:
                for method in METHODS:
                    for replicate_index in range(audit.continuation_replicates):
                        allowed_a1_keys.add(
                            a1_record_key(
                                profile.checkpoint_id,
                                example_id,
                                method,
                                replicate_index,
                            )
                        )
            if completed_stage == "A1":
                result["a1"] = _checkpoint_a1(
                    protocol,
                    by_key,
                    profile.checkpoint_id,
                    a0,
                    bootstrap_replicates=(
                        int(protocol["gate_a1"]["bootstrap_replicates"])
                        if bootstrap_replicates is None
                        else bootstrap_replicates
                    ),
                )
        checkpoint_results[profile.checkpoint_id] = result

    actual_a1_keys = {
        key
        for key, record in by_key.items()
        if record["record_type"] == "a1_continuation"
    }
    unexpected_a1 = sorted(actual_a1_keys - allowed_a1_keys)
    if unexpected_a1:
        raise ValueError(
            f"A1 contains optional or unselected work: {unexpected_a1[0]}"
        )

    a0_decisions = [result["a0"]["decision"] for result in checkpoint_results.values()]
    if any(value.startswith("BLOCKED") for value in a0_decisions):
        overall = "BLOCKED_OPERATIONAL"
    elif any(value.startswith("KILL") for value in a0_decisions):
        overall = "KILL_PCMC_GATE_A0"
    elif completed_stage == "A0":
        overall = "ADVANCE_TO_A1_COLLECTION"
    else:
        a1_decisions = [
            result.get("a1", {}).get("decision", "BLOCKED_INCOMPLETE_A1")
            for result in checkpoint_results.values()
        ]
        if any(value.startswith("BLOCKED") for value in a1_decisions):
            overall = "BLOCKED_OPERATIONAL"
        elif all(value == "PASS_A1" for value in a1_decisions):
            overall = "ADVANCE_TO_B0_ORACLE"
        else:
            overall = "KILL_PCMC_GATE_A1"
    return {
        "protocol_id": audit.protocol_id,
        "protocol_sha256": audit.protocol_sha256,
        "completed_stage": completed_stage,
        "record_count": len(records),
        "checkpoint_results": checkpoint_results,
        "overall_decision": overall,
        "authorization": (
            "B0_CONSTRAINED_ORACLE"
            if overall == "ADVANCE_TO_B0_ORACLE"
            else (
                "A1_CAUSAL_CONTINUATION"
                if overall == "ADVANCE_TO_A1_COLLECTION"
                else "NONE"
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--completed-stage", choices=("A0", "A1"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    summary = summarize_gate_a(
        protocol,
        load_jsonl(args.records),
        completed_stage=args.completed_stage,
    )
    write_json_atomic(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
