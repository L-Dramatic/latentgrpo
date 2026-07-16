"""Pure validation and decision logic for P1 sacrificial discovery v1."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable


DIRECTIONS = ("forward", "reverse")


@dataclass(frozen=True)
class ProtocolAudit:
    protocol_id: str
    protocol_sha256: str
    prompt_count: int
    candidate_count: int
    histories_per_direction: int
    expected_action_records: int
    expected_path_records: int
    materialized_seed_count: int


def load_protocol(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    data = json.loads(payload.decode("utf-8"))
    if not isinstance(data, dict):
        raise TypeError("protocol root must be a JSON object")
    data["_protocol_sha256"] = hashlib.sha256(payload).hexdigest()
    return data


def _require_unique_ints(values: Any, name: str) -> list[int]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty list")
    normalized = [int(value) for value in values]
    if len(set(normalized)) != len(normalized) or any(value <= 0 for value in normalized):
        raise ValueError(f"{name} must contain unique positive integers")
    return normalized


def action_seed(protocol: dict[str, Any], prompt_index: int, action_index: int) -> int:
    return int(protocol["action_seeds"][action_index]) + prompt_index * int(
        protocol["prompt_seed_stride"]
    )


def history_seed(
    protocol: dict[str, Any],
    *,
    direction: str,
    prompt_index: int,
    history_index: int,
    action_index: int | None = None,
) -> int:
    if direction not in DIRECTIONS:
        raise ValueError("direction must be forward or reverse")
    base = int(protocol[f"{direction}_history_seeds"][history_index])
    result = base + prompt_index * int(protocol["prompt_seed_stride"])
    if direction == "reverse":
        if action_index is None:
            raise ValueError("reverse history seeds require an action index")
        result += action_index * int(protocol["reverse_action_seed_stride"])
    elif action_index is not None:
        raise ValueError("forward histories are shared and have no action index")
    return result


def validate_protocol(protocol: dict[str, Any]) -> ProtocolAudit:
    required = {
        "protocol_id",
        "checkpoint",
        "action_seeds",
        "forward_history_seeds",
        "reverse_history_seeds",
        "prompt_seed_stride",
        "reverse_action_seed_stride",
        "horizons",
        "max_visible_horizon",
        "natural_latent_cap",
        "visible_temperature",
        "wall_clock_cap_seconds",
        "prompts",
    }
    missing = sorted(required - protocol.keys())
    if missing:
        raise ValueError(f"protocol is missing fields: {missing}")
    action_bases = _require_unique_ints(protocol["action_seeds"], "action_seeds")
    forward_bases = _require_unique_ints(
        protocol["forward_history_seeds"], "forward_history_seeds"
    )
    reverse_bases = _require_unique_ints(
        protocol["reverse_history_seeds"], "reverse_history_seeds"
    )
    if len(forward_bases) != len(reverse_bases):
        raise ValueError("forward and reverse must use the same history count")
    horizons = _require_unique_ints(protocol["horizons"], "horizons")
    if horizons != sorted(horizons) or horizons[-1] != int(protocol["max_visible_horizon"]):
        raise ValueError("horizons must be sorted and end at max_visible_horizon")
    if float(protocol["visible_temperature"]) <= 0:
        raise ValueError("visible_temperature must be positive")
    if int(protocol["natural_latent_cap"]) < 1:
        raise ValueError("natural_latent_cap must be positive")
    if int(protocol["wall_clock_cap_seconds"]) != 3600:
        raise ValueError("v1 wall-clock cap must remain exactly 3600 seconds")

    prompts = protocol["prompts"]
    if not isinstance(prompts, list) or len(prompts) != 8:
        raise ValueError("v1 requires exactly eight prompts")
    prompt_ids = [item.get("prompt_id") for item in prompts]
    if any(not isinstance(value, str) or not value for value in prompt_ids):
        raise ValueError("every prompt requires a non-empty prompt_id")
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("prompt ids must be unique")
    if any(not isinstance(item.get("problem"), str) or not item["problem"].strip() for item in prompts):
        raise ValueError("every prompt requires non-empty problem text")

    seeds: list[int] = []
    for prompt_index in range(len(prompts)):
        for action_index in range(len(action_bases)):
            seeds.append(action_seed(protocol, prompt_index, action_index))
        for history_index in range(len(forward_bases)):
            seeds.append(
                history_seed(
                    protocol,
                    direction="forward",
                    prompt_index=prompt_index,
                    history_index=history_index,
                )
            )
            for action_index in range(len(action_bases)):
                seeds.append(
                    history_seed(
                        protocol,
                        direction="reverse",
                        prompt_index=prompt_index,
                        action_index=action_index,
                        history_index=history_index,
                    )
                )
    if len(seeds) != len(set(seeds)):
        raise ValueError("materialized action/history seeds are not globally unique")
    expected_actions = len(prompts) * (1 + len(action_bases))
    expected_paths = (
        len(prompts) * len(action_bases) * len(forward_bases) * len(DIRECTIONS)
    )
    return ProtocolAudit(
        protocol_id=str(protocol["protocol_id"]),
        protocol_sha256=str(protocol.get("_protocol_sha256", "UNHASHED")),
        prompt_count=len(prompts),
        candidate_count=len(action_bases),
        histories_per_direction=len(forward_bases),
        expected_action_records=expected_actions,
        expected_path_records=expected_paths,
        materialized_seed_count=len(seeds),
    )


def path_record_key(
    prompt_id: str, action_index: int, direction: str, history_index: int
) -> str:
    if direction not in DIRECTIONS:
        raise ValueError("direction must be forward or reverse")
    return f"path|{prompt_id}|a{action_index}|{direction}|h{history_index}"


def action_record_key(prompt_id: str, action_index: int | None) -> str:
    role = "reference" if action_index is None else f"a{action_index}"
    return f"action|{prompt_id}|{role}"


def cumulative_at_horizons(per_step_kl: Iterable[float], horizons: Iterable[int]) -> dict[str, float]:
    values = [float(value) for value in per_step_kl]
    if any(not math.isfinite(value) or value < -1e-12 for value in values):
        raise ValueError("per-step KL must be finite and nonnegative")
    prefix: list[float] = []
    running = 0.0
    for value in values:
        running += max(value, 0.0)
        prefix.append(running)
    result: dict[str, float] = {}
    for horizon in horizons:
        normalized = int(horizon)
        if normalized < 1:
            raise ValueError("horizons must be positive")
        result[str(normalized)] = prefix[min(normalized, len(prefix)) - 1] if prefix else 0.0
    return result


def _sign(value: float) -> int:
    return 1 if value > 0 else (-1 if value < 0 else 0)


def _mean(values: Iterable[float]) -> float:
    normalized = list(values)
    if not normalized:
        raise ValueError("mean requires at least one value")
    return sum(normalized) / len(normalized)


def _directional_groups(
    path_records: list[dict[str, Any]], direction: str, horizons: list[int]
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in path_records:
        if record.get("record_type") != "path" or record.get("direction") != direction:
            continue
        if record.get("status") != "COMPLETE":
            continue
        enriched = dict(record)
        enriched["cumulative"] = cumulative_at_horizons(record["per_step_kl"], horizons)
        groups[(str(record["prompt_id"]), int(record["action_index"]))].append(enriched)
    for values in groups.values():
        values.sort(key=lambda item: int(item["history_index"]))
    return groups


def _action_means(
    groups: dict[tuple[str, int], list[dict[str, Any]]], horizons: list[int]
) -> dict[tuple[str, int], dict[str, float]]:
    return {
        key: {
            str(horizon): _mean(record["cumulative"][str(horizon)] for record in records)
            for horizon in horizons
        }
        for key, records in groups.items()
    }


def _robust_flip_prompts(
    groups: dict[tuple[str, int], list[dict[str, Any]]],
    means: dict[tuple[str, int], dict[str, float]],
    history_count: int,
) -> dict[str, list[list[int]]]:
    by_prompt: dict[str, list[int]] = defaultdict(list)
    for prompt_id, action_index in groups:
        by_prompt[prompt_id].append(action_index)
    flips: dict[str, list[list[int]]] = {}
    for prompt_id, action_indices in by_prompt.items():
        prompt_flips: list[list[int]] = []
        for left_pos, left in enumerate(sorted(action_indices)):
            for right in sorted(action_indices)[left_pos + 1 :]:
                left_records = groups[(prompt_id, left)]
                right_records = groups[(prompt_id, right)]
                if len(left_records) != history_count or len(right_records) != history_count:
                    continue
                short_sign = _sign(means[(prompt_id, left)]["8"] - means[(prompt_id, right)]["8"])
                long_sign = _sign(means[(prompt_id, left)]["64"] - means[(prompt_id, right)]["64"])
                if short_sign == 0 or long_sign == 0 or short_sign == long_sign:
                    continue
                stable = True
                for omitted in range(history_count):
                    keep = [index for index in range(history_count) if index != omitted]
                    short_diff = _mean(
                        left_records[index]["cumulative"]["8"]
                        - right_records[index]["cumulative"]["8"]
                        for index in keep
                    )
                    long_diff = _mean(
                        left_records[index]["cumulative"]["64"]
                        - right_records[index]["cumulative"]["64"]
                        for index in keep
                    )
                    if _sign(short_diff) != short_sign or _sign(long_diff) != long_sign:
                        stable = False
                        break
                if stable:
                    prompt_flips.append([left, right])
        if prompt_flips:
            flips[prompt_id] = prompt_flips
    return flips


def _rank_stability(
    groups: dict[tuple[str, int], list[dict[str, Any]]],
    means: dict[tuple[str, int], dict[str, float]],
    history_count: int,
) -> float | None:
    by_prompt: dict[str, list[int]] = defaultdict(list)
    for prompt_id, action_index in groups:
        by_prompt[prompt_id].append(action_index)
    matches = 0
    total = 0
    for prompt_id, action_indices in by_prompt.items():
        ordered = sorted(action_indices)
        for left_pos, left in enumerate(ordered):
            for right in ordered[left_pos + 1 :]:
                left_records = groups[(prompt_id, left)]
                right_records = groups[(prompt_id, right)]
                if len(left_records) != history_count or len(right_records) != history_count:
                    continue
                target = _sign(means[(prompt_id, left)]["64"] - means[(prompt_id, right)]["64"])
                if target == 0:
                    continue
                for index in range(history_count):
                    observed = _sign(
                        left_records[index]["cumulative"]["64"]
                        - right_records[index]["cumulative"]["64"]
                    )
                    if observed == 0:
                        continue
                    matches += int(observed == target)
                    total += 1
    return matches / total if total else None


def summarize_discovery(
    protocol: dict[str, Any], records: list[dict[str, Any]], *, run_complete: bool
) -> dict[str, Any]:
    audit = validate_protocol(protocol)
    action_records = [item for item in records if item.get("record_type") == "action"]
    path_records = [item for item in records if item.get("record_type") == "path"]
    action_keys = {str(item.get("key")) for item in action_records}
    path_keys = {str(item.get("key")) for item in path_records}
    duplicate_records = (
        len(action_keys) != len(action_records) or len(path_keys) != len(path_records)
    )

    references = {
        str(item["prompt_id"]): item
        for item in action_records
        if item.get("role") == "reference"
    }
    candidates = [item for item in action_records if item.get("role") == "candidate"]
    natural_pairs = 0
    for item in candidates:
        reference = references.get(str(item["prompt_id"]))
        if reference and reference.get("endpoint") == "NATURAL_VISIBLE" and item.get(
            "endpoint"
        ) == "NATURAL_VISIBLE":
            natural_pairs += 1
    pair_count = audit.prompt_count * audit.candidate_count
    natural_visible_rate = natural_pairs / pair_count

    horizons = [int(value) for value in protocol["horizons"]]
    history_count = audit.histories_per_direction
    forward_groups = _directional_groups(path_records, "forward", horizons)
    reverse_groups = _directional_groups(path_records, "reverse", horizons)
    forward_means = _action_means(forward_groups, horizons)
    reverse_means = _action_means(reverse_groups, horizons)
    forward_flips = _robust_flip_prompts(
        forward_groups, forward_means, history_count
    )
    reverse_flips = _robust_flip_prompts(
        reverse_groups, reverse_means, history_count
    )
    rank_stability = _rank_stability(forward_groups, forward_means, history_count)

    floor = float(protocol["eligible_d64_floor"])
    late_fraction_threshold = float(protocol["late_tail_fraction"])
    action_summaries: list[dict[str, Any]] = []
    ratios: list[float] = []
    late_count = 0
    for (prompt_id, action_index), means in sorted(forward_means.items()):
        d8 = means["8"]
        d64 = means["64"]
        eligible = d64 >= floor
        ratio = d8 / d64 if d64 > 0 else None
        tail_fraction = (d64 - d8) / d64 if d64 > 0 else None
        late = bool(eligible and tail_fraction is not None and tail_fraction >= late_fraction_threshold)
        if eligible and ratio is not None:
            ratios.append(ratio)
            late_count += int(late)
        action_summaries.append(
            {
                "prompt_id": prompt_id,
                "action_index": action_index,
                "forward_mean": means,
                "reverse_mean": reverse_means.get((prompt_id, action_index)),
                "risk_eligible": eligible,
                "tail_8_64": d64 - d8,
                "tail_fraction_8_64": tail_fraction,
                "d8_d64_ratio": ratio,
                "late_mass": late,
            }
        )
    eligible_count = len(ratios)
    late_mass_rate = late_count / eligible_count if eligible_count else None
    median_ratio = median(ratios) if ratios else None

    error_records = [
        item for item in records if item.get("status") in {"ERROR", "OOM", "INVALID"}
    ]
    expected_counts_met = (
        len(action_records) == audit.expected_action_records
        and len(path_records) == audit.expected_path_records
        and not duplicate_records
    )
    decision = "HOLD-INSUFFICIENT"
    reasons: list[str] = []
    if error_records:
        decision = "KILL"
        reasons.append("runtime/source/cache/density error record exists")
    elif run_complete and natural_visible_rate < float(protocol["minimum_natural_visible_rate"]):
        decision = "KILL"
        reasons.append("natural-visible pair rate is below the frozen minimum")
    elif (
        run_complete
        and not forward_flips
        and late_mass_rate is not None
        and late_mass_rate < 0.10
        and median_ratio is not None
        and median_ratio >= 0.90
    ):
        decision = "KILL"
        reasons.append("no robust flips and the horizon tail is negligible")
    elif (
        run_complete
        and expected_counts_met
        and natural_visible_rate >= float(protocol["minimum_natural_visible_rate"])
        and len(forward_flips) >= int(protocol["minimum_go_flip_prompts"])
        and late_mass_rate is not None
        and late_mass_rate >= float(protocol["minimum_go_late_mass_rate"])
        and median_ratio is not None
        and median_ratio <= float(protocol["maximum_go_median_d8_d64_ratio"])
        and rank_stability is not None
        and rank_stability >= float(protocol["minimum_rank_stability"])
    ):
        decision = "GO-REWRITE-METHOD-CONTRACT"
        reasons.append("all frozen sacrificial GO gates passed")
    else:
        if not run_complete or not expected_counts_met:
            reasons.append("run is incomplete or record counts do not match the protocol")
        if rank_stability is None or rank_stability < float(protocol["minimum_rank_stability"]):
            reasons.append("forward H64 ranking stability is insufficient")
        if eligible_count == 0:
            reasons.append("no action crossed the frozen D64 numerical eligibility floor")

    return {
        "protocol": asdict(audit),
        "run_complete": run_complete,
        "record_counts": {
            "action": len(action_records),
            "path": len(path_records),
            "expected_counts_met": expected_counts_met,
            "duplicate_records": duplicate_records,
            "error_records": len(error_records),
        },
        "natural_visible_pair_rate": natural_visible_rate,
        "eligible_action_count": eligible_count,
        "late_mass_rate": late_mass_rate,
        "median_d8_d64_ratio": median_ratio,
        "forward_rank_stability": rank_stability,
        "robust_forward_flip_prompts": forward_flips,
        "robust_reverse_flip_prompts": reverse_flips,
        "action_summaries": action_summaries,
        "decision": decision,
        "decision_reasons": reasons,
    }
