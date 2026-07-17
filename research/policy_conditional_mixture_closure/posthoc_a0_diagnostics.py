"""Descriptive-only diagnostics for a completed, already-decided PCMC A0."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from .checkpoint_protocol import checkpoint_profiles, load_protocol, quantile
from .gate_a_analysis import (
    load_jsonl,
    spearman,
    summarize_gate_a,
    write_json_atomic,
)


QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": mean(values),
        "quantiles": {
            f"q{int(probability * 100):02d}": quantile(values, probability)
            for probability in QUANTILES
        },
    }


def describe_a0(
    protocol: dict[str, Any], records: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    decision = summarize_gate_a(protocol, records, completed_stage="A0")
    checkpoint_results: dict[str, Any] = {}
    for profile in checkpoint_profiles(protocol):
        checkpoint_records = [
            record
            for record in records
            if record["record_type"] == "a0_event"
            and record["checkpoint_id"] == profile.checkpoint_id
        ]
        complete = [
            record for record in checkpoint_records if record["status"] == "COMPLETE"
        ]
        if not complete:
            raise ValueError(f"no complete A0 records for {profile.checkpoint_id}")
        js_values = [float(record["js_branch_arithmetic_nats"]) for record in complete]
        maximum_weights = [float(record["maximum_weight"]) for record in complete]
        effective_support = [float(record["effective_support"]) for record in complete]
        entropy = [float(record["weight_entropy_normalized"]) for record in complete]
        prompt_lengths = [float(record["prompt_token_count"]) for record in complete]
        disagreement = [
            int(record["arithmetic_top_token_id"] != record["branch_teacher_top_token_id"])
            for record in complete
        ]
        top_examples = sorted(
            (
                {
                    "example_id": str(record["example_id"]),
                    "js_nats": float(record["js_branch_arithmetic_nats"]),
                    "maximum_weight": float(record["maximum_weight"]),
                    "effective_support": float(record["effective_support"]),
                }
                for record in complete
            ),
            key=lambda value: value["js_nats"],
            reverse=True,
        )[:10]
        checkpoint_results[profile.checkpoint_id] = {
            "status_counts": dict(Counter(record["status"] for record in checkpoint_records)),
            "content_support_counts": {
                str(key): value
                for key, value in sorted(
                    Counter(
                        int(record["content_support_size"]) for record in complete
                    ).items()
                )
            },
            "js_nats": _distribution(js_values),
            "maximum_weight": _distribution(maximum_weights),
            "effective_support": _distribution(effective_support),
            "normalized_weight_entropy": _distribution(entropy),
            "fraction_maximum_weight_at_least_0_99": mean(
                value >= 0.99 for value in maximum_weights
            ),
            "fraction_maximum_weight_at_least_0_999": mean(
                value >= 0.999 for value in maximum_weights
            ),
            "fraction_js_at_least_0_005": mean(value >= 0.005 for value in js_values),
            "top_token_disagreement_rate": mean(disagreement),
            "spearman_js_vs_one_minus_maximum_weight": spearman(
                js_values, [1.0 - value for value in maximum_weights]
            ),
            "spearman_js_vs_effective_support": spearman(
                js_values, effective_support
            ),
            "spearman_js_vs_prompt_length": spearman(js_values, prompt_lengths),
            "largest_js_examples": top_examples,
        }
    return {
        "analysis_type": "POSTHOC_DESCRIPTION_NOT_GATE",
        "may_change_frozen_decision": False,
        "protocol_id": decision["protocol_id"],
        "protocol_sha256": decision["protocol_sha256"],
        "frozen_a0_decision": decision["overall_decision"],
        "record_count": len(records),
        "checkpoint_results": checkpoint_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = describe_a0(load_protocol(args.protocol), load_jsonl(args.records))
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
