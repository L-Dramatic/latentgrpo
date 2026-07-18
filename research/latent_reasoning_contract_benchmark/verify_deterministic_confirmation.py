"""Integrity verification for the untouched deterministic confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .deterministic_confirmation import validate_confirmation_config
from .deterministic_intervention_a0 import summarize_records


ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_confirmation(
    root: Path, config_path: Path, records_path: Path, summary_path: Path
) -> dict[str, Any]:
    root = root.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_confirmation_config(root, config)
    prompt_manifest = json.loads(
        (root / config["prompt_manifest"]["path"]).read_text(encoding="utf-8")
    )
    expected_ranks = {
        row["selection_rank"]
        for row in prompt_manifest["records"]
        if row["split"] == config["prompt_manifest"]["split"]
    }
    expected_keys = {
        (method_id, rank) for method_id in config["methods"] for rank in expected_ranks
    }
    config_sha = _sha256(config_path)
    runner_path = Path(__file__).with_name("deterministic_intervention_a0.py")
    wrapper_path = Path(__file__).with_name("deterministic_confirmation.py")
    runner_sha = _sha256(runner_path)
    wrapper_sha = _sha256(wrapper_path)
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    observed_keys = {(record["method_id"], record["selection_rank"]) for record in records}
    recomputed = summarize_records(records, config)
    memory_values = {
        method: float(summary["runtime"]["methods"][method]["cuda_peak_allocated_mib"])
        for method in config["methods"]
    }
    controls = {
        "exact_untouched_record_coverage": observed_keys == expected_keys and len(records) == len(expected_keys),
        "record_provenance": all(
            record.get("protocol_id") == config["protocol_id"]
            and record.get("config_file_sha256") == config_sha
            and record.get("implementation_sha256") == runner_sha
            for record in records
        ),
        "condition_coverage_and_finiteness": all(
            set(record["conditions"]) == set(config["conditions"])
            and all(condition.get("finite") is True for condition in record["conditions"].values())
            for record in records
        ),
        "frozen_rule_recomputation": (
            summary.get("base_a0_rule_decision") == recomputed["decision"]
            and summary.get("controls_pass") == recomputed["controls_pass"]
            and summary.get("signal_pass") == recomputed["signal_pass"]
            and summary.get("methods") == recomputed["methods"]
        ),
        "confirmation_mapping": (
            summary.get("decision") == "PASS_CONFIRMATION"
            and summary.get("stage") == "deterministic_intervention_untouched_confirmation"
            and summary.get("scientific_evidence") == "untouched_confirmation"
        ),
        "wrapper_provenance": summary.get("confirmation_wrapper_sha256") == wrapper_sha,
        "cuda_memory_bounded": all(
            value <= float(config["controls"]["maximum_cuda_peak_allocated_mib"])
            for value in memory_values.values()
        ),
    }
    passed = all(controls.values())
    return {
        "protocol_id": config["protocol_id"],
        "decision": "PASS_CONFIRMATION_AUDIT" if passed else "HOLD_CONFIRMATION_AUDIT",
        "controls": controls,
        "record_count": len(records),
        "config_sha256": config_sha,
        "runner_sha256": runner_sha,
        "wrapper_sha256": wrapper_sha,
        "records_sha256": _sha256(records_path),
        "summary_sha256": _sha256(summary_path),
        "cuda_peak_allocated_mib": memory_values,
        "training_used": False,
        "scientific_evidence": "integrity_audit_only",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify_confirmation(args.workspace, args.config, args.records, args.summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "controls": report["controls"]}))
    return 0 if report["decision"] == "PASS_CONFIRMATION_AUDIT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
