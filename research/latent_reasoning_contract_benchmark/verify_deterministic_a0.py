"""Independent post-run integrity verification for deterministic A0 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .deterministic_intervention_a0 import summarize_records, validate_config


ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(
    root: Path, config_path: Path, records_path: Path, summary_path: Path
) -> dict[str, Any]:
    root = root.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(root, config)
    config_sha = _sha256(config_path)
    runner_path = Path(__file__).with_name("deterministic_intervention_a0.py")
    runner_sha = _sha256(runner_path)
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_keys = {
        (method, rank)
        for method in config["methods"]
        for rank in range(int(config["prompt_manifest"]["count"]))
    }
    observed_keys = {(record["method_id"], record["selection_rank"]) for record in records}
    provenance_pass = all(
        record.get("protocol_id") == config["protocol_id"]
        and record.get("config_file_sha256") == config_sha
        and record.get("implementation_sha256") == runner_sha
        for record in records
    )
    condition_set = set(config["conditions"])
    condition_pass = all(set(record["conditions"]) == condition_set for record in records)
    finite_pass = all(
        condition.get("finite") is True
        for record in records
        for condition in record["conditions"].values()
    )
    recomputed = summarize_records(records, config)
    recomputation_pass = all(
        summary.get(key) == recomputed.get(key)
        for key in ("decision", "controls_pass", "signal_pass", "methods")
    )
    memory_values = {
        method: float(summary["runtime"]["methods"][method]["cuda_peak_allocated_mib"])
        for method in config["methods"]
    }
    memory_pass = all(
        value <= float(config["controls"]["maximum_cuda_peak_allocated_mib"])
        for value in memory_values.values()
    )
    source_summary_pass = (
        summary.get("config_file_sha256") == config_sha
        and summary.get("implementation_sha256") == runner_sha
        and summary.get("prompt_manifest_sha256") == config["prompt_manifest"]["sha256"]
        and summary.get("split") == config["prompt_manifest"]["split"]
        and summary.get("training_used") is False
        and summary.get("scientific_evidence") == "calibration_only"
    )
    controls = {
        "exact_record_key_coverage": observed_keys == expected_keys and len(records) == len(expected_keys),
        "record_provenance": provenance_pass,
        "condition_coverage": condition_pass,
        "all_conditions_finite": finite_pass,
        "summary_recomputation": recomputation_pass,
        "cuda_memory_bounded": memory_pass,
        "summary_scope_and_provenance": source_summary_pass,
    }
    passed = all(controls.values()) and summary.get("decision") == "PASS_A0_SIGNAL"
    return {
        "protocol_id": config["protocol_id"],
        "decision": "PASS_A0_AUDIT" if passed else "HOLD_A0_AUDIT",
        "controls": controls,
        "record_count": len(records),
        "config_sha256": config_sha,
        "runner_sha256": runner_sha,
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
    report = verify(args.workspace, args.config, args.records, args.summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "controls": report["controls"]}))
    return 0 if report["decision"] == "PASS_A0_AUDIT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
