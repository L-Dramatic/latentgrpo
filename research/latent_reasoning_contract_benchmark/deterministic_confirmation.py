"""Untouched deterministic confirmation wrapper with locked A0 reuse."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .deterministic_intervention_a0 import run_a0, validate_config


ROOT = Path(__file__).resolve().parents[2]
REUSED_KEYS = (
    "methods",
    "conditions",
    "equal_depth_effect_conditions",
    "random_seed",
    "bootstrap_seed",
    "bootstrap_replicates",
    "controls",
    "calibration_signal_gates",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"path escaped workspace: {relative}")
    return path


def validate_confirmation_config(root: Path, config: dict[str, Any]) -> None:
    validate_config(root, config)
    if config["prompt_manifest"].get("split") != "confirmation":
        raise ValueError("confirmation protocol must use the untouched split")
    evidence = config.get("calibration_evidence", {})
    calibration_config_path = _inside(root, evidence["config_path"])
    if _sha256(calibration_config_path) != evidence["config_sha256"]:
        raise ValueError("calibration config hash drift")
    calibration = json.loads(calibration_config_path.read_text(encoding="utf-8"))
    for key in REUSED_KEYS:
        if config.get(key) != calibration.get(key):
            raise ValueError(f"confirmation changed frozen A0 field: {key}")
    summary_path = _inside(root, evidence["summary_path"])
    audit_path = _inside(root, evidence["audit_path"])
    if _sha256(summary_path) != evidence["summary_sha256"]:
        raise ValueError("calibration summary hash drift")
    if _sha256(audit_path) != evidence["audit_sha256"]:
        raise ValueError("calibration audit hash drift")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("decision") != evidence["required_decision"]:
        raise ValueError("calibration audit decision drift")


def map_confirmation_report(report: dict[str, Any]) -> dict[str, Any]:
    mapped = dict(report)
    base_decision = report["decision"]
    mapped["base_a0_rule_decision"] = base_decision
    mapped["decision"] = {
        "PASS_A0_SIGNAL": "PASS_CONFIRMATION",
        "HOLD_A0_CONTROLS": "HOLD_CONFIRMATION",
        "KILL_DETERMINISTIC_EFFECT_BRANCH": "KILL_DETERMINISTIC_EFFECT_BRANCH",
    }[base_decision]
    mapped["stage"] = "deterministic_intervention_untouched_confirmation"
    mapped["scientific_evidence"] = "untouched_confirmation"
    mapped["next_required_action"] = (
        "freeze stochastic trained-checkpoint contract-consequence preflight"
        if mapped["decision"] == "PASS_CONFIRMATION"
        else "do not promote deterministic branch"
    )
    return mapped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        validate_confirmation_config(args.workspace.resolve(), config)
        report = map_confirmation_report(
            run_a0(args.workspace, args.config, args.records)
        )
        report["confirmation_wrapper_sha256"] = _sha256(Path(__file__))
    except Exception as exc:
        report = {
            "stage": "deterministic_intervention_untouched_confirmation",
            "decision": "HOLD_CONFIRMATION",
            "failures": [f"{type(exc).__name__}: {exc}"],
            "training_used": False,
            "scientific_evidence": False,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "failures": report.get("failures", [])}))
    return 0 if report["decision"] == "PASS_CONFIRMATION" else 2


if __name__ == "__main__":
    raise SystemExit(main())
