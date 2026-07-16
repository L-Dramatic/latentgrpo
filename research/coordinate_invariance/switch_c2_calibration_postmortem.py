from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch

from .switch_c2_scientific_gate import _select_calibration, canonical_config_hash


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_probe(record: dict[str, Any], scale: float) -> dict[str, Any]:
    return next(
        item
        for item in record["probes"]
        if math.isclose(
            float(item["relative_hidden_l2"]), scale, abs_tol=1e-15
        )
    )


def _find_update(record: dict[str, Any], gain: float) -> dict[str, Any]:
    return next(
        item
        for item in record["updates"]
        if math.isclose(
            float(item["predicted_objective_gain"]), gain, abs_tol=1e-15
        )
    )


def _probe_diagnostics(
    config: dict[str, Any], records: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    simple_order = list(config["updates"]["simple_baseline_selection_order"])
    tolerance = float(config["updates"]["selection_tie_tolerance"])
    candidates = []
    for raw_scale in config["probe_bank"]["relative_hidden_l2_grid"]:
        scale = float(raw_scale)
        means = {
            name: statistics.fmean(
                _find_probe(record, scale)["metrics"][name]["strict_spearman"]
                for record in records
            )
            for name in simple_order + ["visible_prefix_32", "visible_prefix_64"]
        }
        best_simple = simple_order[0]
        for name in simple_order[1:]:
            if means[name] > means[best_simple] + tolerance:
                best_simple = name
        candidates.append(
            {
                "relative_hidden_l2": scale,
                "best_simple": best_simple,
                "best_simple_mean_spearman": means[best_simple],
                "v32_mean_spearman": means["visible_prefix_32"],
                "v64_mean_spearman": means["visible_prefix_64"],
                "v32_margin": means["visible_prefix_32"] - means[best_simple],
            }
        )
    selected = candidates[0]
    for candidate in candidates[1:]:
        if candidate["v32_margin"] > selected["v32_margin"] + tolerance:
            selected = candidate
    return candidates, selected


def _gain_diagnostics(
    config: dict[str, Any],
    records: list[dict[str, Any]],
    best_simple: str,
) -> list[dict[str, Any]]:
    candidate_name = str(config["metrics"]["candidate_method"])
    output = []
    for raw_gain in config["updates"]["predicted_objective_gain_grid"]:
        gain = float(raw_gain)
        method_rows = {}
        raw_rows: dict[str, list[dict[str, Any]]] = {}
        all_valid = True
        for method in (candidate_name, best_simple):
            invalid = []
            rows = []
            for record in records:
                row = _find_update(record, gain)["methods"][method]
                rows.append(row)
                if not (row.get("valid_radius") and "strict_kl" in row):
                    invalid.append(
                        {
                            "selected_index": int(record["selected_index"]),
                            "unique_id": record["unique_id"],
                            "relative_hidden_l2": float(row["relative_hidden_l2"]),
                        }
                    )
            all_valid = all_valid and not invalid
            raw_rows[method] = rows
            method_rows[method] = {
                "valid_prompt_count": len(records) - len(invalid),
                "invalid_prompt_count": len(invalid),
                "invalid_prompts": invalid,
                "maximum_relative_hidden_l2": max(
                    float(row["relative_hidden_l2"]) for row in rows
                ),
            }
        candidate_objective = None
        baseline_objective = None
        candidate_kl = None
        baseline_kl = None
        objective_ratio = None
        kl_ratio = None
        formally_valid = False
        if all_valid:
            candidate_objective = statistics.fmean(
                float(row["objective_utility_gain"])
                for row in raw_rows[candidate_name]
            )
            baseline_objective = statistics.fmean(
                float(row["objective_utility_gain"])
                for row in raw_rows[best_simple]
            )
            candidate_kl = statistics.fmean(
                float(row["strict_kl"]) for row in raw_rows[candidate_name]
            )
            baseline_kl = statistics.fmean(
                float(row["strict_kl"]) for row in raw_rows[best_simple]
            )
            if baseline_objective > 0.0:
                objective_ratio = candidate_objective / baseline_objective
                formally_valid = candidate_objective >= float(
                    config["updates"]["minimum_calibration_objective_gain_ratio"]
                ) * baseline_objective
            kl_ratio = candidate_kl / max(baseline_kl, 1e-30)
        output.append(
            {
                "predicted_objective_gain": gain,
                "all_prompts_within_radius": all_valid,
                "methods": method_rows,
                "candidate_objective_gain_mean": candidate_objective,
                "baseline_objective_gain_mean": baseline_objective,
                "v32_to_best_simple_objective_gain_ratio": objective_ratio,
                "candidate_strict_kl_mean": candidate_kl,
                "baseline_strict_kl_mean": baseline_kl,
                "v32_to_best_simple_strict_kl_ratio": kl_ratio,
                "formally_valid": formally_valid,
            }
        )
    return output


def _finite_difference_diagnostics(
    config: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    fd_records = [record for record in records if record["finite_difference"]]
    candidates = []
    for raw_step in config["derivatives"]["finite_difference_relative_step_grid"]:
        step = float(raw_step)
        errors = [
            next(
                float(item["relative_error"])
                for item in record["finite_difference"]
                if math.isclose(
                    float(item["relative_hidden_l2"]), step, abs_tol=1e-15
                )
            )
            for record in fd_records
        ]
        candidates.append(
            {
                "relative_hidden_l2": step,
                "errors": errors,
                "median_relative_error": statistics.median(errors),
                "p90_relative_error": float(
                    torch.quantile(torch.tensor(errors, dtype=torch.float64), 0.9)
                ),
            }
        )
    selected = min(
        candidates,
        key=lambda item: (
            item["median_relative_error"],
            item["relative_hidden_l2"],
        ),
    )
    median_limit = float(config["derivatives"]["maximum_median_relative_error"])
    p90_limit = float(config["derivatives"]["maximum_p90_relative_error"])
    return {
        "prompt_count": len(fd_records),
        "candidates": candidates,
        "selected": selected,
        "thresholds": {
            "maximum_median_relative_error": median_limit,
            "maximum_p90_relative_error": p90_limit,
        },
        "pass": (
            selected["median_relative_error"] <= median_limit
            and selected["p90_relative_error"] <= p90_limit
        ),
    }


def build_diagnostic(
    config: dict[str, Any],
    header: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    config_path: Path,
    journal_path: Path,
) -> dict[str, Any]:
    if not records:
        raise ValueError("calibration journal contains no prompt records")
    actual_config_hash = canonical_config_hash(config)
    if header.get("config_sha256") != actual_config_hash:
        raise ValueError("journal and calibration config hashes do not match")

    probe_candidates, selected_probe = _probe_diagnostics(config, records)
    gains = _gain_diagnostics(
        config, records, str(selected_probe["best_simple"])
    )
    finite_difference = _finite_difference_diagnostics(config, records)

    selection_error = None
    try:
        _select_calibration(config, records)
    except ValueError as exc:
        selection_error = str(exc)
    if selection_error is None:
        raise ValueError("journal unexpectedly satisfies frozen calibration selection")

    decision = config["decision_rules"]
    projected = [float(record["projected_gradient_relative_error"]) for record in records]
    metric_transport = [
        float(
            record["transport_controls"][
                "maximum_metric_update_transport_relative_error"
            ]
        )
        for record in records
    ]
    orthogonal = [
        float(
            record["transport_controls"][
                "orthogonal_euclidean_direction_discrepancy"
            ]
        )
        for record in records
    ]
    condition12 = [
        float(
            record["transport_controls"][
                "condition12_euclidean_direction_discrepancy"
            ]
        )
        for record in records
    ]
    zero_point = max(float(record["zero_point_logit_max_abs_error"]) for record in records)
    basis_error = max(
        float(record["basis_orthogonality_max_abs_error"]) for record in records
    )
    runtime_keys = sorted(
        set().union(*(record.get("runtime", {}).keys() for record in records))
    )
    header_binding_keys = (
        "kind",
        "phase",
        "config_sha256",
        "eligibility_sha256",
        "calibration_sha256",
        "implementation_sha256",
        "state_collection_model_forward_calls",
        "whitening_variance_floor",
    )
    header_binding = {
        key: header.get(key) for key in header_binding_keys if key in header
    }
    header_binding["whitening_precision_dimension"] = len(
        header.get("whitening_precision", [])
    )

    def display_path(path: Path) -> str:
        try:
            return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            return path.as_posix()

    return {
        "schema_version": 1,
        "experiment_name": "switch-fctr-c2-calibration-postmortem-v1",
        "evidence_level": (
            "deterministic post-hoc diagnostic from the frozen append-only "
            "calibration journal; not a preregistered scientific artifact"
        ),
        "status": "fail",
        "decision": "no_go_for_frozen_switch_c2_v1",
        "failure_stage": "calibration_selection",
        "formal_failure_reason": selection_error,
        "held_out_test_authorized": False,
        "inputs": {
            "config": display_path(config_path),
            "config_file_sha256": _sha256(config_path),
            "config_canonical_sha256": actual_config_hash,
            "journal": display_path(journal_path),
            "journal_sha256": _sha256(journal_path),
            "journal_header_binding": header_binding,
            "record_count": len(records),
        },
        "probe_selection": {
            "candidates": probe_candidates,
            "selected": selected_probe,
            "all_v32_margins_nonpositive": all(
                item["v32_margin"] <= 0.0 for item in probe_candidates
            ),
        },
        "gain_selection": {
            "maximum_relative_hidden_l2": float(
                config["updates"]["maximum_relative_hidden_l2"]
            ),
            "candidates": gains,
            "valid_gain_count": sum(item["formally_valid"] for item in gains),
        },
        "finite_difference": finite_difference,
        "numerical_controls": {
            "projected_gradient": {
                "maximum_observed_relative_error": max(projected),
                "maximum_allowed_relative_error": float(
                    decision["maximum_projected_gradient_relative_error"]
                ),
                "pass": max(projected)
                <= float(decision["maximum_projected_gradient_relative_error"]),
            },
            "metric_update_transport": {
                "maximum_observed_relative_error": max(metric_transport),
                "maximum_allowed_relative_error": float(
                    decision["maximum_metric_update_transport_relative_error"]
                ),
                "pass": max(metric_transport)
                <= float(decision["maximum_metric_update_transport_relative_error"]),
            },
            "orthogonal_euclidean_transport": {
                "maximum_observed_discrepancy": max(orthogonal),
                "maximum_allowed_discrepancy": float(
                    decision["maximum_orthogonal_euclidean_direction_discrepancy"]
                ),
                "pass": max(orthogonal)
                <= float(
                    decision["maximum_orthogonal_euclidean_direction_discrepancy"]
                ),
            },
            "condition12_chart_sensitivity": {
                "median_observed_discrepancy": statistics.median(condition12),
                "minimum_required_median": float(
                    decision[
                        "minimum_median_condition12_euclidean_direction_discrepancy"
                    ]
                ),
                "pass": statistics.median(condition12)
                >= float(
                    decision[
                        "minimum_median_condition12_euclidean_direction_discrepancy"
                    ]
                ),
            },
            "zero_point": {
                "maximum_observed_logit_absolute_error": zero_point,
                "maximum_allowed_logit_absolute_error": float(
                    config["derivatives"][
                        "maximum_zero_point_logit_absolute_error"
                    ]
                ),
                "pass": zero_point
                <= float(
                    config["derivatives"][
                        "maximum_zero_point_logit_absolute_error"
                    ]
                ),
            },
            "basis_orthogonality": {
                "maximum_observed_absolute_error": basis_error,
                "maximum_allowed_absolute_error": float(
                    config["subspace"]["maximum_basis_orthogonality_error"]
                ),
                "pass": basis_error
                <= float(config["subspace"]["maximum_basis_orthogonality_error"]),
            },
        },
        "runtime": {
            "seconds": sum(
                float(record.get("runtime", {}).get("seconds", 0.0))
                for record in records
            ),
            "counters": {
                key: sum(
                    float(record.get("runtime", {}).get(key, 0.0))
                    for record in records
                )
                for key in runtime_keys
                if key != "seconds"
            },
        },
        "interpretation": (
            "The complete calibration journal cannot authorize the held-out C2 "
            "test. Every global gain violates the all-prompt radius control, the "
            "selected finite-difference check exceeds both tolerances, and V32 "
            "does not beat the calibration-selected simple baseline at any probe "
            "scale. Passing transport controls show that this is not explained by "
            "a basic coordinate-transport implementation defect."
        ),
    }


def load_journal(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError("calibration journal is incomplete")
    return json.loads(lines[0]), [json.loads(line) for line in lines[1:]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    journal_path = args.journal.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    header, records = load_journal(journal_path)
    diagnostic = build_diagnostic(
        config,
        header,
        records,
        config_path=config_path,
        journal_path=journal_path,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(diagnostic, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(diagnostic, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
