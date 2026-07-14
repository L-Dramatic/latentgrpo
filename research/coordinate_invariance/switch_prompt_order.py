from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


REQUIRED_FIELDS = {"problem", "solution", "answer", "subject", "level", "unique_id"}


def canonical_config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _ordering_key(row: dict[str, Any], seed: int) -> str:
    payload = (
        f"{seed}\0{row['unique_id']}\0{row['problem']}".encode("utf-8")
    )
    return hashlib.sha256(payload).hexdigest()


def build_prompt_order(
    config: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    *,
    dataset_sha256: str,
) -> dict[str, Any]:
    dataset_config = config["dataset"]
    if dataset_sha256 != str(dataset_config["file_sha256"]):
        raise ValueError(
            f"dataset SHA-256 is {dataset_sha256}, "
            f"expected {dataset_config['file_sha256']}"
        )
    expected_rows = int(dataset_config["expected_rows"])
    if len(rows) != expected_rows:
        raise ValueError(f"dataset has {len(rows)} rows, expected {expected_rows}")

    unique_ids: set[str] = set()
    candidates: list[dict[str, Any]] = []
    seed = int(config["ordering"]["seed"])
    for dataset_index, row in enumerate(rows):
        missing = REQUIRED_FIELDS - set(row)
        if missing:
            raise ValueError(f"dataset row {dataset_index} lacks {sorted(missing)}")
        unique_id = str(row["unique_id"])
        if unique_id in unique_ids:
            raise ValueError(f"duplicate unique_id: {unique_id}")
        unique_ids.add(unique_id)
        problem = str(row["problem"])
        candidates.append(
            {
                "dataset_index": dataset_index,
                "ordering_key": _ordering_key(row, seed),
                "unique_id": unique_id,
                "problem_sha256": hashlib.sha256(
                    problem.encode("utf-8")
                ).hexdigest(),
                "subject": str(row["subject"]),
                "level": int(row["level"]),
            }
        )
    candidates.sort(key=lambda row: (row["ordering_key"], row["dataset_index"]))
    for rank, row in enumerate(candidates):
        row["scan_rank"] = rank

    return {
        "experiment_name": config["experiment_name"],
        "status": "pass",
        "evidence_level": "model-independent selection contract only",
        "config_sha256": canonical_config_hash(config),
        "dataset": {
            "id": dataset_config["id"],
            "revision": dataset_config["revision"],
            "file_sha256": dataset_sha256,
            "rows": len(rows),
        },
        "ordering": config["ordering"],
        "eligibility": config["eligibility"],
        "splits": config["splits"],
        "ordered_candidates": candidates,
        "interpretation": (
            "The model must scan this order and assign the first eligible prompts "
            "to calibration, then the next eligible prompts to test. Accuracy and "
            "chart effects cannot influence eligibility."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    dataset_bytes = args.dataset_file.read_bytes()
    rows = [
        json.loads(line)
        for line in dataset_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    report = build_prompt_order(
        config,
        rows,
        dataset_sha256=hashlib.sha256(dataset_bytes).hexdigest(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "ordered_candidates"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
