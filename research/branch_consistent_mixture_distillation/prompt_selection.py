"""Create the frozen label-blind BCMD prompt manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data = config["data"]
    count = int(data["example_count"])
    prompt_count = int(data["prompt_count"])
    calibration_count = int(config["oracle_gate"]["calibration_prompt_count"])
    confirmation_count = int(config["oracle_gate"]["confirmation_prompt_count"])
    if prompt_count != calibration_count + confirmation_count:
        raise ValueError("calibration and confirmation counts must cover the manifest")
    if not 0 < prompt_count <= count:
        raise ValueError("prompt count must be within the dataset")

    salt = str(data["selection_salt"])
    ranked = sorted(
        (
            hashlib.sha256(f"math500:{index:03d}|{salt}".encode("utf-8")).hexdigest(),
            f"math500:{index:03d}",
        )
        for index in range(count)
    )[:prompt_count]
    records = [
        {
            "selection_rank": rank,
            "example_id": example_id,
            "selection_sha256": digest,
            "split": "calibration" if rank < calibration_count else "confirmation",
        }
        for rank, (digest, example_id) in enumerate(ranked)
    ]
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return {
        "protocol_id": config["protocol_id"],
        "selection": "label_blind_sha256_rank",
        "selection_salt": salt,
        "labels_read": False,
        "dataset_id": data["dataset_id"],
        "dataset_revision": data["revision"],
        "config_sha256": _sha256(config_path),
        "record_count": len(records),
        "calibration_count": calibration_count,
        "confirmation_count": confirmation_count,
        "records_canonical_sha256": hashlib.sha256(canonical).hexdigest(),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "research"
        / "branch_consistent_mixture_distillation"
        / "configs"
        / "gate_minus_one_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "record_count": manifest["record_count"],
                "records_canonical_sha256": manifest["records_canonical_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
