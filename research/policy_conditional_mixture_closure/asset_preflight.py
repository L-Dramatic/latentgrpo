"""Verify every pinned local asset before a PCMC checkpoint process starts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .checkpoint_protocol import checkpoint_profiles, load_protocol, validate_protocol
from .gate_a_analysis import write_json_atomic


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _git_output(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *arguments],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def verify_assets(protocol: dict[str, Any], workspace_root: Path) -> dict[str, Any]:
    audit = validate_protocol(protocol)
    root = workspace_root.resolve()
    failures: list[str] = []
    dataset = protocol["dataset"]
    dataset_path = (root / str(dataset["arrow_path"])).resolve()
    if root not in dataset_path.parents:
        raise ValueError("dataset path escaped the workspace")
    dataset_observed = file_sha256(dataset_path) if dataset_path.is_file() else None
    if dataset_observed != str(dataset["arrow_sha256"]):
        failures.append("dataset Arrow SHA-256 mismatch")

    checkpoint_results: list[dict[str, Any]] = []
    for profile in checkpoint_profiles(protocol):
        checkpoint_root = (root / profile.checkpoint_path).resolve()
        source_root = (root / profile.source_path).resolve()
        if root not in checkpoint_root.parents or root not in source_root.parents:
            raise ValueError("checkpoint or source path escaped the workspace")
        model_path = checkpoint_root / "model.safetensors"
        model_observed = file_sha256(model_path) if model_path.is_file() else None
        config_path = checkpoint_root / "config.json"
        model_type = None
        if config_path.is_file():
            model_type = json.loads(config_path.read_text(encoding="utf-8")).get(
                "model_type"
            )
        source_commit = None
        source_dirty = None
        if source_root.is_dir():
            source_commit = _git_output(source_root, "rev-parse", "HEAD")
            source_dirty = bool(_git_output(source_root, "status", "--porcelain"))
        local_failures: list[str] = []
        if model_observed != profile.checkpoint_sha256:
            local_failures.append("checkpoint SHA-256 mismatch")
        if model_type != profile.model_type:
            local_failures.append("checkpoint model_type mismatch")
        if source_commit != profile.source_commit:
            local_failures.append("source commit mismatch")
        if source_dirty:
            local_failures.append("source repository has local modifications")
        failures.extend(
            f"{profile.checkpoint_id}: {failure}" for failure in local_failures
        )
        checkpoint_results.append(
            {
                "checkpoint_id": profile.checkpoint_id,
                "checkpoint_path": str(checkpoint_root),
                "checkpoint_sha256_expected": profile.checkpoint_sha256,
                "checkpoint_sha256_observed": model_observed,
                "model_type_expected": profile.model_type,
                "model_type_observed": model_type,
                "source_path": str(source_root),
                "source_commit_expected": profile.source_commit,
                "source_commit_observed": source_commit,
                "source_dirty": source_dirty,
                "status": "PASS" if not local_failures else "FAIL",
            }
        )

    free_bytes = shutil.disk_usage(root).free
    minimum_free = int(protocol["execution"]["minimum_free_disk_bytes"])
    if free_bytes < minimum_free:
        failures.append("workspace free disk is below frozen minimum")
    return {
        "protocol_id": audit.protocol_id,
        "protocol_sha256": audit.protocol_sha256,
        "workspace_root": str(root),
        "dataset": {
            "path": str(dataset_path),
            "sha256_expected": str(dataset["arrow_sha256"]),
            "sha256_observed": dataset_observed,
            "status": (
                "PASS"
                if dataset_observed == str(dataset["arrow_sha256"])
                else "FAIL"
            ),
        },
        "checkpoints": checkpoint_results,
        "free_disk_bytes": free_bytes,
        "minimum_free_disk_bytes": minimum_free,
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_assets(load_protocol(args.protocol), args.workspace_root)
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
