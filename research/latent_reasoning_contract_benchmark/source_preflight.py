"""Zero-GPU source and checkpoint readiness audit for LRC-Bench Gate -1."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from .schema import load_manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), *args], text=True, encoding="utf-8"
    ).strip()


def _inside(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"path escaped workspace: {relative}")
    return candidate


def _audit_method(root: Path, method: dict[str, Any]) -> dict[str, Any]:
    method_id = method["id"]
    source = method["source"]
    source_root = _inside(root, source["path"])
    failures: list[str] = []
    observed_commit = None
    observed_remote = None
    if not source_root.is_dir():
        failures.append("source repository missing")
    else:
        try:
            observed_commit = _git(source_root, "rev-parse", "HEAD")
            observed_remote = _git(source_root, "remote", "get-url", "origin")
        except (OSError, subprocess.CalledProcessError):
            failures.append("source repository is not readable by git")
    if observed_commit != source["commit"]:
        failures.append("source commit mismatch")
    normalized_remote = (observed_remote or "").removesuffix(".git").lower()
    expected_remote = source["repository"].removesuffix(".git").lower()
    if normalized_remote != expected_remote:
        failures.append("source remote mismatch")

    source_files = []
    for entry in source["files"]:
        path = _inside(source_root, entry["path"])
        observed_sha = _sha256(path) if path.is_file() else None
        missing_anchors = []
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            missing_anchors = [anchor for anchor in entry["anchors"] if anchor not in text]
        if observed_sha != entry["sha256"]:
            failures.append(f"source file mismatch: {entry['path']}")
        if missing_anchors:
            failures.append(f"source anchors missing: {entry['path']}")
        source_files.append(
            {
                "path": entry["path"],
                "sha256_expected": entry["sha256"],
                "sha256_observed": observed_sha,
                "missing_anchors": missing_anchors,
            }
        )

    checkpoint = method["checkpoint"]
    checkpoint_root = _inside(root, checkpoint["local_path"])
    checkpoint_files = []
    local_ready = bool(checkpoint["local_files"])
    for entry in checkpoint["local_files"]:
        path = _inside(checkpoint_root, entry["path"])
        observed_sha = _sha256(path) if path.is_file() else None
        if observed_sha != entry["sha256"]:
            local_ready = False
        checkpoint_files.append(
            {
                "path": entry["path"],
                "sha256_expected": entry["sha256"],
                "sha256_observed": observed_sha,
            }
        )

    return {
        "id": method_id,
        "family": method["family"],
        "stochastic": method["stochastic"],
        "source_pass": not failures,
        "source_failures": failures,
        "source_commit_expected": source["commit"],
        "source_commit_observed": observed_commit,
        "source_remote_observed": observed_remote,
        "source_files": source_files,
        "checkpoint": {
            "id": checkpoint["id"],
            "revision": checkpoint["revision"],
            "provenance": checkpoint["provenance"],
            "public": checkpoint["public"],
            "local_ready": local_ready,
            "files": checkpoint_files,
        },
    }


def run_preflight(root: Path, manifest_path: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest = load_manifest(manifest_path.resolve())
    methods = [_audit_method(root, method) for method in manifest["methods"]]
    counts = {
        "methods": len(methods),
        "families": len({method["family"] for method in methods}),
        "stochastic_methods": sum(method["stochastic"] for method in methods),
        "source_pass": sum(method["source_pass"] for method in methods),
        "public_checkpoints": sum(method["checkpoint"]["public"] for method in methods),
        "official_checkpoints": sum(
            method["checkpoint"]["provenance"] == "official" for method in methods
        ),
        "local_checkpoints": sum(
            method["checkpoint"]["local_ready"] for method in methods
        ),
    }
    criteria = {
        "four_methods": counts["methods"] == 4,
        "four_mechanism_families": counts["families"] >= 4,
        "two_stochastic_rl_methods": counts["stochastic_methods"] >= 2,
        "all_sources_pinned_and_verified": counts["source_pass"] == 4,
        "all_checkpoints_public_and_revision_pinned": counts["public_checkpoints"] == 4,
        "at_least_three_official_checkpoints": counts["official_checkpoints"] >= 3,
        "at_least_three_checkpoints_local": counts["local_checkpoints"] >= 3,
        "broad_collision_explicitly_excluded": (
            manifest["scope"]["broad_mechanism_claim"] == "excluded_by_collision"
        ),
    }
    passed = all(criteria.values())
    return {
        "benchmark_id": manifest["benchmark_id"],
        "stage": "gate_minus_one",
        "gpu_used": False,
        "decision": "PASS_GATE_MINUS_ONE" if passed else "HOLD_GATE_MINUS_ONE",
        "counts": counts,
        "criteria": criteria,
        "methods": methods,
        "next_required_action": (
            "implement source-native fixture replay"
            if passed and counts["local_checkpoints"] == 4
            else (
                "download and hash missing pinned checkpoints, then implement source-native fixture replay"
                if passed
                else "repair only failed provenance or source-readiness controls"
            )
        ),
        "scientific_evidence": "none; this report establishes feasibility only",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("SOURCE_MANIFEST.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_preflight(args.workspace, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "counts": report["counts"]}))
    return 0 if report["decision"] == "PASS_GATE_MINUS_ONE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
