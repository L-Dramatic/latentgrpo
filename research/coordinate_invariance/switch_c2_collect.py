from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PRIMARY_ARTIFACTS = (
    "artifacts/coordinate_invariance/switch_checkpoint_identity_smoke_v1.json",
    "artifacts/coordinate_invariance/switch_c2_eligibility_v1.json",
    "artifacts/coordinate_invariance/switch_c2_calibration_v1.json",
    "artifacts/coordinate_invariance/switch_c2_test_v1.json",
)

STAGE_REQUIREMENTS = {
    "none": (),
    "identity": PRIMARY_ARTIFACTS[:1],
    "eligibility": PRIMARY_ARTIFACTS[:2],
    "calibration": PRIMARY_ARTIFACTS[:3],
    "test": PRIMARY_ARTIFACTS,
}

FIXED_EVIDENCE = (
    ".gitattributes",
    "pytest.ini",
    "RESEARCH_IDEA_ARCHIVE.md",
    "artifacts/coordinate_invariance/fctr_coconut_smoke_v1c.json",
    "artifacts/coordinate_invariance/switch_c2_prompt_order_v1.json",
    "artifacts/coordinate_invariance/switch_c2_source_preflight_v1.json",
    "research/coordinate_invariance/AUTODL_SWITCH_C2_RUNBOOK_ZH.md",
    "research/coordinate_invariance/EXPERIMENT_LOG.md",
    "research/coordinate_invariance/FCTR_REVIVAL_PREREGISTRATION.md",
    "research/coordinate_invariance/SOURCE_MANIFEST.json",
    "research/coordinate_invariance/SWITCH_C2_PREREGISTRATION.md",
    "research/coordinate_invariance/configs/switch_c2_scientific_gate_v1.json",
    "research/coordinate_invariance/configs/switch_checkpoint_identity_smoke_v1.json",
    "research/coordinate_invariance/requirements-switch-c2.txt",
    "research/coordinate_invariance/run_switch_c2_autodl.sh",
    "research/coordinate_invariance/switch_c2_collect.py",
    "research/coordinate_invariance/real_models/switch.py",
    "research/coordinate_invariance/switch_checkpoint_identity_smoke.py",
    "research/coordinate_invariance/switch_c2_eligibility_scan.py",
    "research/coordinate_invariance/switch_c2_geometry.py",
    "research/coordinate_invariance/switch_c2_scientific_gate.py",
    "research/coordinate_invariance/switch_prompt_order.py",
    "research/coordinate_invariance/fctr.py",
    "research/coordinate_invariance/charts.py",
    "research/coordinate_invariance/metrics.py",
    "research/coordinate_invariance/statistics.py",
    "tests/test_switch_source_equivalence.py",
    "tests/test_switch_prompt_order.py",
    "tests/test_switch_c2_geometry.py",
    "tests/test_switch_c2_protocol.py",
    "tests/test_switch_c2_collect.py",
)

PACKAGE_NAMES = (
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "huggingface-hub",
    "safetensors",
    "tokenizers",
    "datasets",
    "numpy",
    "psutil",
    "pytest",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _torch_environment() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on collection host
        return {"available": False, "import_error": repr(exc)}

    cuda_available = bool(torch.cuda.is_available())
    report: dict[str, Any] = {
        "available": True,
        "version": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "cuda_available": cuda_available,
    }
    if cuda_available:
        devices = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                }
            )
        report["devices"] = devices
    return report


def environment_report(root: Path) -> dict[str, Any]:
    porcelain = _run_git(root, "status", "--porcelain=v1")
    return {
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(root),
        "git": {
            "head": _run_git(root, "rev-parse", "HEAD"),
            "branch": _run_git(root, "branch", "--show-current"),
            "dirty": None if porcelain is None else bool(porcelain),
            "changed_paths": [] if not porcelain else porcelain.splitlines(),
        },
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "executable": sys.executable,
            "cpu_count": os.cpu_count(),
        },
        "packages": _package_versions(),
        "torch": _torch_environment(),
    }


def artifact_status(root: Path) -> dict[str, dict[str, Any]]:
    statuses: dict[str, dict[str, Any]] = {}
    for relative in PRIMARY_ARTIFACTS:
        path = root / relative
        if not path.is_file():
            statuses[relative] = {"state": "missing"}
            continue
        entry: dict[str, Any] = {
            "state": "present",
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            entry.update(state="unreadable", error=repr(exc))
        else:
            if not isinstance(report, dict):
                entry.update(
                    state="unreadable", error="top-level JSON is not an object"
                )
            else:
                entry.update(
                    status=report.get("status"),
                    phase=report.get("phase"),
                    config_sha256=report.get("config_sha256"),
                    implementation_sha256=report.get("implementation_sha256"),
                )
        statuses[relative] = entry
    return statuses


def _selected_files(root: Path) -> list[Path]:
    selected: set[Path] = set()
    for relative in (*PRIMARY_ARTIFACTS, *FIXED_EVIDENCE):
        path = (root / relative).resolve()
        if path.is_file():
            selected.add(path)

    artifact_root = root / "artifacts" / "coordinate_invariance"
    for pattern in ("journals/*.jsonl", "*.log"):
        for path in artifact_root.glob(pattern):
            resolved = path.resolve()
            if resolved.is_file():
                selected.add(resolved)
    for path in selected:
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"selected evidence escapes workspace root: {path}") from exc
    return sorted(selected, key=lambda item: item.relative_to(root).as_posix())


def _tar_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mtime = 0
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(data))


def _verify_bundle(path: Path) -> None:
    with tarfile.open(path, "r:gz") as archive:
        manifest_member = archive.getmember("collection/C2_BUNDLE_MANIFEST.json")
        handle = archive.extractfile(manifest_member)
        if handle is None:
            raise ValueError("bundle manifest could not be read")
        manifest = json.loads(handle.read().decode("utf-8"))
        for record in manifest["files"]:
            member = archive.getmember(record["path"])
            file_handle = archive.extractfile(member)
            if file_handle is None:
                raise ValueError(f"bundle member {record['path']} could not be read")
            data = file_handle.read()
            if len(data) != int(record["bytes"]):
                raise ValueError(f"bundle member {record['path']} has wrong size")
            if _sha256_bytes(data) != record["sha256"]:
                raise ValueError(f"bundle member {record['path']} has wrong SHA-256")


def _require_stage(
    statuses: dict[str, dict[str, Any]], required_stage: str
) -> None:
    failures = [
        relative
        for relative in STAGE_REQUIREMENTS[required_stage]
        if statuses[relative]["state"] != "present"
    ]
    if failures:
        joined = ", ".join(failures)
        raise ValueError(
            f"required stage {required_stage!r} is incomplete or unreadable: {joined}"
        )


def collect_bundle(
    workspace_root: Path,
    output_path: Path,
    required_stage: str = "none",
) -> dict[str, Any]:
    root = workspace_root.resolve()
    if not root.is_dir():
        raise ValueError(f"workspace root does not exist: {root}")
    statuses = artifact_status(root)
    _require_stage(statuses, required_stage)

    environment = environment_report(root)
    environment_bytes = json.dumps(
        environment, indent=2, sort_keys=True, ensure_ascii=False
    ).encode("utf-8")

    payloads: list[tuple[str, bytes]] = []
    for path in _selected_files(root):
        payloads.append((path.relative_to(root).as_posix(), _tar_bytes(path)))
    payloads.append(("collection/C2_ENVIRONMENT.json", environment_bytes))

    records = [
        {"path": name, "bytes": len(data), "sha256": _sha256_bytes(data)}
        for name, data in payloads
    ]
    manifest = {
        "schema_version": 1,
        "required_stage": required_stage,
        "artifact_status": statuses,
        "files": records,
    }
    manifest_bytes = json.dumps(
        manifest, indent=2, sort_keys=True, ensure_ascii=False
    ).encode("utf-8")

    output = output_path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with tarfile.open(temporary_path, "w:gz") as archive:
            for name, data in payloads:
                _add_bytes(archive, name, data)
            _add_bytes(
                archive, "collection/C2_BUNDLE_MANIFEST.json", manifest_bytes
            )
        _verify_bundle(temporary_path)
        temporary_path.replace(output)
    finally:
        temporary_path.unlink(missing_ok=True)

    return {
        "output": str(output),
        "sha256": _sha256_file(output),
        "bytes": output.stat().st_size,
        "required_stage": required_stage,
        "artifact_status": statuses,
    }


def status_report(workspace_root: Path) -> dict[str, Any]:
    root = workspace_root.resolve()
    return {
        "workspace_root": str(root),
        "git_head": _run_git(root, "rev-parse", "HEAD"),
        "artifacts": artifact_status(root),
    }


def _print_summary(report: dict[str, Any]) -> None:
    if "output" in report:
        print(
            f"collection: output={report['output']} bytes={report['bytes']} "
            f"sha256={report['sha256']}"
        )
    status_key = "artifact_status" if "output" in report else "artifacts"
    for path, state in report[status_key].items():
        detail = state["state"]
        if state.get("status") is not None:
            detail += f" status={state['status']}"
        print(f"artifact: {path} {detail}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect and integrity-package SWITCH C2 evidence."
    )
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-stage", choices=tuple(STAGE_REQUIREMENTS), default="none"
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.output is None:
        report = status_report(args.workspace_root)
    else:
        report = collect_bundle(
            args.workspace_root, args.output, required_stage=args.require_stage
        )
    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
