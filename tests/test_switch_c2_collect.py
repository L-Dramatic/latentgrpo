import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from research.coordinate_invariance.switch_c2_collect import (
    PRIMARY_ARTIFACTS,
    artifact_status,
    collect_bundle,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_artifact(root: Path, relative: str, status: str = "pass") -> None:
    _write(
        root / relative,
        json.dumps(
            {
                "status": status,
                "phase": "identity",
                "config_sha256": "config-hash",
                "implementation_sha256": {"runner": "implementation-hash"},
            }
        ),
    )


def test_artifact_status_distinguishes_missing_and_unreadable(tmp_path: Path) -> None:
    _write(tmp_path / PRIMARY_ARTIFACTS[0], "not-json")
    _write(tmp_path / PRIMARY_ARTIFACTS[1], "[]")
    statuses = artifact_status(tmp_path)

    assert statuses[PRIMARY_ARTIFACTS[0]]["state"] == "unreadable"
    assert statuses[PRIMARY_ARTIFACTS[1]]["state"] == "unreadable"
    assert statuses[PRIMARY_ARTIFACTS[2]]["state"] == "missing"


def test_collect_bundle_has_verified_manifest_and_runtime_evidence(
    tmp_path: Path,
) -> None:
    _write_artifact(tmp_path, PRIMARY_ARTIFACTS[0])
    _write(tmp_path / "RESEARCH_IDEA_ARCHIVE.md", "frozen evidence\n")
    _write(
        tmp_path
        / "artifacts"
        / "coordinate_invariance"
        / "journals"
        / "identity.jsonl",
        '{"rank": 0}\n',
    )
    _write(
        tmp_path / "artifacts" / "coordinate_invariance" / "switch_c2.log",
        "run log\n",
    )
    output = tmp_path / "return" / "c2.tar.gz"

    report = collect_bundle(tmp_path, output, required_stage="identity")

    assert output.is_file()
    assert report["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())
        assert PRIMARY_ARTIFACTS[0] in names
        assert "artifacts/coordinate_invariance/journals/identity.jsonl" in names
        assert "artifacts/coordinate_invariance/switch_c2.log" in names
        assert "collection/C2_ENVIRONMENT.json" in names
        assert "collection/C2_BUNDLE_MANIFEST.json" in names

        handle = archive.extractfile("collection/C2_BUNDLE_MANIFEST.json")
        assert handle is not None
        manifest = json.loads(handle.read().decode("utf-8"))
        for record in manifest["files"]:
            member = archive.extractfile(record["path"])
            assert member is not None
            data = member.read()
            assert len(data) == record["bytes"]
            assert hashlib.sha256(data).hexdigest() == record["sha256"]


def test_collect_bundle_refuses_incomplete_required_stage(tmp_path: Path) -> None:
    _write_artifact(tmp_path, PRIMARY_ARTIFACTS[0])

    with pytest.raises(ValueError, match="required stage 'eligibility'"):
        collect_bundle(
            tmp_path,
            tmp_path / "c2.tar.gz",
            required_stage="eligibility",
        )
