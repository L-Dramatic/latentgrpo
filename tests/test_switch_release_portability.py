from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_prerequisite_artifacts_are_lf_and_match_config() -> None:
    config = json.loads(
        (
            ROOT
            / "research/coordinate_invariance/configs/switch_c2_scientific_gate_v1.json"
        ).read_text(encoding="utf-8")
    )
    prerequisites = config["prerequisites"]
    records = (
        (
            prerequisites["source_preflight_artifact"],
            prerequisites["source_preflight_artifact_sha256"],
        ),
        (
            prerequisites["coconut_c1_artifact"],
            prerequisites["coconut_c1_artifact_sha256"],
        ),
        (
            config["dataset"]["prompt_order_artifact"],
            config["dataset"]["prompt_order_artifact_sha256"],
        ),
    )
    for relative, expected in records:
        path = ROOT / relative
        payload = path.read_bytes()
        assert b"\r\n" not in payload
        assert hashlib.sha256(payload).hexdigest() == expected


def test_identity_and_scientific_config_bindings_are_current() -> None:
    identity_path = (
        ROOT
        / "research/coordinate_invariance/configs/switch_checkpoint_identity_smoke_v1.json"
    )
    scientific_path = (
        ROOT / "research/coordinate_invariance/configs/switch_c2_scientific_gate_v1.json"
    )
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    scientific = json.loads(scientific_path.read_text(encoding="utf-8"))

    identity_canonical = json.dumps(
        identity, sort_keys=True, separators=(",", ":")
    ).encode()
    assert hashlib.sha256(identity_canonical).hexdigest() == scientific[
        "prerequisites"
    ]["checkpoint_identity_config_sha256"]
    assert identity["dataset"]["prompt_order_sha256"] == scientific["dataset"][
        "prompt_order_artifact_sha256"
    ]
    assert _sha256(scientific_path) == (
        "7efe9caf1b0ef35f8c63149c4ce38d9319d25689c410449358c461ba8a471056"
    )
