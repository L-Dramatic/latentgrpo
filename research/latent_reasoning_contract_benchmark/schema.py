"""Fail-closed schema validation for the LRC-Bench source manifest."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
LAYER_IDS = (
    "provenance",
    "recurrence",
    "execution",
    "learning",
    "policy_measure",
    "outcome_relevance",
)
LAYER_STATUSES = {"source_anchored", "pending_runtime", "not_applicable"}
CHECKPOINT_PROVENANCE = {"official", "independent_reproduction"}


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_relative_path(value: Any, label: str) -> str:
    path = Path(_require_nonempty_string(value, label))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must stay inside the workspace")
    return path.as_posix()


def _validate_source_file(value: Any, method_id: str) -> None:
    entry = _require_mapping(value, f"{method_id} source file")
    _require_relative_path(entry.get("path"), f"{method_id} source file path")
    sha256 = _require_nonempty_string(
        entry.get("sha256"), f"{method_id} source file sha256"
    )
    if not SHA256_RE.fullmatch(sha256):
        raise ValueError(f"{method_id} source file sha256 must be lowercase SHA-256")
    anchors = entry.get("anchors")
    if not isinstance(anchors, list) or not anchors:
        raise ValueError(f"{method_id} source file requires text anchors")
    for anchor in anchors:
        _require_nonempty_string(anchor, f"{method_id} source anchor")


def _validate_checkpoint(value: Any, method_id: str) -> None:
    checkpoint = _require_mapping(value, f"{method_id} checkpoint")
    _require_nonempty_string(checkpoint.get("id"), f"{method_id} checkpoint id")
    revision = _require_nonempty_string(
        checkpoint.get("revision"), f"{method_id} checkpoint revision"
    )
    if not COMMIT_RE.fullmatch(revision):
        raise ValueError(f"{method_id} checkpoint revision must be immutable")
    provenance = checkpoint.get("provenance")
    if provenance not in CHECKPOINT_PROVENANCE:
        raise ValueError(f"{method_id} checkpoint provenance is invalid")
    _require_relative_path(
        checkpoint.get("local_path"), f"{method_id} checkpoint local_path"
    )
    local_files = checkpoint.get("local_files")
    if not isinstance(local_files, list):
        raise ValueError(f"{method_id} checkpoint local_files must be a list")
    for raw_file in local_files:
        file_entry = _require_mapping(raw_file, f"{method_id} checkpoint file")
        _require_relative_path(
            file_entry.get("path"), f"{method_id} checkpoint file path"
        )
        sha256 = _require_nonempty_string(
            file_entry.get("sha256"), f"{method_id} checkpoint file sha256"
        )
        if not SHA256_RE.fullmatch(sha256):
            raise ValueError(
                f"{method_id} checkpoint file sha256 must be lowercase SHA-256"
            )


def _validate_layers(value: Any, method_id: str, stochastic: bool) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{method_id} contract_layers must be a list")
    layers = {}
    for raw_layer in value:
        layer = _require_mapping(raw_layer, f"{method_id} layer")
        layer_id = _require_nonempty_string(layer.get("id"), f"{method_id} layer id")
        if layer_id in layers:
            raise ValueError(f"{method_id} has duplicate layer {layer_id}")
        status = layer.get("status")
        if status not in LAYER_STATUSES:
            raise ValueError(f"{method_id} layer {layer_id} has invalid status")
        evidence = layer.get("evidence")
        if not isinstance(evidence, list):
            raise ValueError(f"{method_id} layer {layer_id} evidence must be a list")
        if status == "source_anchored" and not evidence:
            raise ValueError(f"{method_id} layer {layer_id} lacks source evidence")
        if status == "not_applicable":
            _require_nonempty_string(
                layer.get("reason"), f"{method_id} layer {layer_id} N/A reason"
            )
        layers[layer_id] = layer
    if tuple(layers) != LAYER_IDS:
        raise ValueError(f"{method_id} must declare the six ordered contract layers")
    policy_status = layers["policy_measure"]["status"]
    if stochastic and policy_status == "not_applicable":
        raise ValueError(f"{method_id} is stochastic but omits policy_measure")
    if not stochastic and policy_status != "not_applicable":
        raise ValueError(f"{method_id} is deterministic but scores a policy measure")
    if layers["outcome_relevance"]["status"] != "pending_runtime":
        raise ValueError(f"{method_id} outcome_relevance must remain pending at Gate -1")


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    _require_nonempty_string(manifest.get("benchmark_id"), "benchmark_id")
    scope = _require_mapping(manifest.get("scope"), "scope")
    if scope.get("broad_mechanism_claim") != "excluded_by_collision":
        raise ValueError("the collided broad mechanism claim must remain excluded")
    if scope.get("stage") != "gate_minus_one":
        raise ValueError("source manifest is only valid for Gate -1")

    methods = manifest.get("methods")
    if not isinstance(methods, list) or len(methods) != 4:
        raise ValueError("Gate -1 requires exactly four methods")
    ids: set[str] = set()
    families: set[str] = set()
    stochastic_count = 0
    for raw_method in methods:
        method = _require_mapping(raw_method, "method")
        method_id = _require_nonempty_string(method.get("id"), "method id")
        if method_id in ids:
            raise ValueError(f"duplicate method id: {method_id}")
        ids.add(method_id)
        families.add(_require_nonempty_string(method.get("family"), "method family"))
        stochastic = method.get("stochastic")
        if not isinstance(stochastic, bool):
            raise ValueError(f"{method_id} stochastic must be boolean")
        stochastic_count += int(stochastic)
        _require_nonempty_string(method.get("paper"), f"{method_id} paper")

        source = _require_mapping(method.get("source"), f"{method_id} source")
        _require_relative_path(source.get("path"), f"{method_id} source path")
        _require_nonempty_string(source.get("repository"), f"{method_id} repository")
        commit = _require_nonempty_string(source.get("commit"), f"{method_id} commit")
        if not COMMIT_RE.fullmatch(commit):
            raise ValueError(f"{method_id} commit must be a full lowercase git hash")
        files = source.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError(f"{method_id} source files must be non-empty")
        for file_entry in files:
            _validate_source_file(file_entry, method_id)

        _validate_checkpoint(method.get("checkpoint"), method_id)
        _validate_layers(method.get("contract_layers"), method_id, stochastic)

    if len(families) < 4:
        raise ValueError("Gate -1 requires four distinct mechanism families")
    if stochastic_count < 2:
        raise ValueError("Gate -1 requires two stochastic latent-RL methods")


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    return manifest
