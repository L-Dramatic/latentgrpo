from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor


TRACE_SCHEMA_VERSION = 1
_TRACE_FILE = "trace.pt"
_MANIFEST_FILE = "manifest.json"


def _clone_tensor_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("tensor-tree dictionary keys must be strings")
        return {key: _clone_tensor_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tensor_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tensor_tree(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported trace value type: {type(value).__name__}")


def _validate_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    copied = _clone_tensor_tree(dict(metadata))
    try:
        json.dumps(copied, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise TypeError("trace metadata must be JSON serializable") from error
    return copied


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class LatentStepRecord:
    sample_id: str
    step_index: int
    prefix_state: Any
    latent: Tensor
    rng_seed: int
    chart_name: str = "native"
    metadata: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def capture(
        cls,
        *,
        sample_id: str,
        step_index: int,
        prefix_state: Any,
        latent: Tensor,
        rng_seed: int,
        chart_name: str = "native",
        metadata: Mapping[str, object] | None = None,
    ) -> "LatentStepRecord":
        if not sample_id:
            raise ValueError("sample_id must be non-empty")
        if not isinstance(step_index, int) or step_index < 0:
            raise ValueError("step_index must be a non-negative integer")
        if not isinstance(rng_seed, int):
            raise TypeError("rng_seed must be an integer")
        if not chart_name:
            raise ValueError("chart_name must be non-empty")
        if not isinstance(latent, Tensor) or latent.ndim != 1:
            raise ValueError("latent must be a rank-1 tensor")
        if not latent.is_floating_point() or not torch.isfinite(latent).all():
            raise ValueError("latent must contain finite floating-point values")
        return cls(
            sample_id=sample_id,
            step_index=step_index,
            prefix_state=_clone_tensor_tree(prefix_state),
            latent=latent.detach().cpu().clone(),
            rng_seed=rng_seed,
            chart_name=chart_name,
            metadata=_validate_metadata(metadata or {}),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "step_index": self.step_index,
            "prefix_state": _clone_tensor_tree(self.prefix_state),
            "latent": self.latent.detach().cpu().clone(),
            "rng_seed": self.rng_seed,
            "chart_name": self.chart_name,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "LatentStepRecord":
        required = {
            "sample_id",
            "step_index",
            "prefix_state",
            "latent",
            "rng_seed",
            "chart_name",
            "metadata",
        }
        if set(payload) != required:
            raise ValueError("trace record payload has an unexpected schema")
        return cls.capture(
            sample_id=str(payload["sample_id"]),
            step_index=int(payload["step_index"]),
            prefix_state=payload["prefix_state"],
            latent=payload["latent"],
            rng_seed=int(payload["rng_seed"]),
            chart_name=str(payload["chart_name"]),
            metadata=payload["metadata"],
        )


class LatentTrace:
    """A replayable, integrity-checked collection of latent-step snapshots."""

    def __init__(self, records: list[LatentStepRecord] | None = None) -> None:
        self._records: list[LatentStepRecord] = []
        self._keys: set[tuple[str, int]] = set()
        for record in records or []:
            self.append(record)

    @property
    def records(self) -> tuple[LatentStepRecord, ...]:
        return tuple(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def append(self, record: LatentStepRecord) -> None:
        if not isinstance(record, LatentStepRecord):
            raise TypeError("record must be a LatentStepRecord")
        key = (record.sample_id, record.step_index)
        if key in self._keys:
            raise ValueError(f"duplicate trace step: {key}")
        self._records.append(record)
        self._keys.add(key)

    def get(self, sample_id: str, step_index: int) -> LatentStepRecord:
        key = (sample_id, step_index)
        for record in self._records:
            if (record.sample_id, record.step_index) == key:
                return record
        raise KeyError(key)

    def save(self, directory: str | os.PathLike[str]) -> Path:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "records": [record.to_payload() for record in self._records],
        }

        trace_fd, trace_temp_name = tempfile.mkstemp(prefix="trace-", suffix=".pt", dir=target)
        manifest_fd, manifest_temp_name = tempfile.mkstemp(
            prefix="manifest-", suffix=".json", dir=target
        )
        os.close(trace_fd)
        os.close(manifest_fd)
        trace_temp = Path(trace_temp_name)
        manifest_temp = Path(manifest_temp_name)
        try:
            torch.save(payload, trace_temp)
            manifest = {
                "schema_version": TRACE_SCHEMA_VERSION,
                "record_count": len(self._records),
                "trace_file": _TRACE_FILE,
                "sha256": _sha256_file(trace_temp),
            }
            manifest_temp.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.replace(trace_temp, target / _TRACE_FILE)
            os.replace(manifest_temp, target / _MANIFEST_FILE)
        finally:
            trace_temp.unlink(missing_ok=True)
            manifest_temp.unlink(missing_ok=True)
        return target

    @classmethod
    def load(
        cls,
        directory: str | os.PathLike[str],
        *,
        map_location: str | torch.device = "cpu",
    ) -> "LatentTrace":
        target = Path(directory)
        manifest_path = target / _MANIFEST_FILE
        trace_path = target / _TRACE_FILE
        if not manifest_path.is_file() or not trace_path.is_file():
            raise FileNotFoundError("trace directory must contain manifest.json and trace.pt")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_keys = {"schema_version", "record_count", "trace_file", "sha256"}
        if set(manifest) != expected_keys:
            raise ValueError("trace manifest has an unexpected schema")
        if manifest["schema_version"] != TRACE_SCHEMA_VERSION:
            raise ValueError("unsupported trace schema version")
        if manifest["trace_file"] != _TRACE_FILE:
            raise ValueError("trace manifest points to an unexpected data file")
        if _sha256_file(trace_path) != manifest["sha256"]:
            raise ValueError("trace integrity check failed")

        payload = torch.load(trace_path, map_location=map_location, weights_only=True)
        if not isinstance(payload, dict) or set(payload) != {"schema_version", "records"}:
            raise ValueError("trace payload has an unexpected schema")
        if payload["schema_version"] != TRACE_SCHEMA_VERSION:
            raise ValueError("trace payload schema does not match the reader")
        records = [LatentStepRecord.from_payload(item) for item in payload["records"]]
        if len(records) != manifest["record_count"]:
            raise ValueError("trace record count does not match the manifest")
        return cls(records)

