"""Deterministic staged task manifests and crash-safe result shards for PCMC."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from .checkpoint_protocol import (
    METHODS,
    action_seed,
    checkpoint_profiles,
    continuation_seed,
    example_ids,
    load_protocol,
    selected_confirmation_ids,
    split_example_ids,
    validate_protocol,
)
from .gate_a_analysis import load_jsonl, summarize_gate_a, write_json_atomic


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _manifest(protocol: dict[str, Any], stage: str, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    audit = validate_protocol(protocol)
    keys = [str(task["task_key"]) for task in tasks]
    if len(keys) != len(set(keys)):
        raise ValueError("task manifest contains duplicate keys")
    task_sha256 = hashlib.sha256(_canonical_bytes(tasks)).hexdigest()
    return {
        "protocol_id": audit.protocol_id,
        "protocol_sha256": audit.protocol_sha256,
        "stage": stage,
        "task_count": len(tasks),
        "tasks_sha256": task_sha256,
        "tasks": tasks,
    }


def build_a0_manifest(protocol: dict[str, Any]) -> dict[str, Any]:
    splits = split_example_ids(protocol)
    tasks: list[dict[str, Any]] = []
    for checkpoint_index, profile in enumerate(checkpoint_profiles(protocol)):
        for example_index, example_id in enumerate(example_ids(protocol)):
            tasks.append(
                {
                    "task_key": f"a0|{profile.checkpoint_id}|{example_id}",
                    "checkpoint_id": profile.checkpoint_id,
                    "example_id": example_id,
                    "dataset_partition": splits[example_id],
                    "action_seed": action_seed(
                        protocol, checkpoint_index, example_index
                    ),
                }
            )
    return _manifest(protocol, "A0_ONE_STEP_CLOSURE", tasks)


def build_a1_manifest(
    protocol: dict[str, Any], a0_records: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    a0_summary = summarize_gate_a(
        protocol, a0_records, completed_stage="A0"
    )
    if a0_summary["overall_decision"] != "ADVANCE_TO_A1_COLLECTION":
        raise ValueError("A0 did not authorize A1 task construction")
    splits = split_example_ids(protocol)
    index_by_id = {
        example_id: index for index, example_id in enumerate(example_ids(protocol))
    }
    tasks: list[dict[str, Any]] = []
    for checkpoint_index, profile in enumerate(checkpoint_profiles(protocol)):
        selection = selected_confirmation_ids(
            protocol, a0_records, checkpoint_id=profile.checkpoint_id
        )
        gate_a0 = protocol["gate_a0"]
        if selection["high_threshold"] < float(
            gate_a0["minimum_calibration_high_gap_js_nats"]
        ):
            raise ValueError(f"{profile.checkpoint_id} did not pass A0 JS")
        if selection["high_candidate_count"] < int(
            gate_a0["minimum_confirmation_high_gap_prompts"]
        ):
            raise ValueError(f"{profile.checkpoint_id} lacks confirmation high-gap prompts")
        selected = [
            ("high", example_id) for example_id in selection["selected_high"]
        ] + [("low", example_id) for example_id in selection["selected_low"]]
        for stratum, example_id in selected:
            example_index = index_by_id[example_id]
            for method in METHODS:
                for replicate_index in range(
                    validate_protocol(protocol).continuation_replicates
                ):
                    tasks.append(
                        {
                            "task_key": (
                                f"a1|{profile.checkpoint_id}|{example_id}|"
                                f"{method}|r{replicate_index}"
                            ),
                            "checkpoint_id": profile.checkpoint_id,
                            "example_id": example_id,
                            "dataset_partition": splits[example_id],
                            "stratum": stratum,
                            "method": method,
                            "replicate_index": replicate_index,
                            "continuation_seed": continuation_seed(
                                protocol,
                                checkpoint_index,
                                example_index,
                                replicate_index,
                            ),
                        }
                    )
    return _manifest(protocol, "A1_CAUSAL_CONTINUATION", tasks)


class RecordStore:
    """One immutable JSON file per record, committed with atomic rename."""

    def __init__(self, root: Path):
        self.root = root
        self.shard_root = root / "record_shards"
        self.shard_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _filename(key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest() + ".json"

    def _path(self, key: str) -> Path:
        return self.shard_root / self._filename(key)

    def put(self, record: dict[str, Any]) -> str:
        key = str(record.get("key", ""))
        if not key:
            raise ValueError("record requires a non-empty key")
        payload = _canonical_bytes(record) + b"\n"
        destination = self._path(key)
        if destination.exists():
            existing = destination.read_bytes()
            if existing != payload:
                raise ValueError(f"immutable record mismatch for {key}")
            return "RESUMED"
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
        return "CREATED"

    def records(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        keys: set[str] = set()
        for path in sorted(self.shard_root.glob("*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or not value.get("key"):
                raise ValueError(f"invalid record shard: {path.name}")
            key = str(value["key"])
            if path.name != self._filename(key):
                raise ValueError(f"record shard filename mismatch: {path.name}")
            if key in keys:
                raise ValueError(f"duplicate record key in shards: {key}")
            keys.add(key)
            result.append(value)
        return sorted(result, key=lambda record: str(record["key"]))

    def compact(self, destination: Path) -> dict[str, Any]:
        records = self.records()
        payload = b"".join(_canonical_bytes(record) + b"\n" for record in records)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
        return {
            "record_count": len(records),
            "records_sha256": hashlib.sha256(payload).hexdigest(),
            "destination": str(destination),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--stage", choices=("A0", "A1"), required=True)
    parser.add_argument("--a0-records", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    if args.stage == "A0":
        manifest = build_a0_manifest(protocol)
    else:
        if args.a0_records is None:
            parser.error("--a0-records is required for A1")
        manifest = build_a1_manifest(protocol, load_jsonl(args.a0_records))
    write_json_atomic(args.output, manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key != "tasks"}, indent=2))


if __name__ == "__main__":
    main()
