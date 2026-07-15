#!/usr/bin/env python3
"""Verify the exact offline assets used by the frozen SWITCH C2 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


BASE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
ADAPTER_REVISION = "246fee75d774c02a110ea8608ac841a916dd5d35"
DATASET_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"

REQUIRED_FILES = {
    f"models--Qwen--Qwen3-8B/snapshots/{BASE_REVISION}/model-00001-of-00005.safetensors": (
        3_996_250_744,
        "31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f",
    ),
    f"models--Qwen--Qwen3-8B/snapshots/{BASE_REVISION}/model-00002-of-00005.safetensors": (
        3_993_160_032,
        "5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282",
    ),
    f"models--Qwen--Qwen3-8B/snapshots/{BASE_REVISION}/model-00003-of-00005.safetensors": (
        3_959_604_768,
        "c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836",
    ),
    f"models--Qwen--Qwen3-8B/snapshots/{BASE_REVISION}/model-00004-of-00005.safetensors": (
        3_187_841_392,
        "b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a",
    ),
    f"models--Qwen--Qwen3-8B/snapshots/{BASE_REVISION}/model-00005-of-00005.safetensors": (
        1_244_659_840,
        "20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff",
    ),
    f"models--LARK-Lab--SWITCH-Phase3-GRPO-LoRA-Qwen3-8B/snapshots/{ADAPTER_REVISION}/adapter_model.safetensors": (
        7_804_226_664,
        "0cdeafb628cdadd4c0fe21507ec7d61c98ace506d9bdad87775515de032a5e2c",
    ),
    f"datasets--HuggingFaceH4--MATH-500/snapshots/{DATASET_REVISION}/test.jsonl": (
        446_564,
        "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132",
    ),
}

REQUIRED_METADATA = {
    f"models--Qwen--Qwen3-8B/snapshots/{BASE_REVISION}": (
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ),
    f"models--LARK-Lab--SWITCH-Phase3-GRPO-LoRA-Qwen3-8B/snapshots/{ADAPTER_REVISION}": (
        "adapter_config.json",
        "adapter_model.safetensors",
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub-cache", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    hub_cache = args.hub_cache.resolve()

    records: list[dict[str, object]] = []
    failures: list[str] = []
    for relative, (expected_size, expected_sha256) in REQUIRED_FILES.items():
        path = hub_cache / relative
        if not path.is_file():
            failures.append(f"missing:{relative}")
            continue
        size = path.stat().st_size
        actual_sha256 = sha256(path)
        passed = size == expected_size and actual_sha256 == expected_sha256
        records.append(
            {
                "path": relative,
                "size": size,
                "expected_size": expected_size,
                "sha256": actual_sha256,
                "expected_sha256": expected_sha256,
                "status": "pass" if passed else "fail",
            }
        )
        if not passed:
            failures.append(f"integrity:{relative}")

    for directory, filenames in REQUIRED_METADATA.items():
        for filename in filenames:
            relative = f"{directory}/{filename}"
            if not (hub_cache / relative).is_file():
                failures.append(f"missing:{relative}")

    result = {
        "schema_version": 1,
        "verified_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hub_cache": str(hub_cache),
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "files": records,
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
