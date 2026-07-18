"""Label-blind, revision-pinned prompt selection for LRC-Bench."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET_ID = "openai/gsm8k"
DEFAULT_DATASET_REVISION = "740312add88f781978c0658806c59bc2815b9866"
DEFAULT_SALT = "lrc-gate-a-deterministic-v1"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def extract_final_answer(answer: str) -> str:
    if not isinstance(answer, str) or "####" not in answer:
        raise ValueError("GSM8K answer must contain a final-answer delimiter")
    final = answer.rsplit("####", 1)[1].strip().replace(",", "")
    if not final:
        raise ValueError("GSM8K final answer must be non-empty")
    return final


def select_records(
    rows: Iterable[dict[str, Any]], *, salt: str, calibration_count: int, confirmation_count: int
) -> list[dict[str, Any]]:
    if not salt or calibration_count < 1 or confirmation_count < 1:
        raise ValueError("selection requires a salt and two positive split sizes")
    keyed = []
    for dataset_index, row in enumerate(rows):
        question = row.get("question")
        answer = row.get("answer")
        if not isinstance(question, str) or not question.strip() or not isinstance(answer, str):
            raise ValueError("every GSM8K row must contain question and answer strings")
        selection_hash = sha256_text(f"{salt}\0{question}")
        keyed.append((selection_hash, dataset_index, question, answer))
    keyed.sort(key=lambda item: (item[0], item[1]))
    count = calibration_count + confirmation_count
    if len(keyed) < count:
        raise ValueError("dataset is smaller than the frozen selection")

    records = []
    for rank, (selection_hash, dataset_index, question, answer) in enumerate(keyed[:count]):
        split = "calibration" if rank < calibration_count else "confirmation"
        final_answer = extract_final_answer(answer)
        records.append(
            {
                "selection_rank": rank,
                "split": split,
                "dataset_index": dataset_index,
                "selection_sha256": selection_hash,
                "question_sha256": sha256_text(question),
                "source_answer_sha256": sha256_text(answer),
                "target_sha256": sha256_text(final_answer),
                "question": question,
                "final_answer": final_answer,
            }
        )
    return records


def build_manifest(
    *, dataset_id: str, revision: str, salt: str, calibration_count: int, confirmation_count: int,
    cache_dir: Path,
) -> dict[str, Any]:
    from datasets import load_dataset

    dataset = load_dataset(
        dataset_id,
        "main",
        split="test",
        revision=revision,
        cache_dir=str(cache_dir),
    )
    records = select_records(
        dataset,
        salt=salt,
        calibration_count=calibration_count,
        confirmation_count=confirmation_count,
    )
    return {
        "schema_version": 1,
        "dataset": {
            "id": dataset_id,
            "revision": revision,
            "configuration": "main",
            "split": "test",
            "row_count": len(dataset),
        },
        "selection": {
            "salt": salt,
            "algorithm": "sort ascending by SHA256(salt + NUL + question), tie-break by dataset index",
            "label_blind": True,
            "calibration_count": calibration_count,
            "confirmation_count": confirmation_count,
        },
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("_models/hf_datasets"))
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--salt", default=DEFAULT_SALT)
    parser.add_argument("--calibration-count", type=int, default=32)
    parser.add_argument("--confirmation-count", type=int, default=32)
    args = parser.parse_args()
    manifest = build_manifest(
        dataset_id=args.dataset_id,
        revision=args.revision,
        salt=args.salt,
        calibration_count=args.calibration_count,
        confirmation_count=args.confirmation_count,
        cache_dir=args.cache_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(manifest["records"]), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
