import hashlib

import pytest

from research.coordinate_invariance.switch_prompt_order import build_prompt_order


def _config(dataset_sha256: str):
    return {
        "experiment_name": "test",
        "dataset": {
            "id": "test/math",
            "revision": "abc",
            "file_sha256": dataset_sha256,
            "expected_rows": 3,
        },
        "ordering": {"seed": 7, "key": "test"},
        "eligibility": {"target_block": 0},
        "splits": {"calibration_eligible": 1, "test_eligible": 1},
    }


def _rows():
    return [
        {
            "problem": f"problem {index}",
            "solution": "solution",
            "answer": str(index),
            "subject": "Algebra",
            "level": index + 1,
            "unique_id": f"id-{index}",
        }
        for index in range(3)
    ]


def test_order_is_deterministic_and_model_independent() -> None:
    digest = hashlib.sha256(b"dataset").hexdigest()
    first = build_prompt_order(_config(digest), _rows(), dataset_sha256=digest)
    second = build_prompt_order(_config(digest), _rows(), dataset_sha256=digest)
    assert first == second
    ordered = first["ordered_candidates"]
    assert [row["scan_rank"] for row in ordered] == [0, 1, 2]
    assert [row["ordering_key"] for row in ordered] == sorted(
        row["ordering_key"] for row in ordered
    )
    assert all("answer" not in row and "solution" not in row for row in ordered)


def test_dataset_identity_is_mandatory() -> None:
    digest = hashlib.sha256(b"dataset").hexdigest()
    with pytest.raises(ValueError, match="SHA-256"):
        build_prompt_order(_config(digest), _rows(), dataset_sha256="0" * 64)
