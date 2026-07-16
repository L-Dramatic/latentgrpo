from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.coordinate_invariance.switch_c2_calibration_postmortem import (
    build_diagnostic,
    load_journal,
)


def test_attempt5_postmortem_replays_the_frozen_calibration_failure() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = (
        root
        / "research/coordinate_invariance/configs/switch_c2_scientific_gate_v1.json"
    )
    journal_path = (
        root
        / "artifacts/coordinate_invariance/journals/"
        "switch_c2_calibration_4b0c45b26fbd8138.jsonl"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    header, records = load_journal(journal_path)

    diagnostic = build_diagnostic(
        config,
        header,
        records,
        config_path=config_path,
        journal_path=journal_path,
    )

    assert diagnostic["status"] == "fail"
    assert diagnostic["decision"] == "no_go_for_frozen_switch_c2_v1"
    assert diagnostic["inputs"]["record_count"] == 16
    assert diagnostic["gain_selection"]["valid_gain_count"] == 0
    smallest_gain = diagnostic["gain_selection"]["candidates"][0]
    assert smallest_gain["methods"]["visible_prefix_32"][
        "invalid_prompt_count"
    ] == 1
    assert smallest_gain["methods"]["visible_prefix_32"]["invalid_prompts"][0][
        "selected_index"
    ] == 8
    assert diagnostic["probe_selection"]["all_v32_margins_nonpositive"]
    assert diagnostic["finite_difference"]["pass"] is False
    assert diagnostic["finite_difference"]["selected"][
        "median_relative_error"
    ] == pytest.approx(0.2618461652187859)
