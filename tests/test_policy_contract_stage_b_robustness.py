from __future__ import annotations

import pytest

from research.policy_contract_audit.stage_b_robustness import (
    analyze,
    build_complete_panel,
)


def _records() -> list[dict[str, float | int]]:
    rows = []
    for state, gains in [(10, (0.1, 0.2)), (20, (-0.1, 0.4))]:
        for temperature, gain in zip((0.3, 0.5), gains):
            rows.append(
                {
                    "dataset_index": state,
                    "temperature": temperature,
                    "gate_gain_difference": gain,
                    "gate_exact_positive": float(gain > -0.05),
                    "gate_surrogate_positive": float(gain > 0.25),
                    "gate_gain_sign_disagreement": float(abs(gain) >= 0.2),
                }
            )
    return rows


def _config() -> dict:
    return {
        "selection": {"count": 2},
        "sampler": {"temperatures": [0.3, 0.5]},
        "gates": {
            "exact_minus_surrogate_gain_mean_min": 0.05,
            "exact_positive_rate_advantage_min": 0.10,
        },
    }


def test_complete_panel_rejects_missing_temperature() -> None:
    with pytest.raises(ValueError, match="missing temperatures"):
        build_complete_panel(
            _records()[:-1],
            expected_states=2,
            expected_temperatures=[0.3, 0.5],
        )


def test_analysis_is_deterministic_and_prompt_clustered() -> None:
    first = analyze(records=_records(), config=_config(), replicates=200, seed=9)
    second = analyze(records=_records(), config=_config(), replicates=200, seed=9)
    assert first == second
    assert first["status"] == "post_hoc_non_gating_robustness"
    assert first["record_count"] == 4
    assert first["state_count"] == 2
    assert first["point_estimates"]["gain_difference_mean"] == pytest.approx(0.15)
    assert set(
        first["prompt_cluster_bootstrap"]["gain_difference_by_temperature"]
    ) == {"0.3", "0.5"}
