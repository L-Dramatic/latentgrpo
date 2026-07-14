from __future__ import annotations

import json
from pathlib import Path

import torch

from research.simplex_policy.densities import concrete_score, sample_concrete
from research.policy_contract_audit.stage_b_common import (
    canonical_config_hash,
    select_rows,
)
from research.policy_contract_audit.stage_b_sequential import (
    SequentialConfig,
    distribution_metrics,
    evaluate_gates,
    sample_filtered_action,
)
from research.policy_contract_audit.stage_b_gradient import (
    GradientConfig,
    evaluate_candidate_updates,
    evaluate_gates as evaluate_gradient_gates,
    group_normalized_advantages,
)


def test_stage_b_configs_have_frozen_hashes() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = {
        "stage_b_sequential_v1.json": (
            "63ea4518ae2ac5e2ff52eecd385eda68cba125f950612205a8eee802c512a25f"
        ),
        "stage_b_gradient_v1.json": (
            "1605fe744bdca5dad7d4dc1a99b295f80a682c11841bf2c3010ea3999d218f3f"
        ),
    }
    for name, digest in expected.items():
        raw = json.loads(
            (root / "research" / "policy_contract_audit" / "configs" / name)
            .read_text(encoding="utf-8")
        )
        assert canonical_config_hash(raw) == digest


def test_hash_selection_is_order_independent() -> None:
    rows = [
        {"unique_id": "b", "value": 2},
        {"unique_id": "a", "value": 1},
        {"unique_id": "c", "value": 3},
    ]
    selection = {"count": 2, "key": "unique_id", "salt": "fixed"}
    first = select_rows(rows, selection)
    second = select_rows(list(rows), selection)
    assert [(row.dataset_index, row.selection_hash) for row in first] == [
        (row.dataset_index, row.selection_hash) for row in second
    ]


def test_distribution_identity_control() -> None:
    logits = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    metrics = distribution_metrics(logits, logits.clone())
    assert metrics["js"] == 0.0
    assert metrics["total_variation"] == 0.0
    assert metrics["top1_disagreement"] == 0.0


def test_singleton_filtered_action_is_deterministic() -> None:
    action = sample_filtered_action(
        torch.tensor([3.0], dtype=torch.float64),
        temperature=0.3,
        generator=torch.Generator().manual_seed(1),
    )
    torch.testing.assert_close(action, torch.ones_like(action))


def test_sequential_gate_requires_controls_and_three_effects() -> None:
    summary = {
        "action_sum_error_p999": 0.0,
        "identity_branch_js_max": 0.0,
        "finite_trajectory_rate": 1.0,
        "completed_trajectory_rate": 1.0,
        "final_js_median": 0.02,
        "final_total_variation_median": 0.2,
        "final_top1_disagreement_mean": 0.3,
        "final_support_jaccard_median": 0.95,
        "reference_nll_absolute_gap_median": 0.0,
    }
    gates = {
        "probability_sum_error_p999_max": 1e-5,
        "identity_branch_js_max": 1e-7,
        "finite_trajectory_rate_min": 1.0,
        "completed_trajectory_rate_min": 1.0,
        "final_js_median_min": 0.01,
        "final_total_variation_median_min": 0.1,
        "final_top1_disagreement_mean_min": 0.2,
        "final_support_jaccard_median_max": 0.8,
        "reference_nll_absolute_gap_median_min": 0.02,
        "effect_required_count": 3,
    }
    result = evaluate_gates(summary, gates)
    assert result["effect_pass_count"] == 3
    assert result["stage_b1_pass"]
    summary["identity_branch_js_max"] = 1e-4
    assert not evaluate_gates(summary, gates)["stage_b1_pass"]


def test_sequential_config_validates_final_horizon() -> None:
    root = Path(__file__).resolve().parents[1]
    raw = json.loads(
        (
            root
            / "research"
            / "policy_contract_audit"
            / "configs"
            / "stage_b_sequential_v1.json"
        ).read_text(encoding="utf-8")
    )
    config = SequentialConfig.from_mapping(raw)
    assert config.config_hash == (
        "63ea4518ae2ac5e2ff52eecd385eda68cba125f950612205a8eee802c512a25f"
    )


def test_group_advantages_match_sample_standard_deviation() -> None:
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    advantages = group_normalized_advantages(
        rewards, group_size=4, epsilon=0.0
    )
    assert abs(float(advantages.mean())) < 1e-12
    torch.testing.assert_close(
        advantages.std(unbiased=True), torch.tensor(1.0, dtype=torch.float64)
    )


def test_candidate_update_with_exact_direction_improves_linear_reward() -> None:
    logits = torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64)
    generator = torch.Generator().manual_seed(8)
    samples, _ = sample_concrete(
        logits,
        temperature=0.7,
        sample_shape=(4096,),
        generator=generator,
    )
    rewards = samples[:, 0]
    exact_score = concrete_score(
        samples,
        logits.expand_as(samples),
        temperature=0.7,
    )
    exact_gradient = ((rewards - rewards.mean()).unsqueeze(1) * exact_score).mean(0)
    updates = evaluate_candidate_updates(
        evaluation_samples=samples,
        evaluation_rewards=rewards,
        active_logits=logits,
        temperature=0.7,
        exact_gradient=exact_gradient,
        surrogate_gradient=exact_gradient,
        steps=[0.01],
    )
    assert updates["0.01"]["exact"]["gain"] > 0


def test_gradient_gate_requires_superiority_not_sign_difference_alone() -> None:
    summary = {
        "action_sum_error_p999": 0.0,
        "finite_reward_rate": 1.0,
        "exact_gradient_split_half_cosine_median": 0.5,
        "candidate_ratio_z_p99": 1.0,
        "candidate_ess_fraction_p10": 0.9,
        "gradient_cosine_median": 0.5,
        "gradient_relative_error_median": 0.8,
        "candidate_gain_sign_disagreement_mean": 0.5,
        "exact_minus_surrogate_gain_mean": 0.0,
        "exact_minus_surrogate_gain_bootstrap_l95": -0.1,
        "exact_positive_rate_advantage": 0.0,
    }
    gates = {
        "action_sum_error_p999_max": 1e-5,
        "finite_reward_rate_min": 1.0,
        "exact_gradient_split_half_cosine_median_min": 0.25,
        "candidate_ratio_z_p99_max": 6.0,
        "candidate_ess_fraction_p10_min": 0.5,
        "gradient_cosine_median_max": 0.8,
        "gradient_relative_error_median_min": 0.25,
        "candidate_gain_sign_disagreement_mean_min": 0.1,
        "exact_minus_surrogate_gain_mean_min": 0.0005,
        "exact_minus_surrogate_gain_bootstrap_l95_min": 0.0,
        "exact_positive_rate_advantage_min": 0.1,
    }
    result = evaluate_gradient_gates(summary, gates)
    assert result["operational_effects"]["gain_sign_disagreement"]
    assert not result["authorize_matched_training"]
    summary["exact_positive_rate_advantage"] = 0.2
    assert evaluate_gradient_gates(summary, gates)["authorize_matched_training"]


def test_gradient_config_hash_and_validation() -> None:
    root = Path(__file__).resolve().parents[1]
    raw = json.loads(
        (
            root
            / "research"
            / "policy_contract_audit"
            / "configs"
            / "stage_b_gradient_v1.json"
        ).read_text(encoding="utf-8")
    )
    config = GradientConfig.from_mapping(raw)
    assert config.config_hash == (
        "1605fe744bdca5dad7d4dc1a99b295f80a682c11841bf2c3010ea3999d218f3f"
    )
