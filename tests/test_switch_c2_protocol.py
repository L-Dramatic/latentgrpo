from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from research.coordinate_invariance import (
    switch_c2_eligibility_scan,
    switch_c2_scientific_gate,
    switch_checkpoint_identity_smoke,
)
from research.coordinate_invariance.real_models.switch import (
    SwitchReplayPlan,
    load_pinned_switch_types,
)
from research.coordinate_invariance.switch_c2_eligibility_scan import (
    implementation_hashes as eligibility_implementation_hashes,
    require_current_identity_artifact,
)
from research.coordinate_invariance.switch_c2_scientific_gate import (
    _analyze_test,
    _measure_prompt,
    _select_calibration,
    implementation_hashes as scientific_implementation_hashes,
    require_bound_pass_artifact,
)
from research.coordinate_invariance.switch_checkpoint_identity_smoke import (
    implementation_hashes as identity_implementation_hashes,
)


def _probe(scale: float, simple: float, v32: float, oracle: float) -> dict:
    return {
        "relative_hidden_l2": scale,
        "metrics": {
            "visible_prefix_1": {
                "strict_spearman": simple,
                "strict_top_risk_recall": 0.5,
            },
            "visible_prefix_32": {
                "strict_spearman": v32,
                "strict_top_risk_recall": 0.8,
            },
            "visible_prefix_64": {
                "strict_spearman": oracle,
                "strict_top_risk_recall": 0.85,
            },
        },
    }


def _update(gain: float, candidate_kl: float, baseline_kl: float) -> dict:
    return {
        "predicted_objective_gain": gain,
        "methods": {
            "visible_prefix_1": {
                "valid_radius": True,
                "strict_kl": baseline_kl,
                "objective_utility_gain": 0.01,
                "strict_utility_gain": 0.0,
                "prefix8_kl": 0.002,
            },
            "visible_prefix_32": {
                "valid_radius": True,
                "strict_kl": candidate_kl,
                "objective_utility_gain": 0.011,
                "strict_utility_gain": 0.1,
                "prefix8_kl": 0.001,
            },
        },
    }


def test_c2_artifacts_bind_all_behavior_defining_implementations() -> None:
    assert set(identity_implementation_hashes()) == {
        "identity_runner",
        "switch_adapter",
    }
    assert set(eligibility_implementation_hashes()) == {
        "eligibility_runner",
        "switch_adapter",
    }
    assert set(scientific_implementation_hashes()) == {
        "scientific_runner",
        "geometry",
        "switch_adapter",
        "fctr_solver",
        "charts",
        "metrics",
        "statistics",
    }


@pytest.mark.parametrize(
    "module",
    (
        switch_checkpoint_identity_smoke,
        switch_c2_eligibility_scan,
        switch_c2_scientific_gate,
    ),
)
def test_cuda_allocator_is_initialized_before_peak_stats_reset(
    monkeypatch: pytest.MonkeyPatch,
    module: object,
) -> None:
    calls: list[object] = []
    device = torch.device("cuda:0")
    monkeypatch.setattr(torch.cuda, "init", lambda: calls.append("init"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda selected: calls.append(("reset", selected)),
    )

    module._reset_cuda_peak_memory_stats(device)

    assert calls == ["init", "empty_cache", ("reset", device)]


def test_c2_source_manifest_freezes_current_implementation_hashes() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "research/coordinate_invariance/SOURCE_MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )
    switch = next(
        item
        for item in manifest["architectures"]
        if item["name"] == "SWITCH Qwen3-8B Phase-3 GRPO"
    )
    frozen = switch["c2_contract"]["implementation_sha256"]
    assert frozen["identity"] == identity_implementation_hashes()
    assert frozen["eligibility"] == eligibility_implementation_hashes()
    assert frozen["scientific"] == scientific_implementation_hashes()


def test_c2_rejects_stale_upstream_artifacts() -> None:
    identity = {
        "status": "pass",
        "config_sha256": "identity-config",
        "implementation_sha256": identity_implementation_hashes(),
    }
    require_current_identity_artifact(identity, "identity-config")
    identity["implementation_sha256"] = {}
    with pytest.raises(ValueError, match="different implementation"):
        require_current_identity_artifact(identity, "identity-config")

    eligibility = {
        "status": "pass",
        "config_sha256": "c2-config",
        "implementation_sha256": eligibility_implementation_hashes(),
    }
    require_bound_pass_artifact(
        eligibility,
        name="eligibility",
        expected_config_sha256="c2-config",
        expected_implementation=eligibility_implementation_hashes(),
    )
    eligibility["implementation_sha256"] = {}
    with pytest.raises(ValueError, match="different implementation"):
        require_bound_pass_artifact(
            eligibility,
            name="eligibility",
            expected_config_sha256="c2-config",
            expected_implementation=eligibility_implementation_hashes(),
        )


def test_calibration_selection_freezes_scale_simple_baseline_and_gain() -> None:
    config = {
        "probe_bank": {"relative_hidden_l2_grid": [0.01, 0.02]},
        "updates": {
            "simple_baseline_selection_order": ["visible_prefix_1"],
            "selection_tie_tolerance": 1e-12,
            "minimum_calibration_objective_gain_ratio": 0.9,
            "predicted_objective_gain_grid": [0.001, 0.002],
        },
        "derivatives": {"finite_difference_relative_step_grid": [0.01, 0.02]},
    }
    records = []
    for index in range(4):
        records.append(
            {
                "probes": [
                    _probe(0.01, 0.5, 0.55, 0.6),
                    _probe(0.02, 0.45, 0.8, 0.82),
                ],
                "updates": [
                    _update(0.001, 0.4, 1.0),
                    _update(0.002, 0.8, 1.0),
                ],
                "finite_difference": [
                    {"relative_hidden_l2": 0.01, "relative_error": 0.05 + index * 0.001},
                    {"relative_hidden_l2": 0.02, "relative_error": 0.15},
                ],
            }
        )
    selection = _select_calibration(config, records)
    assert selection["selected_probe_relative_hidden_l2"] == 0.02
    assert selection["selected_best_simple_baseline"] == "visible_prefix_1"
    assert selection["selected_predicted_objective_gain"] == 0.001
    assert selection["selected_exact_v8_kl_budget"] == 0.001
    assert (
        selection["finite_difference_selection"]["relative_hidden_l2"] == 0.01
    )


def test_heldout_analysis_applies_every_scientific_gate() -> None:
    config = {
        "statistics": {
            "bootstrap_samples": 1000,
            "confidence": 0.95,
            "seed": 71556,
        },
        "derivatives": {"maximum_zero_point_logit_absolute_error": 0.0},
        "subspace": {"maximum_basis_orthogonality_error": 1e-5},
        "decision_rules": {
            "maximum_projected_gradient_relative_error": 0.01,
            "maximum_metric_update_transport_relative_error": 0.005,
            "maximum_orthogonal_euclidean_direction_discrepancy": 0.005,
            "chart_affected_min_strict_symmetric_kl": 0.0001,
            "minimum_median_condition12_euclidean_direction_discrepancy": 0.25,
            "minimum_chart_affected_prompt_fraction": 0.25,
            "minimum_v32_spearman_margin_over_best_simple": 0.1,
            "minimum_v32_spearman_margin_ci_low": 0.03,
            "minimum_v32_top_risk_recall_margin": 0.1,
            "maximum_v32_to_best_simple_strict_holdout_kl_ratio": 0.9,
            "maximum_v32_to_best_simple_strict_holdout_kl_ratio_ci_high": 1.0,
            "minimum_v32_minus_best_simple_strict_holdout_utility_ci_low": 0.0,
            "maximum_v32_spearman_gap_to_v64_oracle": 0.05,
            "minimum_exact_kl_retuning_success_fraction": 1.0,
        },
    }
    calibration = {
        "selections": {
            "selected_probe_relative_hidden_l2": 0.02,
            "selected_predicted_objective_gain": 0.001,
            "selected_best_simple_baseline": "visible_prefix_1",
        }
    }
    records = []
    for index in range(8):
        records.append(
            {
                "probes": [_probe(0.02, 0.5, 0.8, 0.82)],
                "updates": [_update(0.001, 0.5, 1.0)],
                "transport_controls": {
                    "condition12_euclidean_direction_discrepancy": 0.5,
                    "maximum_metric_update_transport_relative_error": 1e-8,
                    "orthogonal_euclidean_direction_discrepancy": 1e-8,
                },
                "zero_point_logit_max_abs_error": 0.0,
                "basis_orthogonality_max_abs_error": 1e-12,
                "projected_gradient_relative_error": 1e-5,
                "exact_kl_retuned_coordinate_updates": {
                    "identity_chart": {"success": True},
                    "condition12_chart": {"success": True},
                    "strict_pair_symmetric_kl": 0.001,
                },
                "free_rollouts": {
                    "coordinate_euclidean_identity": {
                        "post_block_ids": [index, 1]
                    },
                    "coordinate_euclidean_condition_12": {
                        "post_block_ids": [index, 2]
                    },
                },
            }
        )
    summary, gates = _analyze_test(config, records, calibration)
    assert summary["v32_spearman_margin"]["value"] == pytest.approx(0.3)
    assert summary["v32_to_best_simple_strict_kl_ratio"]["value"] == pytest.approx(0.5)
    assert summary["chart_affected_prompt_fraction"] == 1.0
    assert all(gates.values())


class _TinyDifferentiableLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(71580)
        self.embedding = nn.Embedding(20, 8)
        self.projection = nn.Parameter(
            torch.randn(8, 20, generator=generator, dtype=torch.float32) * 0.1
        )
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.randn(20, 8, generator=generator, dtype=torch.float32)
            )
        self.config = SimpleNamespace(model_type="tiny-switch-c2")

    def get_input_embeddings(self):
        return self.embedding

    def forward(
        self,
        *,
        inputs_embeds,
        attention_mask,
        position_ids,
        past_key_values=None,
        use_cache=True,
        output_hidden_states=False,
    ):
        del attention_mask, use_cache
        positions = position_ids.to(inputs_embeds.dtype).unsqueeze(-1)
        context = 0.0
        if (
            isinstance(past_key_values, tuple)
            and past_key_values
            and isinstance(past_key_values[0], torch.Tensor)
        ):
            context = 0.1 * past_key_values[0]
        hidden = torch.tanh(inputs_embeds + context + 0.001 * positions)
        logits = hidden @ self.projection
        return SimpleNamespace(
            logits=logits,
            hidden_states=(hidden,) if output_hidden_states else None,
            past_key_values=(hidden[:, -1:, :],),
        )


class _TinyTokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return " ".join(str(value) for value in token_ids)


def test_prompt_measurement_runs_gradient_jvp_probes_and_updates_end_to_end() -> None:
    config = json.loads(
        Path("research/coordinate_invariance/configs/switch_c2_scientific_gate_v1.json").read_text(
            encoding="utf-8"
        )
    )
    config["probe_bank"]["paired_native_directions"] = 1
    config["probe_bank"]["relative_hidden_l2_grid"] = [0.0025]
    config["updates"]["predicted_objective_gain_grid"] = [1e-5]
    config["updates"]["maximum_relative_hidden_l2"] = 10.0
    switch_type, config_type = load_pinned_switch_types("_external/switch")
    tiny = _TinyDifferentiableLM()
    for parameter in tiny.parameters():
        parameter.requires_grad_(False)
    token_config = config_type(
        swi_start_id=3,
        swi_end_id=4,
        latent_id=6,
        eos_token_id=5,
    )
    model = switch_type(tiny, token_config)
    model.eval()
    bundle = SimpleNamespace(
        model=model,
        tokenizer=_TinyTokenizer(),
        token_config=token_config,
        device=torch.device("cpu"),
    )
    plan = SwitchReplayPlan(
        prompt_input_ids=torch.tensor([1, 2]),
        prompt_attention_mask=torch.tensor([1, 1]),
        visible_prefix_ids=torch.tensor([9]),
        latent_steps=4,
        visible_target_ids=torch.tensor([7 + index % 10 for index in range(64)]),
        visible_decision_start_index=2,
    )
    selected = {
        "scan_rank": 0,
        "dataset_index": 0,
        "unique_id": "tiny/0",
        "subject": "Algebra",
        "level": 1,
        "selected_split": "calibration",
        "selected_index": 0,
        "replay_plan": {
            "prompt_input_ids": plan.prompt_input_ids.tolist(),
            "prompt_attention_mask": plan.prompt_attention_mask.tolist(),
            "visible_prefix_ids": plan.visible_prefix_ids.tolist(),
            "latent_steps": plan.latent_steps,
            "visible_target_ids": plan.visible_target_ids.tolist(),
            "visible_decision_start_index": plan.visible_decision_start_index,
        },
    }
    record = _measure_prompt(
        config,
        bundle,
        selected,
        torch.ones(8, dtype=torch.float64),
        probe_scales=[0.0025],
        gains=[1e-5],
        run_finite_difference=False,
        exact_kl_budget=None,
        free_method_names=None,
    )
    assert record["zero_point_logit_max_abs_error"] == 0.0
    assert record["basis_orthogonality_max_abs_error"] < 1e-10
    assert record["runtime"]["jvp_calls"] == 4
    assert record["runtime"]["full_gradient_backward_calls"] == 1
    assert len(record["probes"][0]["actual_strict_kl"]) == 2
    assert "visible_prefix_32" in record["updates"][0]["methods"]
