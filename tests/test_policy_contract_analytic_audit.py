import unittest

from research.policy_contract_audit.analytic_audit import AnalyticAuditConfig
from research.policy_contract_audit.source_faithful_audit import (
    SourceFaithfulAuditConfig,
)
from research.policy_contract_audit.public_checkpoint_audit import (
    evaluate_gates,
    summarize_records,
)


class PolicyContractAnalyticAuditTest(unittest.TestCase):
    def test_config_rejects_topk_equal_to_vocabulary(self):
        raw = {
            "experiment_name": "test",
            "sample_count": 1000,
            "scenario_seeds": [1],
            "vocabulary_size": 4,
            "top_k": 4,
            "temperatures": [0.5],
            "logit_scales": [1.0],
            "policy_drift_scale": 0.05,
            "clip_epsilon": 0.2,
            "thresholds": {},
        }
        with self.assertRaises(ValueError):
            AnalyticAuditConfig.from_mapping(raw)

    def test_source_faithful_config_rejects_invalid_top_p(self):
        raw = {
            "experiment_name": "test",
            "sample_count": 1000,
            "scenario_seeds": [1],
            "vocabulary_size": 4,
            "top_k": 3,
            "top_p": 0.0,
            "temperatures": [0.5],
            "logit_scales": [1.0],
            "policy_drift_scale": 0.05,
            "thresholds": {},
        }
        with self.assertRaises(ValueError):
            SourceFaithfulAuditConfig.from_mapping(raw)

    def test_public_checkpoint_gate_requires_controls_and_four_effects(self):
        base = {
            "state_index": 0,
            "archive_mass_error_max": 0.0,
            "reconstruction_l1_max": 0.0,
            "active_support_size": 30,
            "exact_score_snr": 1.0,
            "exact_ratio_z_from_one": 1.0,
            "excluded_full_model_mass": 0.2,
            "proxy_mode_disagreement": 0.5,
            "surrogate_exact_score_cosine_median": 0.3,
            "surrogate_exact_score_relative_error_median": 0.8,
            "filtered_support_churn": 0.0,
            "mixture_proxy_embedding_relative_l2_median": 0.5,
        }
        summary = summarize_records([base])
        gates = {
            "archive_mass_error_p999_max": 1e-5,
            "reconstruction_l1_p999_max": 1e-5,
            "active_support_size_max": 30,
            "exact_score_snr_p99_max": 6.0,
            "exact_ratio_z_p99_max": 6.0,
            "excluded_full_model_mass_median_min": 0.01,
            "proxy_mode_disagreement_mean_min": 0.1,
            "surrogate_exact_score_cosine_median_max": 0.8,
            "surrogate_exact_score_relative_error_median_min": 0.25,
            "filtered_support_churn_mean_min": 0.01,
        }
        result = evaluate_gates(summary, gates)
        self.assertEqual(result["effect_pass_count"], 4)
        self.assertTrue(result["proceed_to_stage_b"])


if __name__ == "__main__":
    unittest.main()
