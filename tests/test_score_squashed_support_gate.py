import unittest

from research.score_squashed_gumbel.support_concentration_gate import (
    SupportConcentrationGateConfig,
    _run_scenario,
)


class ScoreSquashedSupportGateTest(unittest.TestCase):
    def setUp(self):
        self.config = SupportConcentrationGateConfig.from_mapping(
            {
                "experiment_name": "unit-test",
                "vocabulary_size": 16,
                "top_k": 4,
                "sample_count": 2048,
                "score_sample_count": 1024,
                "scenario_seeds": [17],
                "logit_scales": [1.0],
                "temperature": 1.0,
                "gumbel_scale": 1.0,
                "lower_bound": -1.5,
                "upper_bound": 3.0,
                "policy_drift_scale": 0.1,
                "saturation_derivative_threshold": 0.05,
                "thresholds": {
                    "max_squashed_unbounded_support_mismatch_rate": 0.0,
                    "max_exact_ratio_mean_error": 1.0,
                    "max_exact_score_mean_norm": 1.0,
                    "min_hardclip_unbounded_support_change_rate": 0.0,
                    "min_hardclip_selected_upper_atom_fraction": 0.0,
                    "max_selected_squash_saturation_fraction": 1.0,
                    "max_hardclip_squashed_mixture_entropy_gap": 10.0,
                    "max_hardclip_squashed_qmax_p95_gap": 1.0,
                    "min_support_inclusion_entropy_gain_mean": -10.0,
                    "min_support_inclusion_entropy_gain_positive_fraction": 0.0,
                    "max_theoretical_qmax_bound_violation": 1e-12,
                },
            }
        )

    def test_scenario_preserves_support_and_has_valid_ratio(self):
        scenario = _run_scenario(
            self.config,
            seed=17,
            scenario_seed=117,
            logit_scale=1.0,
        )

        self.assertEqual(
            scenario["squashed_unbounded_support_mismatch_rate"], 0.0
        )
        self.assertLess(scenario["exact_ratio_mean_error"], 0.1)
        self.assertLessEqual(scenario["theoretical_qmax_bound_violation"], 1e-12)


if __name__ == "__main__":
    unittest.main()
