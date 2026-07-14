import unittest

from research.simplex_policy.toy_gate import ToyGateConfig, run_toy_gate


class SimplexPolicyToyGateTest(unittest.TestCase):
    def test_small_gate_returns_complete_schema(self):
        config = ToyGateConfig(
            experiment_name="unit-simplex-gate",
            category_count=4,
            sample_count=2048,
            scenario_seeds=(1, 2),
            temperature=0.8,
            logit_scale=0.5,
            policy_drift_scale=0.2,
            reward_nonlinearity=0.2,
            clip_epsilon=0.2,
            gauge_offsets=(-1.0, 0.0, 1.0),
            thresholds={
                "max_concrete_reference_error": 1e-8,
                "max_simplex_gradient_relative_error": 10.0,
                "min_gradient_variance_reduction": 0.0,
                "min_log_ratio_variance_reduction": 0.0,
                "min_effective_sample_size_ratio": 0.0,
                "min_clip_decision_disagreement": 0.0,
                "min_auxiliary_gauge_sensitivity": 0.0,
            },
        )
        result = run_toy_gate(config)
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["experiment_name"], "unit-simplex-gate")
        self.assertIn(result["status"], {"pass", "fail"})
        self.assertEqual(len(result["scenarios"]), 2)
        self.assertIn("gradient_variance_reduction_geometric_mean", result["summary"])
        self.assertTrue(
            result["surrogate_diagnostic"]["positive_crossed_margin_misaligned"]
        )
        self.assertTrue(
            result["surrogate_diagnostic"]["negative_crossed_margin_misaligned"]
        )


if __name__ == "__main__":
    unittest.main()
