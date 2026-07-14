import unittest

import torch

from research.topk_concrete.factorized_trust_gate import (
    FactorizedTrustGateConfig,
    _make_drift_direction,
    _run_scenario,
)


class FactorizedTrustGateTest(unittest.TestCase):
    def setUp(self):
        self.config = FactorizedTrustGateConfig.from_mapping(
            {
                "experiment_name": "unit-test",
                "vocabulary_size": 12,
                "top_k_values": [3],
                "sample_count": 2048,
                "calibration_sample_count": 1024,
                "scenario_seeds": [7],
                "temperature": 0.8,
                "gumbel_scale": 1.1,
                "logit_scales": [0.7],
                "drift_modes": ["random", "sharpen", "flatten"],
                "target_joint_kl": 0.02,
                "calibration_iterations": 14,
                "maximum_drift_scale": 4.0,
                "clip_epsilon": 0.2,
                "thresholds": {
                    "max_target_joint_kl_relative_error": 1.0,
                    "max_factorization_abs_error": 1e-10,
                    "max_kl_chain_abs_error": 1e-10,
                    "max_exact_ratio_mean_error": 1.0,
                    "min_component_kl_tolerance": -1.0,
                    "component_material_share_floor": 0.0,
                    "min_both_components_material_scenario_fraction": 0.0,
                    "hidden_violation_rate_floor": 0.0,
                    "min_hidden_violation_scenario_fraction": 0.0,
                    "min_hidden_violation_rate_mean": 0.0,
                    "min_opposite_sign_rate_mean": 0.0,
                    "min_support_share_interdecile_range": 0.0,
                },
            }
        )

    def test_drift_directions_are_centered_and_unit_rms(self):
        logits = torch.linspace(-1.0, 1.0, 12, dtype=torch.float64)
        for mode in self.config.drift_modes:
            direction = _make_drift_direction(
                logits,
                mode,
                generator=torch.Generator().manual_seed(81),
            )
            self.assertAlmostEqual(float(direction.mean()), 0.0, delta=1e-12)
            self.assertAlmostEqual(
                float(direction.square().mean().sqrt()), 1.0, delta=1e-12
            )

    def test_scenario_calibrates_and_preserves_factorization(self):
        scenario = _run_scenario(
            self.config,
            seed=7,
            scenario_seed=107,
            logit_scale=0.7,
            top_k=3,
            drift_mode="random",
        )

        self.assertTrue(scenario["calibrated"])
        self.assertLess(scenario["factorization_max_abs_error"], 1e-10)
        self.assertLess(scenario["kl_chain_abs_error"], 1e-10)
        self.assertGreater(scenario["measurement_joint_kl"], 0.0)


if __name__ == "__main__":
    unittest.main()
