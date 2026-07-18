import copy
import json
import unittest
from pathlib import Path

from research.latent_reasoning_contract_benchmark.deterministic_confirmation import (
    map_confirmation_report,
    validate_confirmation_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "research" / "latent_reasoning_contract_benchmark" / "configs"


class LrcDeterministicConfirmationTest(unittest.TestCase):
    def test_confirmation_config_validates_and_reuses_a0_contract(self):
        calibration = json.loads((CONFIG_DIR / "deterministic_intervention_a0_v1.json").read_text(encoding="utf-8"))
        confirmation = json.loads((CONFIG_DIR / "deterministic_intervention_confirmation_v1.json").read_text(encoding="utf-8"))
        validate_confirmation_config(ROOT, confirmation)
        for key in (
            "methods",
            "conditions",
            "equal_depth_effect_conditions",
            "random_seed",
            "bootstrap_seed",
            "bootstrap_replicates",
            "controls",
            "calibration_signal_gates",
        ):
            self.assertEqual(confirmation[key], calibration[key])

    def test_confirmation_cannot_switch_back_to_calibration_rows(self):
        confirmation = json.loads((CONFIG_DIR / "deterministic_intervention_confirmation_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(confirmation["prompt_manifest"]["split"], "confirmation")
        self.assertIn("calibration_evidence", confirmation)

    def test_confirmation_mapping_preserves_frozen_rule(self):
        mapped = map_confirmation_report({"decision": "PASS_A0_SIGNAL", "methods": {}})
        self.assertEqual(mapped["decision"], "PASS_CONFIRMATION")
        self.assertEqual(mapped["base_a0_rule_decision"], "PASS_A0_SIGNAL")

    def test_confirmation_rejects_threshold_change(self):
        confirmation = json.loads((CONFIG_DIR / "deterministic_intervention_confirmation_v1.json").read_text(encoding="utf-8"))
        confirmation["calibration_signal_gates"]["minimum_absolute_mean_nll_delta"] = 0.001
        with self.assertRaisesRegex(ValueError, "changed frozen A0 field"):
            validate_confirmation_config(ROOT, confirmation)


if __name__ == "__main__":
    unittest.main()
