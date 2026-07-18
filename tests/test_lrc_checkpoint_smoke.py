import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research.latent_reasoning_contract_benchmark.checkpoint_smoke import (
    run_checkpoint_smoke,
    validate_config,
    validate_linked_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "research"
    / "latent_reasoning_contract_benchmark"
    / "configs"
    / "checkpoint_smoke_v1.json"
)


class LrcCheckpointSmokeTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_frozen_config_and_linked_evidence_pass(self):
        validate_config(ROOT, self.config)
        linked = validate_linked_evidence(ROOT, self.config)
        self.assertEqual(len(linked), 2)
        self.assertTrue(all(entry["pass"] for entry in linked))

    def test_config_cannot_drop_a_method(self):
        mutated = copy.deepcopy(self.config)
        mutated["linked_evidence"][0]["methods"] = []
        with self.assertRaisesRegex(ValueError, "exactly four methods"):
            validate_config(ROOT, mutated)

    def test_config_cannot_relax_repeatability(self):
        mutated = copy.deepcopy(self.config)
        mutated["gates"]["require_exact_repeatability"] = False
        with self.assertRaisesRegex(ValueError, "mandatory gates"):
            validate_config(ROOT, mutated)

    def test_linked_evidence_hash_drift_fails_closed(self):
        mutated = copy.deepcopy(self.config)
        mutated["linked_evidence"][0]["sha256"] = "0" * 64
        linked = validate_linked_evidence(ROOT, mutated)
        self.assertFalse(linked[0]["pass"])
        self.assertIn("artifact hash mismatch", linked[0]["failures"])

    def test_runtime_exception_produces_hold_artifact(self):
        with patch(
            "research.latent_reasoning_contract_benchmark.checkpoint_smoke.run_codi_smoke",
            side_effect=RuntimeError("synthetic native-load failure"),
        ):
            report = run_checkpoint_smoke(ROOT, CONFIG)
        self.assertEqual(report["decision"], "HOLD_CHECKPOINT_SMOKE")
        self.assertEqual(report["codi"]["status"], "hold")
        self.assertIn("RuntimeError", report["codi"]["failures"][0])
        self.assertFalse(report["scientific_evidence"])


if __name__ == "__main__":
    unittest.main()
