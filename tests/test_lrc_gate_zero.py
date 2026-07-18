import json
import tempfile
import unittest
from pathlib import Path

from research.latent_reasoning_contract_benchmark.deterministic_fixtures import (
    run_coconut_fixture,
    run_codi_fixture,
)
from research.latent_reasoning_contract_benchmark.gate_zero import run_gate_zero


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "research"
    / "latent_reasoning_contract_benchmark"
    / "configs"
    / "gate_zero_fixtures_v1.json"
)


class LrcGateZeroTest(unittest.TestCase):
    def test_coconut_executes_pinned_source_and_matches_independent_recurrence(self):
        result = run_coconut_fixture()
        self.assertTrue(result.source_equivalent)
        self.assertTrue(result.replay_deterministic)
        self.assertEqual(result.executed_latent_steps, 2)

    def test_codi_executes_pinned_source_and_matches_independent_recurrence(self):
        result = run_codi_fixture()
        self.assertTrue(result.source_equivalent)
        self.assertTrue(result.replay_deterministic)
        self.assertEqual(result.executed_latent_steps, 2)

    def test_gate_zero_covers_four_methods_and_passes_all_controls(self):
        report = run_gate_zero(ROOT, CONFIG)
        self.assertEqual(report["decision"], "PASS_GATE_ZERO")
        self.assertEqual(len(report["methods"]), 4)
        self.assertTrue(all(report["controls"].values()))
        self.assertFalse(report["gpu_used"])
        self.assertFalse(report["training_used"])

    def test_gate_zero_rejects_manifest_hash_drift(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["source_manifest_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest hash drift"):
                run_gate_zero(ROOT, path)


if __name__ == "__main__":
    unittest.main()
