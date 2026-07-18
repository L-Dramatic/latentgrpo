import json
import tempfile
import unittest
from pathlib import Path

from research.latent_reasoning_contract_benchmark.verify_deterministic_a0 import verify


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "research" / "latent_reasoning_contract_benchmark" / "configs" / "deterministic_intervention_a0_v1.json"
RECORDS = ROOT / "artifacts" / "latent_reasoning_contract_benchmark" / "deterministic_intervention_a0_v1_records.jsonl"
SUMMARY = ROOT / "artifacts" / "latent_reasoning_contract_benchmark" / "deterministic_intervention_a0_v1.json"


class LrcVerifyDeterministicA0Test(unittest.TestCase):
    def test_completed_artifacts_pass(self):
        report = verify(ROOT, CONFIG, RECORDS, SUMMARY)
        self.assertEqual(report["decision"], "PASS_A0_AUDIT")
        self.assertTrue(all(report["controls"].values()))

    def test_missing_record_fails_coverage_and_recomputation(self):
        lines = RECORDS.read_text(encoding="utf-8").splitlines()
        with tempfile.TemporaryDirectory() as directory:
            truncated = Path(directory) / "records.jsonl"
            truncated.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            report = verify(ROOT, CONFIG, truncated, SUMMARY)
        self.assertEqual(report["decision"], "HOLD_A0_AUDIT")
        self.assertFalse(report["controls"]["exact_record_key_coverage"])
        self.assertFalse(report["controls"]["summary_recomputation"])


if __name__ == "__main__":
    unittest.main()
