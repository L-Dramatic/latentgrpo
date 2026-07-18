import tempfile
import unittest
from pathlib import Path

from research.latent_reasoning_contract_benchmark.verify_deterministic_confirmation import (
    verify_confirmation,
)


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts" / "latent_reasoning_contract_benchmark"
CONFIG = ROOT / "research" / "latent_reasoning_contract_benchmark" / "configs" / "deterministic_intervention_confirmation_v1.json"
RECORDS = BASE / "deterministic_intervention_confirmation_v1_records.jsonl"
SUMMARY = BASE / "deterministic_intervention_confirmation_v1.json"


class LrcVerifyDeterministicConfirmationTest(unittest.TestCase):
    def test_completed_confirmation_passes(self):
        report = verify_confirmation(ROOT, CONFIG, RECORDS, SUMMARY)
        self.assertEqual(report["decision"], "PASS_CONFIRMATION_AUDIT")
        self.assertTrue(all(report["controls"].values()))

    def test_calibration_record_cannot_replace_confirmation_record(self):
        lines = RECORDS.read_text(encoding="utf-8").splitlines()
        calibration_line = (BASE / "deterministic_intervention_a0_v1_records.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()[0]
        with tempfile.TemporaryDirectory() as directory:
            corrupted = Path(directory) / "records.jsonl"
            corrupted.write_text("\n".join([calibration_line, *lines[1:]]) + "\n", encoding="utf-8")
            report = verify_confirmation(ROOT, CONFIG, corrupted, SUMMARY)
        self.assertEqual(report["decision"], "HOLD_CONFIRMATION_AUDIT")
        self.assertFalse(report["controls"]["record_provenance"])


if __name__ == "__main__":
    unittest.main()
