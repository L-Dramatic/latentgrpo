import unittest
from pathlib import Path

from research.latent_reasoning_contract_benchmark.cross_family_thesis_audit import (
    audit_current_thesis,
)


ROOT = Path(__file__).resolve().parents[1]


class LrcCrossFamilyThesisAuditTest(unittest.TestCase):
    def test_frozen_evidence_recomputes_and_triggers_kill(self):
        report = audit_current_thesis(ROOT)
        self.assertEqual(report["decision"], "KILL_LRC_CURRENT_THESIS")
        self.assertTrue(report["integrity_pass"])
        self.assertTrue(all(report["controls"].values()))
        self.assertTrue(report["permanent_kill_condition"]["triggered"])

    def test_cross_family_contrast_is_not_hidden(self):
        report = audit_current_thesis(ROOT)
        evidence = report["stochastic_evidence"]
        self.assertEqual(evidence["latent_grpo"]["a0_decision"], "ADVANCE_A1")
        self.assertEqual(evidence["soft_grpo"]["a0_decision"], "KILL_A0")
        self.assertTrue(evidence["family_specific"])
        self.assertTrue(evidence["soft_effectively_discrete"])
        self.assertGreater(
            evidence["latent_grpo"]["median_effective_support"],
            evidence["soft_grpo"]["median_effective_support"],
        )


if __name__ == "__main__":
    unittest.main()
