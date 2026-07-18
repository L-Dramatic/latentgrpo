import unittest
from pathlib import Path

from research.branch_consistent_mixture_distillation.prompt_selection import (
    build_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "research"
    / "branch_consistent_mixture_distillation"
    / "configs"
    / "gate_minus_one_v1.json"
)


class BcmdPromptSelectionTest(unittest.TestCase):
    def test_selection_is_label_blind_exact_and_deterministic(self):
        first = build_manifest(CONFIG)
        second = build_manifest(CONFIG)
        self.assertEqual(first, second)
        self.assertFalse(first["labels_read"])
        self.assertEqual(first["record_count"], 64)
        self.assertEqual(first["calibration_count"], 32)
        self.assertEqual(first["confirmation_count"], 32)
        ids = [record["example_id"] for record in first["records"]]
        self.assertEqual(len(set(ids)), 64)
        self.assertEqual(
            sum(record["split"] == "calibration" for record in first["records"]),
            32,
        )


if __name__ == "__main__":
    unittest.main()
