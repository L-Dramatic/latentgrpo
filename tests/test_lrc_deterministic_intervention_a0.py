import copy
import json
import unittest
from pathlib import Path

import torch

from research.latent_reasoning_contract_benchmark.deterministic_intervention_a0 import (
    _mean_kl,
    _norm_matched_random,
    _target_scores,
    summarize_records,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "research"
    / "latent_reasoning_contract_benchmark"
    / "configs"
    / "deterministic_intervention_a0_v1.json"
)


class LrcDeterministicInterventionA0Test(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_frozen_config_validates(self):
        manifest = validate_config(ROOT, self.config)
        self.assertEqual(len(manifest["records"]), 64)

    def test_config_cannot_relax_both_method_signal(self):
        mutated = copy.deepcopy(self.config)
        mutated["calibration_signal_gates"]["require_signal_in_both_methods"] = False
        with self.assertRaisesRegex(ValueError, "both-method"):
            validate_config(ROOT, mutated)

    def test_config_cannot_treat_no_latent_as_equal_depth(self):
        mutated = copy.deepcopy(self.config)
        mutated["equal_depth_effect_conditions"].append("no_latent")
        with self.assertRaisesRegex(ValueError, "equal-depth"):
            validate_config(ROOT, mutated)

    def test_target_score_and_kl_controls(self):
        logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
        targets = torch.tensor([0, 1])
        nll, log_probs = _target_scores(logits, targets)
        self.assertGreater(nll, 0.0)
        self.assertAlmostEqual(_mean_kl(log_probs, log_probs), 0.0, places=7)

    def test_norm_matched_random_is_deterministic(self):
        states = [torch.tensor([3.0, 4.0]), torch.tensor([0.0, 2.0])]
        first = _norm_matched_random(states, 7)
        second = _norm_matched_random(states, 7)
        for source, left, right in zip(states, first, second):
            self.assertTrue(torch.equal(left, right))
            self.assertAlmostEqual(
                float(torch.linalg.vector_norm(source)),
                float(torch.linalg.vector_norm(left)),
                places=5,
            )

    def test_empty_records_hold_controls(self):
        summary = summarize_records([], self.config)
        self.assertEqual(summary["decision"], "HOLD_A0_CONTROLS")


if __name__ == "__main__":
    unittest.main()
