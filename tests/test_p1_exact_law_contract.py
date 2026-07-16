import unittest

import torch

from research.behavioral_geometry.p1_exact_law_contract import (
    ExactAutoregressiveLaw,
    enumerate_autoregressive_law,
    exact_chain_rule_kl,
    exact_joint_kl,
    full_support_probs_from_logits,
    source_visible_top_p_probs,
)


def _left_rule(prefix: tuple[int, ...]) -> torch.Tensor:
    history = sum((index + 1) * token for index, token in enumerate(prefix))
    return torch.tensor([0.7 + 0.1 * history, -0.2, 0.35 - 0.05 * history], dtype=torch.float64)


def _right_rule(prefix: tuple[int, ...]) -> torch.Tensor:
    history = sum((index + 1) * token for index, token in enumerate(prefix))
    return torch.tensor([0.1 + 0.05 * history, 0.4, -0.15 + 0.08 * history], dtype=torch.float64)


class ExactLawContractTest(unittest.TestCase):
    def test_full_softmax_has_exact_positive_joint_support(self):
        law = enumerate_autoregressive_law(_left_rule, horizon=3, vocabulary_size=3)
        self.assertEqual(len(law.joint_probabilities), 27)
        self.assertTrue(law.has_full_joint_support)
        torch.testing.assert_close(law.total_probability, torch.tensor(1.0, dtype=torch.float64))

    def test_joint_enumeration_equals_expected_chain_rule_kl(self):
        left = enumerate_autoregressive_law(_left_rule, horizon=3, vocabulary_size=3)
        right = enumerate_autoregressive_law(_right_rule, horizon=3, vocabulary_size=3)
        direct = exact_joint_kl(left, right)
        chain = exact_chain_rule_kl(_left_rule, _right_rule, horizon=3, vocabulary_size=3)
        torch.testing.assert_close(direct, chain, atol=1e-13, rtol=0)

    def test_truncated_visible_law_exposes_support_violation_as_infinite_kl(self):
        logits = torch.tensor([2.5, 0.3, -1.2], dtype=torch.float64)
        full = full_support_probs_from_logits(logits)
        truncated = source_visible_top_p_probs(logits, top_p=0.5)
        self.assertTrue(torch.all(full > 0))
        self.assertTrue(torch.any(truncated == 0))
        left = ExactAutoregressiveLaw(
            1,
            3,
            {(0,): full[0], (1,): full[1], (2,): full[2]},
        )
        right = ExactAutoregressiveLaw(
            1,
            3,
            {(0,): truncated[0], (1,): truncated[1], (2,): truncated[2]},
        )
        self.assertTrue(torch.isinf(exact_joint_kl(left, right)))

    def test_visible_top_p_rule_uses_source_boundary_convention(self):
        logits = torch.tensor([1.0, 0.0, -3.0], dtype=torch.float64)
        probs = full_support_probs_from_logits(logits)
        # A top-p exactly equal to the top token's mass retains the second
        # token under the source helper because it excludes only values > top_p.
        truncated = source_visible_top_p_probs(logits, top_p=float(probs[0]))
        self.assertGreater(float(truncated[1]), 0.0)
        self.assertEqual(float(truncated[2]), 0.0)


if __name__ == "__main__":
    unittest.main()
