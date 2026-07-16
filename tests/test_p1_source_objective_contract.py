import subprocess
import unittest

import torch

from research.behavioral_geometry.p1_official_objective_replay import (
    OFFICIAL_TORCH_FUNCTIONAL_PATH,
    PINNED_OFFICIAL_SOURCE_COMMIT,
    replay_pinned_gumbel_likelihood,
)
from research.behavioral_geometry.p1_source_objective_contract import (
    source_gumbel_likelihood_contract,
)


class SourceObjectiveContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.logits = torch.tensor(
            [[1.3, 0.2, -0.4, 0.7, -1.0]], dtype=torch.float64, requires_grad=True
        )
        self.ids = torch.tensor([[0, 3, 1]])
        self.labels = torch.tensor([0])
        base_log_probs = torch.log_softmax(self.logits.detach().float(), dim=-1)
        self.scores = base_log_probs[:, self.ids[0]] + torch.tensor([[0.25, -0.40, 0.10]])

    def test_replay_targets_same_pinned_official_checkout(self):
        observed = subprocess.check_output(
            ["git", "-C", str(OFFICIAL_TORCH_FUNCTIONAL_PATH.parents[3]), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        self.assertEqual(observed, PINNED_OFFICIAL_SOURCE_COMMIT)

    def test_soft_gumbel_forward_value_matches_unmodified_source_formula(self):
        source = replay_pinned_gumbel_likelihood(
            self.logits,
            self.ids,
            self.scores,
            self.labels,
            top_p=0.8,
            temperature=0.7,
        )
        contract = source_gumbel_likelihood_contract(
            self.logits,
            self.ids,
            self.scores,
            self.labels,
            temperature=0.7,
        )
        torch.testing.assert_close(source, contract)

    def test_negative_advantage_sign_rule_changes_backward_not_forward_value(self):
        source_logits = self.logits.detach().clone().requires_grad_(True)
        contract_logits = self.logits.detach().clone().requires_grad_(True)
        advantages = torch.tensor([-1.0])
        source = replay_pinned_gumbel_likelihood(
            source_logits,
            self.ids,
            self.scores,
            self.labels,
            top_p=1.0,
            temperature=1.0,
            advantages=advantages,
        )
        contract = source_gumbel_likelihood_contract(
            contract_logits,
            self.ids,
            self.scores,
            self.labels,
            temperature=1.0,
            advantages=advantages,
        )
        source.sum().backward()
        contract.sum().backward()

        no_advantage = source_gumbel_likelihood_contract(
            self.logits.detach(), self.ids, self.scores, self.labels, temperature=1.0
        )
        torch.testing.assert_close(source, contract)
        torch.testing.assert_close(source_logits.grad, contract_logits.grad)
        torch.testing.assert_close(source.detach(), no_advantage)
        self.assertFalse(torch.allclose(contract_logits.grad, torch.zeros_like(contract_logits.grad)))

    def test_hard_token_sentinel_uses_temperature_scaled_answer_logprob(self):
        ids = torch.tensor([[3, -100, -100]])
        scores = torch.tensor([[0.2, 0.0, 0.0]])
        label = torch.tensor([3])
        source = replay_pinned_gumbel_likelihood(
            self.logits, ids, scores, label, top_p=1.0, temperature=0.5
        )
        contract = source_gumbel_likelihood_contract(
            self.logits, ids, scores, label, temperature=0.5
        )
        # The source casts logits for the Gumbel path, but its hard-token
        # temperature tensor uses the incoming logits dtype and therefore
        # promotes this artificial float64 test input back to float64.
        expected = torch.log_softmax(self.logits / 0.5, dim=-1)[:, 3]
        torch.testing.assert_close(source, contract)
        torch.testing.assert_close(contract, expected)


if __name__ == "__main__":
    unittest.main()
