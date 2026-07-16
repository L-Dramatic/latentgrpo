import subprocess
import unittest

import torch

from research.behavioral_geometry.p1_official_ppo_replay import (
    OFFICIAL_PPO_CORE_PATH,
    PINNED_OFFICIAL_SOURCE_COMMIT,
    replay_pinned_policy_loss,
)
from research.behavioral_geometry.p1_source_ppo_contract import source_policy_loss_contract


class SourcePPOContractTest(unittest.TestCase):
    def test_replay_targets_same_pinned_official_checkout(self):
        observed = subprocess.check_output(
            ["git", "-C", str(OFFICIAL_PPO_CORE_PATH.parents[3]), "rev-parse", "HEAD"], text=True
        ).strip()
        self.assertEqual(observed, PINNED_OFFICIAL_SOURCE_COMMIT)

    def test_clipped_dual_clip_and_negative_weight_match_source(self):
        old = torch.tensor([[0.0, -0.4, 0.1], [-0.2, 0.3, 0.0]], dtype=torch.float64)
        current = torch.tensor([[0.8, -1.1, 0.0], [-1.8, 0.5, 1.0]], dtype=torch.float64)
        advantages = torch.tensor([[1.0, -1.5, 0.4], [-0.8, 1.2, -0.3]], dtype=torch.float64)
        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
        kwargs = dict(
            cliprange=0.2,
            cliprange_low=0.15,
            cliprange_high=0.25,
            clip_ratio_c=3.0,
            neg_adv_weight=1.7,
        )
        source = replay_pinned_policy_loss(old, current, advantages, mask, **kwargs)
        contract = source_policy_loss_contract(old, current, advantages, mask, **kwargs)
        for source_value, contract_value in zip(source, contract):
            torch.testing.assert_close(source_value, contract_value)

    def test_frozen_self_ratio_is_one_zero_kl_and_has_no_clipping(self):
        old = torch.tensor([[0.2, -0.7, 0.4]], dtype=torch.float64)
        current = old.clone()
        advantages = torch.tensor([[1.5, -0.8, 0.4]], dtype=torch.float64)
        mask = torch.ones_like(old)
        source = replay_pinned_policy_loss(
            old,
            current,
            advantages,
            mask,
            cliprange=0.2,
            cliprange_low=0.2,
            cliprange_high=0.2,
            clip_ratio_c=3.0,
            neg_adv_weight=1.0,
        )
        expected_loss = (-advantages).mean()
        torch.testing.assert_close(source[0], expected_loss)
        torch.testing.assert_close(source[1], torch.zeros_like(source[1]))
        torch.testing.assert_close(source[2], torch.zeros_like(source[2]))
        torch.testing.assert_close(source[3], torch.zeros_like(source[3]))

    def test_policy_loss_contract_preserves_source_gradient(self):
        old = torch.tensor([[0.1, -0.5]], dtype=torch.float64)
        current_source = torch.tensor([[0.9, -1.8]], dtype=torch.float64, requires_grad=True)
        current_contract = current_source.detach().clone().requires_grad_(True)
        advantages = torch.tensor([[1.1, -0.9]], dtype=torch.float64)
        mask = torch.ones_like(old)
        kwargs = dict(
            cliprange=0.2,
            cliprange_low=0.2,
            cliprange_high=0.2,
            clip_ratio_c=3.0,
            neg_adv_weight=1.3,
        )
        source_loss = replay_pinned_policy_loss(
            old, current_source, advantages, mask, **kwargs
        )[0]
        contract_loss = source_policy_loss_contract(
            old, current_contract, advantages, mask, **kwargs
        )[0]
        source_loss.backward()
        contract_loss.backward()
        torch.testing.assert_close(source_loss, contract_loss)
        torch.testing.assert_close(current_source.grad, current_contract.grad)


if __name__ == "__main__":
    unittest.main()
