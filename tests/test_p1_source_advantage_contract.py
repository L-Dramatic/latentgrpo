import subprocess
import unittest

import numpy as np
import torch

from research.behavioral_geometry.p1_official_ppo_replay import (
    OFFICIAL_PPO_CORE_PATH,
    PINNED_OFFICIAL_SOURCE_COMMIT,
    replay_pinned_include_advantage,
)
from research.behavioral_geometry.p1_source_advantage_contract import (
    source_actor_zero_max_length_advantages,
    source_include_advantage_contract,
)


class SourceAdvantageContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rewards = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [1.3, 0.0, 0.0],
                [-0.3, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [-0.8, 0.0, 0.0],
            ],
            dtype=torch.float64,
        )
        self.mask = torch.tensor(
            [[0, 1, 1]] * 8, dtype=torch.float64
        )
        self.index = np.array([0] * 8)
        self.old_log_probs = torch.tensor(
            [
                [-9.0, -0.2, -0.2],
                [-9.0, -0.1, -0.1],
                [-9.0, -0.4, -0.4],
                [-9.0, -0.8, -0.8],
                [-9.0, -0.3, -0.3],
                [-9.0, -1.0, -1.0],
                [-9.0, -0.5, -0.5],
                [-9.0, -0.7, -0.7],
            ],
            dtype=torch.float64,
        )

    def test_replay_targets_same_pinned_official_checkout(self):
        observed = subprocess.check_output(
            ["git", "-C", str(OFFICIAL_PPO_CORE_PATH.parents[3]), "rev-parse", "HEAD"], text=True
        ).strip()
        self.assertEqual(observed, PINNED_OFFICIAL_SOURCE_COMMIT)

    def test_include_advantage_first_mask_winner_matches_source(self):
        source = replay_pinned_include_advantage(
            self.rewards.clone(),
            self.mask,
            self.index,
            old_log_probs=self.old_log_probs,
        )
        contract = source_include_advantage_contract(
            self.rewards.clone(), self.mask, self.index, old_log_probs=self.old_log_probs
        )
        torch.testing.assert_close(source[0], contract[0])
        # Positive candidates other than the largest mean-logprob path lose
        # only their first valid response position.
        self.assertEqual(float(contract[0][0, 1]), 0.0)
        self.assertGreater(float(contract[0][1, 1]), 0.0)
        self.assertGreater(float(contract[0][1, 2]), 0.0)

    def test_actor_max_length_branch_zeros_entire_advantage_row_in_place(self):
        advantages = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
        result = source_actor_zero_max_length_advantages(
            advantages,
            attention_mask,
            response_length=3,
            exclude_overlong_samples_from_advantage=False,
        )
        torch.testing.assert_close(result[0], torch.zeros(3))
        torch.testing.assert_close(result[1], torch.tensor([4.0, 5.0, 6.0]))

    def test_actor_max_length_branch_is_disabled_when_advantages_were_preexcluded(self):
        advantages = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.ones((1, 4), dtype=torch.long)
        result = source_actor_zero_max_length_advantages(
            advantages,
            mask,
            response_length=3,
            exclude_overlong_samples_from_advantage=True,
        )
        torch.testing.assert_close(result, torch.tensor([[1.0, 2.0, 3.0]]))


if __name__ == "__main__":
    unittest.main()
