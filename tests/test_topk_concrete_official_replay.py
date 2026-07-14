import unittest

import torch

from research.topk_concrete.official_replay import (
    official_surrogate_log_density,
    replay_from_gumbels,
    top_p_candidate_mask,
)


class OfficialReplayTest(unittest.TestCase):
    def test_top_p_mask_keeps_prefix_crossing_threshold_and_at_least_topk(self):
        logits = torch.log(torch.tensor([0.50, 0.25, 0.15, 0.07, 0.03], dtype=torch.float64))
        mask = top_p_candidate_mask(logits, top_p=0.70, top_k=3)
        torch.testing.assert_close(
            mask, torch.tensor([True, True, True, False, False])
        )

    def test_one_sided_shift_does_not_change_executed_action(self):
        logits = torch.tensor([0.4, -0.3, 1.2, 0.1], dtype=torch.float64)
        raw_gumbels = torch.tensor(
            [[-2.0, -0.4, 3.5, 0.8], [0.2, 1.4, -1.8, 2.1]],
            dtype=torch.float64,
        )
        mask = torch.ones(4, dtype=torch.bool)
        centered = replay_from_gumbels(
            logits,
            raw_gumbels,
            mask,
            top_k=3,
            temperature=0.8,
            noise_scale=1.0,
            clip_bounds=(-1.5, 3.0),
            one_sided=False,
        )
        shifted = replay_from_gumbels(
            logits,
            raw_gumbels,
            mask,
            top_k=3,
            temperature=0.8,
            noise_scale=1.0,
            clip_bounds=(-1.5, 3.0),
            one_sided=True,
        )
        torch.testing.assert_close(centered.ordered_indices, shifted.ordered_indices)
        torch.testing.assert_close(centered.weights, shifted.weights)
        torch.testing.assert_close(
            shifted.selected_scores - centered.selected_scores,
            torch.full_like(centered.selected_scores, 1.5),
        )

    def test_official_surrogate_is_mean_of_selected_standard_gumbels(self):
        logits = torch.tensor([0.4, -0.3, 1.2, 0.1], dtype=torch.float64)
        indices = torch.tensor([[2, 0], [3, 1]], dtype=torch.long)
        log_probabilities = torch.log_softmax(logits, dim=-1)
        margins = torch.tensor([[1.2, 0.4], [0.7, 1.5]], dtype=torch.float64)
        scores = log_probabilities[indices] + margins
        expected = (-margins - torch.exp(-margins)).mean(dim=-1)
        actual = official_surrogate_log_density(scores, indices, logits)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
