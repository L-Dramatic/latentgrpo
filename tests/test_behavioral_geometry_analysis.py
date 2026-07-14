import unittest

import torch

from research.behavioral_geometry.analysis import (
    average_ranks,
    continuation_rank_report,
    cumulative_prefix_kl,
    rank_diagnostic,
    spearman_correlation,
)


class BehavioralGeometryAnalysisTest(unittest.TestCase):
    def test_average_ranks_handle_ties(self):
        ranks = average_ranks([3.0, 1.0, 1.0, 2.0])
        torch.testing.assert_close(
            ranks, torch.tensor([3.0, 0.5, 0.5, 2.0], dtype=torch.float64)
        )

    def test_spearman_detects_perfect_and_reversed_order(self):
        self.assertAlmostEqual(spearman_correlation([1, 2, 3], [2, 4, 8]), 1.0)
        self.assertAlmostEqual(spearman_correlation([1, 2, 3], [8, 4, 2]), -1.0)

    def test_rank_diagnostic_exposes_hidden_top_risk(self):
        result = rank_diagnostic(
            predictor=[4, 3, 2, 1],
            target=[1, 2, 3, 4],
            top_fraction=0.25,
            predictor_screen_fraction=0.5,
        )
        self.assertEqual(result.top_risk_recall, 0.0)
        self.assertEqual(result.hidden_top_risk_fraction, 1.0)
        self.assertEqual(result.top_risk_count, 1)

    def test_prefix_report_detects_when_second_token_explains_full_risk(self):
        rows = []
        for first_step, full_risk in zip(
            [4.0, 3.0, 2.0, 1.0], [4.0, 5.0, 6.0, 7.0]
        ):
            rows.append(
                {
                    "total_kl": full_risk,
                    "mean_step_kl": [first_step, full_risk - first_step, 0.0],
                    "coordinate_distances": {
                        "euclidean": full_risk,
                        "cosine": full_risk,
                        "diagonal_mahalanobis": full_risk,
                    },
                }
            )

        report = continuation_rank_report(rows, prefix_horizons=[1, 2])

        self.assertAlmostEqual(report["next_token_kl"]["spearman"], -1.0)
        self.assertAlmostEqual(report["prefix_kl_h2"]["spearman"], 1.0)

    def test_prefix_kl_rejects_horizon_beyond_recording(self):
        self.assertAlmostEqual(cumulative_prefix_kl([0.1, 0.2], 2), 0.3)
        with self.assertRaisesRegex(ValueError, "within the recorded"):
            cumulative_prefix_kl([0.1, 0.2], 3)


if __name__ == "__main__":
    unittest.main()
