import unittest

import torch

from research.score_squashed_gumbel.policies import (
    inverse_score_squash,
    sample_score_squashed_topk,
    score_squash,
    score_squashed_selected_log_density,
    score_squashed_topk_from_gumbels,
)
from research.topk_concrete.densities import selected_gumbel_log_density


class ScoreSquashedGumbelPolicyTest(unittest.TestCase):
    def test_squash_round_trip(self):
        raw_scores = torch.linspace(-8.0, 8.0, 257, dtype=torch.float64)
        recovered = inverse_score_squash(score_squash(raw_scores))

        torch.testing.assert_close(
            recovered, raw_scores, atol=2e-12, rtol=2e-12
        )

    def test_monotone_squash_preserves_ordered_topk_support(self):
        logits = torch.tensor(
            [0.2, -0.5, 1.1, 0.4, -0.8, 0.0], dtype=torch.float64
        )
        generator = torch.Generator().manual_seed(901)
        uniforms = torch.rand(2048, 6, dtype=torch.float64, generator=generator)
        raw_gumbels = -torch.log(-torch.log(uniforms))
        sample = score_squashed_topk_from_gumbels(
            logits,
            raw_gumbels,
            top_k=3,
            temperature=0.8,
        )
        canonical_logits = torch.log_softmax(logits, dim=-1)
        expected = torch.topk(
            canonical_logits + raw_gumbels, k=3, dim=-1, sorted=True
        ).indices

        self.assertTrue(torch.equal(sample.ordered_indices, expected))

    def test_canonical_scores_are_invariant_to_common_logit_shift(self):
        logits = torch.tensor([0.2, -0.5, 1.1, 0.4], dtype=torch.float64)
        raw_gumbels = torch.randn(
            128, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(902)
        )
        left = score_squashed_topk_from_gumbels(
            logits, raw_gumbels, top_k=2, temperature=0.7
        )
        right = score_squashed_topk_from_gumbels(
            logits + 19.0, raw_gumbels, top_k=2, temperature=0.7
        )

        self.assertTrue(torch.equal(left.ordered_indices, right.ordered_indices))
        torch.testing.assert_close(left.squashed_scores, right.squashed_scores)
        torch.testing.assert_close(left.weights, right.weights)

    def test_top_one_density_matches_transformed_gumbel_max_law(self):
        logits = torch.tensor([0.4, -0.3, 1.2, 0.1], dtype=torch.float64)
        canonical = torch.log_softmax(logits, dim=-1)
        raw_maxima = torch.linspace(-3.0, 4.0, 31, dtype=torch.float64)
        squashed = score_squash(raw_maxima)
        scores = squashed.repeat_interleave(4).unsqueeze(-1)
        indices = torch.arange(4, dtype=torch.long).repeat(31).unsqueeze(-1)

        actual = score_squashed_selected_log_density(scores, indices, logits)
        unit = (squashed - 0.75) / 2.25
        transformed_max_log_density = (
            -raw_maxima
            - torch.exp(-raw_maxima)
            - torch.log1p(-unit.square())
        )
        expected = (
            transformed_max_log_density.unsqueeze(-1) + canonical.unsqueeze(0)
        ).reshape(-1)

        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)

    def test_full_support_density_matches_ordered_transformed_gumbels(self):
        logits = torch.tensor([0.4, -0.3, 1.2, 0.1], dtype=torch.float64)
        sample = sample_score_squashed_topk(
            logits,
            top_k=4,
            temperature=0.7,
            sample_shape=(128,),
            generator=torch.Generator().manual_seed(903),
        )
        actual = score_squashed_selected_log_density(
            sample.squashed_scores, sample.ordered_indices, logits
        )
        canonical = torch.log_softmax(logits, dim=-1)
        selected_logits = canonical[sample.ordered_indices]
        noise = sample.raw_selected_scores - selected_logits
        unit = (sample.squashed_scores - 0.75) / 2.25
        expected = (
            -noise - torch.exp(-noise) - torch.log1p(-unit.square())
        ).sum(dim=-1)

        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)

    def test_density_matches_independent_selected_gumbel_change_of_variables(self):
        logits = torch.tensor(
            [0.4, -0.3, 1.2, 0.1, -0.8], dtype=torch.float64
        )
        sample = sample_score_squashed_topk(
            logits,
            top_k=3,
            temperature=0.7,
            sample_shape=(256,),
            generator=torch.Generator().manual_seed(906),
        )
        actual = score_squashed_selected_log_density(
            sample.squashed_scores, sample.ordered_indices, logits
        )
        canonical = torch.log_softmax(logits, dim=-1)
        raw_density = selected_gumbel_log_density(
            sample.raw_selected_scores,
            sample.ordered_indices,
            canonical,
        )
        unit = (sample.squashed_scores - 0.75) / 2.25
        expected = raw_density - torch.log1p(-unit.square()).sum(dim=-1)

        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)

    def test_exact_importance_ratio_has_unit_expectation(self):
        old_logits = torch.tensor(
            [0.2, -0.4, 0.7, 0.1, -0.2, 0.5], dtype=torch.float64
        )
        current_logits = old_logits + torch.tensor(
            [0.08, -0.05, 0.03, -0.02, 0.04, -0.08], dtype=torch.float64
        )
        sample = sample_score_squashed_topk(
            old_logits,
            top_k=3,
            temperature=0.7,
            sample_shape=(100_000,),
            generator=torch.Generator().manual_seed(904),
        )
        old_density = score_squashed_selected_log_density(
            sample.squashed_scores, sample.ordered_indices, old_logits
        )
        current_density = score_squashed_selected_log_density(
            sample.squashed_scores, sample.ordered_indices, current_logits
        )

        self.assertAlmostEqual(
            float(torch.exp(current_density - old_density).mean()),
            1.0,
            delta=0.01,
        )

    def test_candidate_mask_has_common_continuous_support(self):
        old_logits = torch.tensor([0.4, -0.3, 1.2, 0.1], dtype=torch.float64)
        current_logits = old_logits + torch.tensor(
            [0.2, 0.0, -0.1, 0.0], dtype=torch.float64
        )
        candidate_mask = torch.tensor([True, False, True, True])
        sample = sample_score_squashed_topk(
            old_logits,
            top_k=2,
            temperature=0.7,
            candidate_mask=candidate_mask,
            sample_shape=(1024,),
            generator=torch.Generator().manual_seed(905),
        )
        current_density = score_squashed_selected_log_density(
            sample.squashed_scores,
            sample.ordered_indices,
            current_logits,
            candidate_mask=candidate_mask,
        )

        self.assertTrue(candidate_mask[sample.ordered_indices].all())
        self.assertTrue(torch.isfinite(current_density).all())

    def test_exact_policy_score_has_zero_mean(self):
        logits = torch.tensor(
            [0.4, -0.3, 1.2, 0.1, -0.8, 0.6], dtype=torch.float64
        )
        sample = sample_score_squashed_topk(
            logits,
            top_k=3,
            temperature=0.7,
            sample_shape=(50_000,),
            generator=torch.Generator().manual_seed(907),
        )
        differentiable_logits = logits.detach().clone().requires_grad_(True)
        mean_log_density = score_squashed_selected_log_density(
            sample.squashed_scores,
            sample.ordered_indices,
            differentiable_logits,
        ).mean()
        score_mean = torch.autograd.grad(mean_log_density, differentiable_logits)[0]

        self.assertLess(float(score_mean.norm()), 0.015)


if __name__ == "__main__":
    unittest.main()
