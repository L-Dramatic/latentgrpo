import unittest
from itertools import permutations

import numpy as np
import torch

from research.topk_concrete.densities import (
    conditional_topk_weight_log_density,
    naive_selected_gumbel_log_density,
    ordered_plackett_luce_log_probability,
    sample_topk_concrete,
    selected_gumbel_log_density,
    topk_concrete_log_density,
)


class TopKConcreteDensityTest(unittest.TestCase):
    def test_top_one_reduces_to_categorical_distribution(self):
        logits = torch.tensor([0.4, -0.3, 1.2, 0.1], dtype=torch.float64)
        indices = torch.arange(4, dtype=torch.long).unsqueeze(-1)
        weights = torch.ones(4, 1, dtype=torch.float64)

        actual = topk_concrete_log_density(
            weights, indices, logits, temperature=0.7, gumbel_scale=1.3
        )
        expected = torch.log_softmax(logits / 1.3, dim=-1)

        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    def test_full_vocabulary_reduces_to_concrete_density(self):
        logits = torch.tensor([0.4, -0.3, 1.2, 0.1], dtype=torch.float64)
        sample = sample_topk_concrete(
            logits,
            top_k=4,
            temperature=0.7,
            sample_shape=(128,),
            generator=torch.Generator().manual_seed(41),
        )
        actual = topk_concrete_log_density(
            sample.weights,
            sample.ordered_indices,
            logits,
            temperature=0.7,
        )
        full_weights = torch.zeros(128, 4, dtype=torch.float64).scatter(
            -1, sample.ordered_indices, sample.weights
        )
        reference = torch.distributions.RelaxedOneHotCategorical(
            temperature=torch.tensor(0.7, dtype=torch.float64), logits=logits
        ).log_prob(full_weights)

        torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-10)

    def test_executed_density_is_invariant_to_common_logit_shift(self):
        logits = torch.tensor([0.4, -0.3, 1.2, 0.1, -0.8], dtype=torch.float64)
        sample = sample_topk_concrete(
            logits,
            top_k=3,
            temperature=0.8,
            sample_shape=(64,),
            generator=torch.Generator().manual_seed(42),
        )
        left = topk_concrete_log_density(
            sample.weights, sample.ordered_indices, logits, temperature=0.8
        )
        right = topk_concrete_log_density(
            sample.weights, sample.ordered_indices, logits + 17.0, temperature=0.8
        )

        torch.testing.assert_close(left, right, atol=1e-11, rtol=1e-11)

    def test_selection_corrected_auxiliary_density_adds_missing_cdf_event(self):
        logits = torch.tensor([0.2, -0.4, 0.7, 0.1], dtype=torch.float64)
        sample = sample_topk_concrete(
            logits,
            top_k=2,
            temperature=0.6,
            sample_shape=(32,),
            generator=torch.Generator().manual_seed(43),
        )
        naive = naive_selected_gumbel_log_density(
            sample.selected_scores, sample.ordered_indices, logits
        )
        corrected = selected_gumbel_log_density(
            sample.selected_scores, sample.ordered_indices, logits
        )

        self.assertTrue(torch.all(corrected < naive))

    def test_exact_importance_ratio_has_unit_expectation(self):
        old_logits = torch.tensor(
            [0.2, -0.4, 0.7, 0.1, -0.2, 0.5], dtype=torch.float64
        )
        new_logits = old_logits + torch.tensor(
            [0.05, -0.03, 0.02, -0.01, 0.03, -0.04], dtype=torch.float64
        )
        sample = sample_topk_concrete(
            old_logits,
            top_k=3,
            temperature=0.7,
            sample_shape=(100_000,),
            generator=torch.Generator().manual_seed(44),
        )
        old_density = topk_concrete_log_density(
            sample.weights, sample.ordered_indices, old_logits, temperature=0.7
        )
        new_density = topk_concrete_log_density(
            sample.weights, sample.ordered_indices, new_logits, temperature=0.7
        )

        self.assertAlmostEqual(float(torch.exp(new_density - old_density).mean()), 1.0, delta=0.01)

    def test_small_topk_density_normalizes_by_independent_quadrature(self):
        logits = torch.tensor([0.4, -0.3, 1.2], dtype=torch.float64)
        nodes, quadrature_weights = np.polynomial.legendre.leggauss(512)
        first_weight = torch.from_numpy(0.25 * nodes + 0.75)
        integration_weights = torch.from_numpy(0.25 * quadrature_weights)
        simplex_weights = torch.stack([first_weight, 1.0 - first_weight], dim=-1)

        total_mass = torch.zeros((), dtype=torch.float64)
        for ordered_pair in permutations(range(3), 2):
            indices = torch.tensor(ordered_pair, dtype=torch.long).expand(512, -1)
            density = torch.exp(
                topk_concrete_log_density(
                    simplex_weights,
                    indices,
                    logits,
                    temperature=0.7,
                    gumbel_scale=0.7,
                )
            )
            total_mass += torch.dot(integration_weights, density)

        self.assertAlmostEqual(float(total_mass), 1.0, delta=1e-10)

    def test_candidate_mask_restricts_sampling_and_density_support(self):
        logits = torch.tensor([0.4, -0.3, 1.2, 0.1], dtype=torch.float64)
        candidate_mask = torch.tensor([True, False, True, False])
        sample = sample_topk_concrete(
            logits,
            top_k=2,
            temperature=0.7,
            candidate_mask=candidate_mask,
            sample_shape=(256,),
            generator=torch.Generator().manual_seed(45),
        )
        self.assertTrue(candidate_mask[sample.ordered_indices].all())

        indices = torch.arange(4, dtype=torch.long).unsqueeze(-1)
        weights = torch.ones(4, 1, dtype=torch.float64)
        actual = topk_concrete_log_density(
            weights,
            indices,
            logits,
            temperature=0.7,
            candidate_mask=candidate_mask,
        )
        expected = torch.log_softmax(
            logits.masked_fill(~candidate_mask, -torch.inf), dim=-1
        )
        torch.testing.assert_close(actual[candidate_mask], expected[candidate_mask])
        self.assertTrue(torch.isneginf(actual[~candidate_mask]).all())

    def test_selection_correction_uses_only_eligible_unselected_items(self):
        logits = torch.tensor([0.2, -0.4, 0.7, 0.1], dtype=torch.float64)
        candidate_mask = torch.tensor([True, False, True, False])
        sample = sample_topk_concrete(
            logits,
            top_k=2,
            temperature=0.6,
            candidate_mask=candidate_mask,
            sample_shape=(32,),
            generator=torch.Generator().manual_seed(46),
        )
        naive = naive_selected_gumbel_log_density(
            sample.selected_scores, sample.ordered_indices, logits
        )
        corrected = selected_gumbel_log_density(
            sample.selected_scores,
            sample.ordered_indices,
            logits,
            candidate_mask=candidate_mask,
        )

        torch.testing.assert_close(corrected, naive)

    def test_joint_density_factorizes_into_support_and_conditional_weight(self):
        logits = torch.tensor(
            [0.4, -0.3, 1.2, 0.1, -0.8], dtype=torch.float64
        )
        sample = sample_topk_concrete(
            logits,
            top_k=3,
            temperature=0.8,
            gumbel_scale=1.1,
            sample_shape=(128,),
            generator=torch.Generator().manual_seed(47),
        )
        joint = topk_concrete_log_density(
            sample.weights,
            sample.ordered_indices,
            logits,
            temperature=0.8,
            gumbel_scale=1.1,
        )
        support = ordered_plackett_luce_log_probability(
            sample.ordered_indices,
            logits,
            gumbel_scale=1.1,
        )
        conditional = conditional_topk_weight_log_density(
            sample.weights,
            sample.ordered_indices,
            logits,
            temperature=0.8,
            gumbel_scale=1.1,
        )

        torch.testing.assert_close(
            joint, support + conditional, atol=1e-12, rtol=1e-12
        )

    def test_ordered_support_probabilities_normalize(self):
        logits = torch.tensor([0.4, -0.3, 1.2, 0.1], dtype=torch.float64)
        ordered_supports = torch.tensor(
            list(permutations(range(4), 3)), dtype=torch.long
        )

        log_probabilities = ordered_plackett_luce_log_probability(
            ordered_supports,
            logits,
            gumbel_scale=0.7,
        )

        self.assertAlmostEqual(
            float(torch.exp(log_probabilities).sum()), 1.0, delta=1e-12
        )

    def test_per_support_quadrature_matches_plackett_luce_mass(self):
        logits = torch.tensor([0.4, -0.3, 1.2], dtype=torch.float64)
        nodes, quadrature_weights = np.polynomial.legendre.leggauss(512)
        first_weight = torch.from_numpy(0.25 * nodes + 0.75)
        integration_weights = torch.from_numpy(0.25 * quadrature_weights)
        simplex_weights = torch.stack([first_weight, 1.0 - first_weight], dim=-1)

        for ordered_pair in permutations(range(3), 2):
            indices = torch.tensor(ordered_pair, dtype=torch.long).expand(512, -1)
            integrated_joint_mass = torch.dot(
                integration_weights,
                torch.exp(
                    topk_concrete_log_density(
                        simplex_weights,
                        indices,
                        logits,
                        temperature=0.7,
                        gumbel_scale=0.7,
                    )
                ),
            )
            support_log_probability = ordered_plackett_luce_log_probability(
                torch.tensor(ordered_pair, dtype=torch.long),
                logits,
                gumbel_scale=0.7,
            )

            self.assertAlmostEqual(
                float(integrated_joint_mass),
                float(torch.exp(support_log_probability)),
                delta=1e-11,
            )


if __name__ == "__main__":
    unittest.main()
