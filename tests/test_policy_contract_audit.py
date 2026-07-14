import unittest

import torch

from research.policy_contract_audit.contracts import (
    aggregate_log_ratios,
    ppo_clip_mask,
    score_mean_diagnostics,
)
from research.policy_contract_audit.lepo import (
    apply_lepo_sampling_filters,
    lepo_soft_target_score,
    replay_lepo_latent_step,
)
from research.simplex_policy.densities import concrete_score, sample_concrete


class PolicyContractAuditTest(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64

    def test_lepo_archive_covers_filtered_support_and_reconstructs_action(self):
        logits = torch.tensor([1.1, 0.4, -0.2, -0.5, -0.8], dtype=self.dtype)
        gumbels = torch.tensor([0.2, -0.1, 1.4, -0.7, 0.8], dtype=self.dtype)
        embeddings = torch.eye(5, dtype=self.dtype)
        step = replay_lepo_latent_step(
            logits, embeddings, gumbels, temperature=0.7, top_k=2
        )
        torch.testing.assert_close(step.archived_mass, torch.tensor(1.0, dtype=self.dtype))
        reconstructed = torch.zeros_like(step.full_action).scatter(
            -1, step.archived_topk_indices, step.archived_topk_probabilities
        )
        torch.testing.assert_close(step.full_action, reconstructed)
        torch.testing.assert_close(step.executed_embedding, step.full_action)

    def test_lepo_proxy_id_can_differ_from_executed_action_mode(self):
        logits = torch.tensor([2.0, 1.7, -1.0], dtype=self.dtype)
        gumbels = torch.tensor([-2.0, 2.0, 0.0], dtype=self.dtype)
        embeddings = torch.eye(3, dtype=self.dtype)
        step = replay_lepo_latent_step(
            logits, embeddings, gumbels, temperature=0.3, top_k=2
        )
        self.assertEqual(int(step.proxy_token_id), 0)
        self.assertEqual(int(step.full_action.argmax()), 1)

    def test_filtered_out_surrogate_score_is_exactly_negative_model_mass(self):
        logits = torch.tensor([1.2, 0.6, 0.1, -0.5, -1.0], dtype=self.dtype)
        filtered = apply_lepo_sampling_filters(logits, top_k=2, top_p=0.95)
        active = torch.isfinite(filtered)
        gumbels = torch.tensor([0.3, -0.2, 1.0, -0.4, 0.1], dtype=self.dtype)
        step = replay_lepo_latent_step(
            logits,
            torch.eye(5, dtype=self.dtype),
            gumbels,
            temperature=0.7,
            top_k=2,
            top_p=0.95,
        )
        score = lepo_soft_target_score(
            step.archived_topk_probabilities,
            step.archived_topk_indices,
            logits,
        )
        expected = -torch.softmax(logits, dim=-1)
        torch.testing.assert_close(score[~active], expected[~active])

    def test_lepo_soft_target_score_is_not_pointwise_concrete_score(self):
        generator = torch.Generator().manual_seed(77)
        logits = torch.tensor([1.3, 0.4, -0.1, -0.8], dtype=self.dtype)
        filtered = apply_lepo_sampling_filters(logits, top_k=3, top_p=0.95)
        active = torch.isfinite(filtered)
        active_logits = filtered[active]
        active_samples, _ = sample_concrete(
            active_logits,
            temperature=0.8,
            sample_shape=(30000,),
            generator=generator,
        )
        samples = torch.zeros(
            active_samples.shape[0], logits.numel(), dtype=self.dtype
        )
        samples[:, active] = active_samples
        topk_probs, topk_ids = torch.topk(samples, k=3, dim=-1)
        expanded_logits = logits.expand_as(samples)
        surrogate_scores = lepo_soft_target_score(
            topk_probs, topk_ids, expanded_logits
        )
        exact_active_scores = concrete_score(
            active_samples,
            active_logits.expand_as(active_samples),
            temperature=0.8,
        )
        exact_scores = torch.zeros_like(samples)
        exact_scores[:, active] = exact_active_scores
        difference = (surrogate_scores - exact_scores).norm(dim=-1).mean()
        self.assertGreater(float(difference), 0.25)

        exact_diagnostics = score_mean_diagnostics(exact_active_scores)
        self.assertLess(exact_diagnostics.signal_to_noise, 4.0)

    def test_sum_and_mean_aggregation_can_flip_clip_decisions(self):
        component_log_ratios = torch.tensor(
            [[0.11, 0.11, 0.11], [-0.09, -0.09, -0.09]], dtype=self.dtype
        )
        mask = torch.ones_like(component_log_ratios, dtype=torch.bool)
        advantages = torch.tensor([1.0, -1.0], dtype=self.dtype)
        summed = aggregate_log_ratios(component_log_ratios, mask, reduction="sum")
        averaged = aggregate_log_ratios(component_log_ratios, mask, reduction="mean")
        summed_clipped = ppo_clip_mask(summed, advantages, epsilon=0.2)
        averaged_clipped = ppo_clip_mask(averaged, advantages, epsilon=0.2)
        self.assertTrue(bool(torch.all(summed_clipped)))
        self.assertFalse(bool(torch.any(averaged_clipped)))


if __name__ == "__main__":
    unittest.main()
