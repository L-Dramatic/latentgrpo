import unittest

import torch

from research.simplex_policy.densities import (
    auxiliary_gumbel_log_density,
    concrete_log_density,
    concrete_score,
    latent_grpo_surrogate_log_prob,
    ppo_clip_mask,
    sample_concrete,
)


class SimplexPolicyDensityTest(unittest.TestCase):
    def test_concrete_density_matches_torch_reference(self):
        logits = torch.tensor([0.2, -0.4, 0.7], dtype=torch.float64)
        generator = torch.Generator().manual_seed(9)
        samples, _ = sample_concrete(
            logits,
            temperature=0.8,
            sample_shape=(32,),
            generator=generator,
        )
        actual = concrete_log_density(
            samples, logits.expand_as(samples), temperature=0.8
        )
        reference = torch.distributions.RelaxedOneHotCategorical(
            temperature=torch.tensor(0.8, dtype=torch.float64), logits=logits
        ).log_prob(samples)
        torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-10)

    def test_concrete_density_and_score_are_shift_invariant(self):
        logits = torch.tensor([0.1, -0.6, 0.9], dtype=torch.float64)
        sample = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
        baseline_density = concrete_log_density(sample, logits, temperature=0.7)
        shifted_density = concrete_log_density(sample, logits + 11.0, temperature=0.7)
        baseline_score = concrete_score(sample, logits, temperature=0.7)
        shifted_score = concrete_score(sample, logits + 11.0, temperature=0.7)
        torch.testing.assert_close(baseline_density, shifted_density)
        torch.testing.assert_close(baseline_score, shifted_score)
        self.assertAlmostEqual(float(baseline_score.sum()), 0.0, places=12)

    def test_analytic_concrete_score_matches_autograd(self):
        sample = torch.tensor([0.15, 0.25, 0.6], dtype=torch.float64)
        logits = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64, requires_grad=True)
        density = concrete_log_density(sample, logits, temperature=0.9)
        density.backward()
        expected = concrete_score(sample, logits.detach(), temperature=0.9)
        torch.testing.assert_close(logits.grad, expected)

    def test_auxiliary_ratio_changes_under_behaviorally_null_score_offset(self):
        old_logits = torch.tensor([0.1, -0.2, 0.4], dtype=torch.float64)
        new_logits = torch.tensor([0.3, -0.4, 0.2], dtype=torch.float64)
        scores = torch.tensor([0.5, 0.1, 1.0], dtype=torch.float64)
        ratio = auxiliary_gumbel_log_density(scores, new_logits) - auxiliary_gumbel_log_density(
            scores, old_logits
        )
        shifted_ratio = auxiliary_gumbel_log_density(
            scores + 2.0, new_logits
        ) - auxiliary_gumbel_log_density(scores + 2.0, old_logits)
        self.assertNotAlmostEqual(float(ratio), float(shifted_ratio), places=6)
        torch.testing.assert_close(
            torch.softmax(scores / 0.7, dim=-1),
            torch.softmax((scores + 2.0) / 0.7, dim=-1),
        )

    def test_inspected_surrogate_crossed_margin_keeps_misaligned_sign(self):
        current = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
        rollout = torch.tensor([-0.5], dtype=torch.float64)
        advantage = torch.tensor([-1.0], dtype=torch.float64)
        surrogate = latent_grpo_surrogate_log_prob(current, rollout, advantage)
        score_gradient = torch.autograd.grad(surrogate.sum(), current)[0]
        objective_gradient = advantage * score_gradient
        self.assertLess(float(score_gradient), 0.0)
        self.assertGreater(float(objective_gradient), 0.0)

    def test_ppo_clip_mask_is_advantage_aware(self):
        log_ratio = torch.log(torch.tensor([1.3, 0.7, 1.3, 0.7]))
        advantages = torch.tensor([1.0, -1.0, -1.0, 1.0])
        result = ppo_clip_mask(log_ratio, advantages, clip_epsilon=0.2)
        self.assertEqual(result.tolist(), [True, True, False, False])


if __name__ == "__main__":
    unittest.main()
