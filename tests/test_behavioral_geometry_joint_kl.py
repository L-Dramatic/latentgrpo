import unittest

import torch

from research.behavioral_geometry.joint_kl import JointContinuationEvaluator
from research.coordinate_invariance.evaluator import ContinuationOutput


class TinyAutoregressivePolicy:
    def __init__(self):
        self.base = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64)
        self.latent_map = torch.tensor(
            [[0.7, -0.2], [-0.3, 0.5], [0.1, -0.4]], dtype=torch.float64
        )
        self.history_map = torch.tensor(
            [[0.2, -0.1, 0.0], [0.0, 0.3, -0.2], [-0.1, 0.0, 0.4]],
            dtype=torch.float64,
        )

    def _run(self, latent, horizon, generator, forced=None):
        history = torch.zeros(3, dtype=torch.float64)
        logits = []
        tokens = []
        for step in range(horizon):
            step_logits = self.base + self.latent_map @ latent + self.history_map @ history
            logits.append(step_logits)
            if forced is None:
                token = torch.multinomial(
                    torch.softmax(step_logits, dim=-1), 1, generator=generator
                )[0]
            else:
                token = forced[step]
            tokens.append(token)
            history = history + torch.nn.functional.one_hot(token, 3).to(torch.float64)
        return ContinuationOutput(
            logits=torch.stack(logits),
            token_ids=torch.stack(tokens).to(torch.long),
            generated_tokens=0 if forced is not None else horizon,
            model_forward_calls=horizon,
        )

    def __call__(self, context, latent, *, horizon, generator):
        del context
        return self._run(latent, horizon, generator)

    def force(self, context, latent, *, token_ids, generator):
        del context
        return self._run(latent, int(token_ids.numel()), generator, forced=token_ids)


class EarlyStopPolicy(TinyAutoregressivePolicy):
    def _run(self, latent, horizon, generator, forced=None):
        output = super()._run(latent, min(horizon, 2), generator, forced=forced)
        return ContinuationOutput(
            logits=output.logits,
            token_ids=output.token_ids,
            generated_tokens=output.generated_tokens,
            model_forward_calls=output.model_forward_calls,
            metadata={
                "terminated": True,
                "requested_horizon": horizon,
                "valid_steps": int(output.logits.shape[0]),
            },
        )


class JointContinuationEvaluatorTest(unittest.TestCase):
    def test_identical_latents_have_zero_joint_kl(self):
        evaluator = JointContinuationEvaluator(TinyAutoregressivePolicy())
        latent = torch.tensor([0.2, -0.4], dtype=torch.float64)
        result = evaluator.directional(
            {}, latent, latent.clone(), horizon=5, seeds=(3, 4)
        )
        torch.testing.assert_close(
            result.per_seed_total_kl, torch.zeros(2, dtype=torch.float64), atol=1e-12, rtol=0
        )
        self.assertEqual(result.per_seed_step_kl.shape, (2, 5))
        self.assertEqual(result.compute.rollout_calls, 4)

    def test_chain_rule_uses_identical_histories(self):
        evaluator = JointContinuationEvaluator(TinyAutoregressivePolicy())
        reference = torch.tensor([0.1, -0.2], dtype=torch.float64)
        candidate = torch.tensor([-0.5, 0.3], dtype=torch.float64)
        result = evaluator.directional(
            {}, reference, candidate, horizon=4, seeds=(17,)
        )
        sampled = result.reference_outputs[0]
        forced = result.forced_candidate_outputs[0]
        self.assertEqual(sampled.token_ids.tolist(), forced.token_ids.tolist())
        per_step = torch.nn.functional.kl_div(
            torch.log_softmax(forced.logits, dim=-1),
            torch.softmax(sampled.logits, dim=-1),
            reduction="none",
        ).sum(dim=-1)
        self.assertAlmostEqual(result.mean_total_kl, float(per_step.sum()), places=12)
        self.assertAlmostEqual(
            result.mean_tail_kl,
            result.mean_total_kl - result.mean_first_step_kl,
            places=12,
        )

    def test_symmetric_runs_both_directional_chain_rules(self):
        evaluator = JointContinuationEvaluator(TinyAutoregressivePolicy())
        left = torch.tensor([0.0, 0.2], dtype=torch.float64)
        right = torch.tensor([0.4, -0.1], dtype=torch.float64)
        result = evaluator.symmetric({}, left, right, horizon=3, seeds=(8, 9))
        self.assertGreater(result.mean_total_kl, 0.0)
        self.assertEqual(result.forward.seeds, result.reverse.seeds)

    def test_sampling_temperature_is_applied_to_distribution_logits(self):
        logits = torch.tensor([[1.0, -1.0]], dtype=torch.float64)
        output = ContinuationOutput(
            logits=logits,
            token_ids=torch.tensor([0]),
            metadata={"sampling_temperature": 0.5},
        )

        scaled = JointContinuationEvaluator._distribution_logits(output)

        torch.testing.assert_close(scaled, logits / 0.5)
        self.assertEqual(scaled.dtype, torch.float64)

    def test_deterministic_temperature_is_rejected(self):
        output = ContinuationOutput(
            logits=torch.zeros(1, 2),
            token_ids=torch.tensor([0]),
            metadata={"sampling_temperature": 0.0},
        )

        with self.assertRaisesRegex(ValueError, "positive stochastic"):
            JointContinuationEvaluator._validate_output(
                output, horizon=1, require_tokens=True
            )

    def test_reference_rollouts_can_be_shared_across_candidates(self):
        evaluator = JointContinuationEvaluator(TinyAutoregressivePolicy())
        reference_latent = torch.tensor([0.1, -0.2], dtype=torch.float64)
        reference = evaluator.sample_reference(
            {}, reference_latent, horizon=4, seeds=(21, 22)
        )

        first = evaluator.directional_from_reference(
            {}, torch.tensor([0.3, -0.2], dtype=torch.float64), reference
        )
        second = evaluator.directional_from_reference(
            {}, torch.tensor([-0.4, 0.5], dtype=torch.float64), reference
        )

        self.assertEqual(reference.compute.rollout_calls, 2)
        self.assertEqual(first.compute.rollout_calls, 4)
        self.assertEqual(second.compute.rollout_calls, 6)
        self.assertIs(first.reference_outputs[0], second.reference_outputs[0])

    def test_early_stopped_sequences_are_padded_as_absorbing_eos(self):
        evaluator = JointContinuationEvaluator(EarlyStopPolicy())
        reference = torch.tensor([0.1, -0.2], dtype=torch.float64)
        candidate = torch.tensor([0.4, 0.3], dtype=torch.float64)

        result = evaluator.directional(
            {}, reference, candidate, horizon=5, seeds=(31,)
        )

        self.assertEqual(result.per_seed_valid_steps.tolist(), [2])
        self.assertEqual(result.per_seed_step_kl.shape, (1, 5))
        torch.testing.assert_close(
            result.per_seed_step_kl[0, 2:], torch.zeros(3, dtype=torch.float64)
        )


if __name__ == "__main__":
    unittest.main()
