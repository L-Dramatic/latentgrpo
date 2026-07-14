import json
import random
import tempfile
import unittest
from pathlib import Path

import torch

from research.coordinate_invariance import (
    ComputeLedger,
    ContinuationEvaluator,
    ContinuationOutput,
    LatentStepRecord,
    LatentTrace,
    categorical_divergence_from_logits,
    categorical_kl_from_logits,
    multi_horizon_divergence,
    stable_categorical_kl_from_logits,
)


def toy_rollout(context, latent, *, horizon, generator):
    matrices = context["matrices"][:horizon]
    bias = context["bias"][:horizon]
    logits = torch.einsum("d,hdv->hv", latent, matrices) + bias
    shared_sampling_noise = 0.02 * torch.randn(
        logits.shape,
        dtype=logits.dtype,
        device=logits.device,
        generator=generator,
    )
    logits = logits + shared_sampling_noise
    reward = -float(torch.linalg.vector_norm(latent - context["target"]))
    return ContinuationOutput(
        logits=logits,
        reward=reward,
        token_ids=torch.argmax(logits, dim=-1),
        prompt_tokens=7,
        generated_tokens=horizon,
        model_forward_calls=horizon,
    )


class FunctionalMetricTest(unittest.TestCase):
    def setUp(self):
        self.reference = torch.tensor(
            [[2.0, 0.0, -1.0], [0.2, 0.5, -0.7]], dtype=torch.float64
        )

    def test_identical_logits_have_zero_divergence(self):
        for divergence in ("kl", "reverse_kl", "symmetric_kl", "js"):
            value = categorical_divergence_from_logits(
                self.reference, self.reference, divergence=divergence
            )
            torch.testing.assert_close(
                value,
                torch.zeros_like(value),
                atol=1e-15,
                rtol=0.0,
            )

    def test_kl_is_positive_for_different_distributions(self):
        candidate = self.reference.clone()
        candidate[0] = torch.tensor([-1.0, 0.0, 2.0], dtype=torch.float64)

        value = categorical_kl_from_logits(self.reference, candidate)

        self.assertGreater(value[0].item(), 0.0)
        self.assertAlmostEqual(value[1].item(), 0.0, places=14)

    def test_stable_kl_resolves_tiny_logit_differences(self):
        reference = torch.linspace(-2.0, 2.0, 1000, dtype=torch.float64)
        perturbation = torch.linspace(-1e-8, 1e-8, 1000, dtype=torch.float64)

        value = stable_categorical_kl_from_logits(
            reference.unsqueeze(0), (reference + perturbation).unsqueeze(0)
        )

        self.assertGreater(float(value), 0.0)

    def test_stable_kl_matches_standard_kl_at_ordinary_scale(self):
        reference = torch.tensor([[0.2, -0.7, 1.1]], dtype=torch.float64)
        candidate = torch.tensor([[-0.1, 0.4, 0.6]], dtype=torch.float64)

        stable = stable_categorical_kl_from_logits(reference, candidate)
        standard = categorical_kl_from_logits(reference, candidate)

        torch.testing.assert_close(stable, standard, atol=1e-14, rtol=1e-12)

    def test_horizon_weights_select_the_intended_step(self):
        candidate = self.reference.clone()
        candidate[1] = torch.tensor([-1.0, 0.0, 2.0], dtype=torch.float64)

        first_only = multi_horizon_divergence(
            self.reference, candidate, horizon_weights=[1.0, 0.0]
        )
        second_only = multi_horizon_divergence(
            self.reference, candidate, horizon_weights=[0.0, 1.0]
        )

        self.assertAlmostEqual(first_only.item(), 0.0, places=14)
        self.assertGreater(second_only.item(), 0.0)


class MatchedContinuationEvaluatorTest(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator(device="cpu").manual_seed(11)
        self.context = {
            "matrices": torch.randn(4, 3, 5, generator=generator, dtype=torch.float64),
            "bias": torch.randn(4, 5, generator=generator, dtype=torch.float64),
            "target": torch.tensor([0.5, -0.25, 0.1], dtype=torch.float64),
        }
        self.latent = torch.tensor([0.2, -0.3, 0.7], dtype=torch.float64)

    def test_matched_rng_makes_identical_latents_identical(self):
        evaluator = ContinuationEvaluator(toy_rollout)

        comparison = evaluator.compare(
            self.context,
            self.latent,
            self.latent.clone(),
            horizon=4,
            seeds=[3, 5, 7],
        )

        torch.testing.assert_close(
            comparison.per_seed_divergence,
            torch.zeros(3, dtype=torch.float64),
            atol=1e-15,
            rtol=0.0,
        )
        self.assertEqual(comparison.mean_reward_delta, 0.0)
        self.assertEqual(comparison.compute.rollout_calls, 6)
        self.assertEqual(comparison.compute.generated_tokens, 24)
        self.assertEqual(comparison.compute.model_forward_calls, 24)

    def test_different_latents_have_functional_and_reward_difference(self):
        evaluator = ContinuationEvaluator(toy_rollout)
        candidate = self.latent + torch.tensor([0.4, 0.0, -0.2], dtype=torch.float64)

        comparison = evaluator.compare(
            self.context,
            self.latent,
            candidate,
            horizon=4,
            seeds=[13, 17],
        )

        self.assertGreater(comparison.mean_divergence, 0.0)
        self.assertIsNotNone(comparison.mean_reward_delta)
        self.assertEqual(comparison.per_seed_reward_delta.shape, torch.Size([2]))

    def test_reference_evaluation_can_be_reused_across_candidates(self):
        evaluator = ContinuationEvaluator(toy_rollout)
        seeds = [41, 43]
        reference = evaluator.evaluate(
            self.context, self.latent, horizon=3, seeds=seeds
        )
        first_candidate = evaluator.evaluate(
            self.context, self.latent + 0.1, horizon=3, seeds=seeds
        )
        second_candidate = evaluator.evaluate(
            self.context, self.latent - 0.1, horizon=3, seeds=seeds
        )

        first = evaluator.compare_evaluations(reference, first_candidate)
        second = evaluator.compare_evaluations(reference, second_candidate)

        self.assertGreater(first.mean_divergence, 0.0)
        self.assertGreater(second.mean_divergence, 0.0)
        self.assertEqual(second.compute.rollout_calls, 6)

    def test_evaluation_does_not_leak_global_rng_state(self):
        torch.manual_seed(101)
        random.seed(101)
        torch_state_before = torch.random.get_rng_state().clone()
        python_state_before = random.getstate()
        evaluator = ContinuationEvaluator(toy_rollout)

        evaluator.compare(
            self.context,
            self.latent,
            self.latent,
            horizon=2,
            seeds=[19],
        )

        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state_before))
        self.assertEqual(random.getstate(), python_state_before)

    def test_captured_record_can_be_replayed(self):
        record = LatentStepRecord.capture(
            sample_id="sample-1",
            step_index=2,
            prefix_state=self.context,
            latent=self.latent,
            rng_seed=23,
        )
        evaluator = ContinuationEvaluator(toy_rollout)

        comparison = evaluator.compare_record(
            record, self.latent.clone(), horizon=3
        )

        self.assertEqual(comparison.seeds, (23,))
        self.assertEqual(comparison.mean_divergence, 0.0)


class TracePersistenceTest(unittest.TestCase):
    def test_capture_clones_tensors_and_rejects_duplicate_steps(self):
        prefix = {"cache": [torch.tensor([1.0, 2.0])], "label": "q"}
        latent = torch.tensor([0.1, 0.2], dtype=torch.float64)
        record = LatentStepRecord.capture(
            sample_id="q-1",
            step_index=0,
            prefix_state=prefix,
            latent=latent,
            rng_seed=31,
            metadata={"task": "toy"},
        )
        trace = LatentTrace([record])
        prefix["cache"][0].add_(10.0)
        latent.add_(10.0)

        torch.testing.assert_close(
            record.prefix_state["cache"][0], torch.tensor([1.0, 2.0])
        )
        torch.testing.assert_close(
            record.latent, torch.tensor([0.1, 0.2], dtype=torch.float64)
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            trace.append(record)

    def test_trace_round_trip_and_integrity_check(self):
        record = LatentStepRecord.capture(
            sample_id="q-2",
            step_index=1,
            prefix_state={"hidden": torch.arange(4, dtype=torch.float32)},
            latent=torch.tensor([0.3, -0.4], dtype=torch.float32),
            rng_seed=37,
            chart_name="identity",
            metadata={"split": "audit", "accepted": True},
        )
        trace = LatentTrace([record])

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = trace.save(temporary_directory)
            loaded = LatentTrace.load(path)

            self.assertEqual(len(loaded), 1)
            loaded_record = loaded.get("q-2", 1)
            torch.testing.assert_close(loaded_record.latent, record.latent)
            self.assertEqual(dict(loaded_record.metadata), dict(record.metadata))

            manifest = json.loads(
                (Path(path) / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["record_count"], 1)

            with (Path(path) / "trace.pt").open("ab") as handle:
                handle.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "integrity"):
                LatentTrace.load(path)


class ComputeLedgerTest(unittest.TestCase):
    def test_snapshot_is_detached_from_mutable_ledger(self):
        ledger = ComputeLedger()
        ledger.record_rollout(prompt_tokens=3, generated_tokens=5, model_forward_calls=2)
        ledger.increment("chart_evaluations", 4)
        snapshot = ledger.snapshot()
        ledger.increment("chart_evaluations", 1)

        self.assertEqual(snapshot.rollout_calls, 1)
        self.assertEqual(snapshot.prompt_tokens, 3)
        self.assertEqual(snapshot.custom_counts["chart_evaluations"], 4)


if __name__ == "__main__":
    unittest.main()
