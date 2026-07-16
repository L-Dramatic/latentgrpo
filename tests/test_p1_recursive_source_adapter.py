import unittest

import torch

from research.behavioral_geometry.p1_fake_preflight import (
    FinishKind,
    LatentActionKind,
    StopConfig,
    directional_jvp,
)
from research.behavioral_geometry.p1_recursive_source_adapter import (
    SOURCE_LITERAL_LATENT_END_ID,
    IsolatedSamplerRNG,
    SourceAdapterState,
    SourceClosureEndpoint,
    SourceRecursiveAdapter,
    ToySourceCacheModel,
    execute_source_scheduled_action,
    propose_source_action,
    source_weighted_embedding,
)
from research.behavioral_geometry.p1_source_sampler_contract import SourceLatentSamplerConfig


class RecursiveSourceAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = ToySourceCacheModel()
        self.noisy = SourceLatentSamplerConfig(
            max_topk=3,
            top_p=0.8,
            add_noise_gumbel_softmax=True,
            latent_end_token_id=SOURCE_LITERAL_LATENT_END_ID,
        )
        self.deterministic = SourceLatentSamplerConfig(
            max_topk=3,
            top_p=1.0,
            add_noise_gumbel_softmax=False,
            latent_end_token_id=SOURCE_LITERAL_LATENT_END_ID,
        )
        self.stop = StopConfig(max_new_tokens=16)
        self.adapter = SourceRecursiveAdapter(
            self.model,
            deterministic_config=self.deterministic,
            stop_config=self.stop,
        )
        self.initial_cache = torch.tensor([0.2, -0.4, 0.1], dtype=torch.float64)

    def _state(self) -> SourceAdapterState:
        return SourceAdapterState((), "", True, self.initial_cache.clone())

    def test_literal_524_is_required_to_avoid_hiding_sampler_scheduler_mismatch(self):
        mismatched = SourceLatentSamplerConfig(
            max_topk=3, add_noise_gumbel_softmax=False, latent_end_token_id=5
        )
        with self.assertRaisesRegex(ValueError, "literal latent_end_token_id=524"):
            SourceRecursiveAdapter(
                self.model, deterministic_config=mismatched, stop_config=self.stop
            )

    def test_exit_replaces_soft_mixture_with_hard_524_and_advances_own_cache(self):
        end_logits = torch.full((525,), -12.0, dtype=torch.float64)
        end_logits[524] = 4.0
        end_logits[2] = 3.9
        end_logits[8] = 3.8
        proposal = propose_source_action(
            end_logits, self.model.embedding_table, self.noisy, IsolatedSamplerRNG(8)
        )
        execution = execute_source_scheduled_action(
            self._state(), proposal, embedding_table=self.model.embedding_table, stop_config=self.stop
        )
        closure = self.adapter.close(
            self._state(), proposal, max_latent_steps=5, rng=IsolatedSamplerRNG(99)
        )

        self.assertEqual(execution.kind, LatentActionKind.EXIT_VISIBLE)
        self.assertFalse(execution.state.latent_mode)
        self.assertEqual(int(execution.executed_topk_indices[0].item()), 524)
        self.assertTrue(torch.all(execution.executed_topk_indices[1:] == -100))
        self.assertEqual(float(execution.executed_topk_probs[0].item()), 1.0)
        torch.testing.assert_close(execution.consumed_embedding, self.model.embedding_table[524])
        self.assertFalse(torch.equal(proposal.proposed_embedding, execution.consumed_embedding))
        self.assertEqual(closure.endpoint, SourceClosureEndpoint.EXIT_VISIBLE.value)
        self.assertEqual(closure.consumed_actions, 1)
        self.assertFalse(torch.equal(closure.state.cache, self.initial_cache))

    def test_generic_stop_preempts_hard_end_and_does_not_advance_cache(self):
        end_logits = torch.full((525,), -12.0, dtype=torch.float64)
        end_logits[524] = 4.0
        proposal = propose_source_action(
            end_logits, self.model.embedding_table, self.noisy, IsolatedSamplerRNG(9)
        )
        blocked_stop = StopConfig(max_new_tokens=16, stop_token_ids=frozenset({524}))
        closure = SourceRecursiveAdapter(
            self.model, deterministic_config=self.deterministic, stop_config=blocked_stop
        ).close(self._state(), proposal, max_latent_steps=5, rng=IsolatedSamplerRNG(1))

        self.assertEqual(closure.endpoint, FinishKind.TOKEN.value)
        self.assertEqual(closure.consumed_actions, 0)
        self.assertTrue(torch.equal(closure.state.cache, self.initial_cache))
        self.assertIsNone(closure.records[0].execution.consumed_embedding)

    def test_forced_candidate_recomputes_its_own_later_source_actions_without_teacher_forcing(self):
        initial = propose_source_action(
            self.model.logits(self.initial_cache),
            self.model.embedding_table,
            self.deterministic,
            IsolatedSamplerRNG(10),
        )
        reference = self.adapter.close(
            self._state(), initial, max_latent_steps=5, rng=IsolatedSamplerRNG(11)
        )
        forced = initial.with_forced_embedding(
            initial.proposed_embedding + torch.tensor([0.11, -0.07, 0.05], dtype=torch.float64)
        )
        candidate = self.adapter.close(
            self._state(), forced, max_latent_steps=5, rng=IsolatedSamplerRNG(12)
        )

        self.assertEqual(reference.endpoint, SourceClosureEndpoint.TIMEOUT.value)
        self.assertEqual(candidate.endpoint, SourceClosureEndpoint.TIMEOUT.value)
        self.assertEqual(candidate.consumed_actions, 5)
        self.assertTrue(candidate.records[0].execution.proposal.externally_forced_embedding)
        self.assertFalse(
            torch.allclose(
                reference.records[1].execution.proposal.topk_probs,
                candidate.records[1].execution.proposal.topk_probs,
            )
        )
        self.assertFalse(torch.equal(reference.state.cache, candidate.state.cache))

    def test_isolated_source_rng_replays_per_candidate_without_advancing_global_rng(self):
        logits = self.model.logits(self.initial_cache)
        first = propose_source_action(logits, self.model.embedding_table, self.noisy, IsolatedSamplerRNG(123))
        second = propose_source_action(logits, self.model.embedding_table, self.noisy, IsolatedSamplerRNG(123))
        torch.testing.assert_close(first.topk_indices, second.topk_indices)
        torch.testing.assert_close(first.topk_probs, second.topk_probs)

        torch.manual_seed(501)
        expected_next = torch.rand(1)
        torch.manual_seed(501)
        propose_source_action(logits, self.model.embedding_table, self.noisy, IsolatedSamplerRNG(124))
        torch.testing.assert_close(torch.rand(1), expected_next)

    def test_jvp_matches_finite_difference_through_four_recomputed_source_actions(self):
        initial = propose_source_action(
            self.model.logits(self.initial_cache),
            self.model.embedding_table,
            self.deterministic,
            IsolatedSamplerRNG(14),
        )
        direction = torch.tensor([0.13, -0.19, 0.08], dtype=torch.float64)

        def final_cache(forced_embedding: torch.Tensor) -> torch.Tensor:
            closure = self.adapter.close(
                self._state(),
                initial.with_forced_embedding(forced_embedding),
                max_latent_steps=5,
                rng=IsolatedSamplerRNG(15),
            )
            self.assertEqual(closure.endpoint, SourceClosureEndpoint.TIMEOUT.value)
            self.assertEqual(closure.consumed_actions, 5)
            return closure.state.cache

        jvp = directional_jvp(final_cache, initial.proposed_embedding, direction)
        # The pinned sampler explicitly casts logits to float32.  A 1e-5
        # perturbation is therefore below a stable central-difference scale
        # after repeated top-k/softmax operations; 1e-2 is the predeclared
        # float32-safe local step for this fake-source contract.
        epsilon = 1e-2
        finite_difference = (
            final_cache(initial.proposed_embedding + epsilon * direction)
            - final_cache(initial.proposed_embedding - epsilon * direction)
        ) / (2.0 * epsilon)
        torch.testing.assert_close(jvp, finite_difference, atol=1e-6, rtol=1e-3)

    def test_hard_sentinel_embedding_ignores_nonfirst_weighted_rows(self):
        ids = torch.tensor([524, -100, -100])
        probs = torch.tensor([0.1, 0.4, 0.5], dtype=torch.float64)
        embedding = source_weighted_embedding(self.model.embedding_table, probs, ids)
        torch.testing.assert_close(embedding, self.model.embedding_table[524])


if __name__ == "__main__":
    unittest.main()
