import unittest

import torch

from research.behavioral_geometry.p1_fake_preflight import (
    FinishKind,
    LatentActionKind,
    LatentRequestState,
    StopConfig,
    ToyRecursiveLatentPolicy,
    directional_jvp,
    execute_latent_action,
)


def _embedding_table() -> torch.Tensor:
    return torch.arange(18, dtype=torch.float64).reshape(6, 3) / 10.0


class LatentStopOrderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.embedding_table = _embedding_table()
        self.embedding = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64)

    def _execute(self, config: StopConfig, proxy: int, **state_kwargs):
        return execute_latent_action(
            LatentRequestState(**state_kwargs),
            self.embedding,
            proxy,
            latent_end_token_id=5,
            embedding_table=self.embedding_table,
            stop_config=config,
        )

    def test_every_token_stop_source_preempts_latent_end_embedding(self):
        configs = {
            "request stop id": StopConfig(max_new_tokens=8, stop_token_ids=frozenset({5})),
            "request eos id": StopConfig(max_new_tokens=8, eos_token_ids=frozenset({5})),
            "tokenizer eos id": StopConfig(max_new_tokens=8, tokenizer_eos_token_id=5),
            "additional tokenizer stop id": StopConfig(
                max_new_tokens=8, additional_stop_token_ids=frozenset({5})
            ),
        }
        for name, config in configs.items():
            with self.subTest(name=name):
                result = self._execute(config, 5)
                self.assertEqual(result.kind, LatentActionKind.STOP)
                self.assertEqual(result.finish_event.kind, FinishKind.TOKEN)
                self.assertIsNone(result.consumed_embedding)
                self.assertTrue(result.state.latent_mode)

    def test_ignore_eos_disables_token_stops_but_not_string_stops(self):
        result = self._execute(
            StopConfig(
                max_new_tokens=8,
                ignore_eos=True,
                eos_token_ids=frozenset({5}),
                additional_stop_token_ids=frozenset({5}),
            ),
            5,
        )

        self.assertEqual(result.kind, LatentActionKind.EXIT_VISIBLE)
        self.assertFalse(result.state.latent_mode)

    def test_length_stop_preempts_latent_end_embedding(self):
        result = self._execute(StopConfig(max_new_tokens=1), 5)

        self.assertEqual(result.kind, LatentActionKind.STOP)
        self.assertEqual(result.finish_event.kind, FinishKind.LENGTH)
        self.assertIsNone(result.consumed_embedding)

    def test_stop_string_preempts_latent_end_even_when_eos_is_ignored(self):
        config = StopConfig(
            max_new_tokens=8,
            ignore_eos=True,
            stop_strings=("<END>",),
            stop_str_max_len=3,
            decode=lambda ids: "".join({5: "<END>"}.get(token_id, "x") for token_id in ids),
        )
        result = self._execute(config, 5)

        self.assertEqual(result.kind, LatentActionKind.STOP)
        self.assertEqual(result.finish_event.kind, FinishKind.STRING)
        self.assertEqual(result.finish_event.detail, "<END>")

    def test_nonstopping_latent_end_uses_hard_embedding_and_enters_visible_mode(self):
        result = self._execute(StopConfig(max_new_tokens=8), 5)

        self.assertEqual(result.kind, LatentActionKind.EXIT_VISIBLE)
        self.assertFalse(result.state.latent_mode)
        torch.testing.assert_close(result.consumed_embedding, self.embedding_table[5])
        self.assertFalse(torch.equal(result.consumed_embedding, self.embedding))

    def test_nonterminal_proxy_consumes_the_proposed_soft_embedding(self):
        result = self._execute(StopConfig(max_new_tokens=8), 2)

        self.assertEqual(result.kind, LatentActionKind.CONSUME_LATENT)
        self.assertTrue(result.state.latent_mode)
        self.assertIs(result.consumed_embedding, self.embedding)

    def test_abort_is_a_hard_structural_failure(self):
        result = self._execute(
            StopConfig(max_new_tokens=8), 2, to_abort=True, abort_message="cancelled"
        )

        self.assertEqual(result.kind, LatentActionKind.STOP)
        self.assertEqual(result.finish_event.kind, FinishKind.ABORT)


class RecursiveClosureAndJVPTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = ToyRecursiveLatentPolicy()
        self.hidden = torch.tensor([0.2, -0.4, 0.1], dtype=torch.float64)
        self.reference_embedding, self.reference_proxy = self.policy.deterministic_action(
            self.hidden
        )
        self.stop_config = StopConfig(max_new_tokens=12)

    def _closure(self, embedding: torch.Tensor):
        return self.policy.close(
            initial_hidden=self.hidden,
            initial_embedding=embedding,
            initial_proxy=self.reference_proxy,
            max_latent_steps=5,
            stop_config=self.stop_config,
        )

    def test_candidate_closure_recomputes_at_least_four_later_actions(self):
        candidate_embedding = self.reference_embedding + torch.tensor(
            [0.08, -0.03, 0.04], dtype=torch.float64
        )
        reference = self._closure(self.reference_embedding)
        candidate = self._closure(candidate_embedding)

        self.assertEqual(reference.terminal_kind, "LATENT_TIMEOUT")
        self.assertEqual(candidate.terminal_kind, "LATENT_TIMEOUT")
        self.assertEqual(reference.consumed_latent_actions, 5)
        self.assertEqual(candidate.consumed_latent_actions, 5)
        self.assertEqual(len(candidate.records) - 1, 4)
        self.assertFalse(
            torch.allclose(
                reference.records[1].proposed_embedding,
                candidate.records[1].proposed_embedding,
            )
        )

    def test_directional_jvp_matches_finite_difference_through_recursive_suffix(self):
        direction = torch.tensor([0.15, -0.20, 0.10], dtype=torch.float64)

        def final_hidden(embedding: torch.Tensor) -> torch.Tensor:
            closure = self._closure(embedding)
            self.assertEqual(closure.terminal_kind, "LATENT_TIMEOUT")
            self.assertGreaterEqual(closure.consumed_latent_actions, 5)
            return closure.hidden

        jvp = directional_jvp(final_hidden, self.reference_embedding, direction)
        epsilon = 1e-5
        finite_difference = (
            final_hidden(self.reference_embedding + epsilon * direction)
            - final_hidden(self.reference_embedding - epsilon * direction)
        ) / (2.0 * epsilon)

        torch.testing.assert_close(jvp, finite_difference, atol=2e-6, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
