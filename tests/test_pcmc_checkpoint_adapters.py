import copy
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from research.policy_conditional_mixture_closure.checkpoint_adapters import (
    SourceAction,
    action_fingerprint,
    condition_on_content,
    sample_source_action,
    weighted_embedding,
)
from research.policy_conditional_mixture_closure.checkpoint_gate_runner import (
    _cache_ptrs,
    _clone_cache,
    _jensen_shannon,
    _one_step_closure,
    _repeat_cache,
)
from research.policy_conditional_mixture_closure.checkpoint_protocol import (
    checkpoint_profiles,
    load_protocol,
)
from research.policy_conditional_mixture_closure.soft_official_sampler_replay import (
    replay_pinned_soft_sampler,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "research"
    / "policy_conditional_mixture_closure"
    / "configs"
    / "pcmc_gate_ab_v1.json"
)


def _profile(checkpoint_id):
    protocol = load_protocol(CONFIG)
    return next(
        value
        for value in checkpoint_profiles(protocol)
        if value.checkpoint_id == checkpoint_id
    )


class PcmcCheckpointAdapterTest(unittest.TestCase):
    def test_soft_grpo_sampler_matches_pinned_operation_order(self):
        profile = _profile("soft_grpo_qwen_1_5b")
        logits = torch.tensor([3.0, 2.0, 1.0, 0.5, 0.0, -1.0])

        torch.manual_seed(1701)
        observed = sample_source_action(logits, profile)

        torch.manual_seed(1701)
        probabilities = torch.softmax(logits.float(), dim=-1)
        top_values, top_indices = torch.topk(probabilities, k=5)
        truncated = torch.zeros_like(probabilities)
        truncated.scatter_(0, top_indices, top_values)
        truncated /= truncated.sum()
        sorted_values, sorted_indices = torch.sort(truncated, descending=True)
        keep = (torch.cumsum(sorted_values, dim=0) - sorted_values) < 0.95
        top_p = torch.zeros_like(probabilities)
        top_p.scatter_(0, sorted_indices, sorted_values * keep)
        top_p /= top_p.sum()
        base_weights, base_ids = torch.topk(top_p, k=5)
        base_weights /= base_weights.sum()
        gumbels = -torch.empty_like(base_weights).exponential_().log()
        gumbels = gumbels.clamp(-1.5, 3.0)
        expected_weights = torch.softmax((torch.log(base_weights + 1e-6) + gumbels) / 0.1, dim=-1)
        expected_weights, order = torch.sort(expected_weights, descending=True)
        expected_ids = base_ids[order]

        self.assertTrue(torch.equal(observed.token_ids, expected_ids))
        self.assertTrue(torch.equal(observed.weights, expected_weights))
        self.assertEqual(observed.proxy_token_id, int(expected_ids[0]))

    def test_soft_grpo_adapter_matches_unmodified_official_sampler(self):
        profile = _profile("soft_grpo_qwen_1_5b")
        logits = torch.linspace(-4.0, 4.0, 37)
        torch.manual_seed(90210)
        observed = sample_source_action(logits, profile)
        torch.manual_seed(90210)
        official = replay_pinned_soft_sampler(
            logits,
            top_p=profile.sampler.top_p,
            top_k=profile.sampler.top_k,
            max_topk=profile.sampler.max_topk,
            temperature=profile.sampler.temperature,
            gumbel_softmax_temperature=profile.sampler.gumbel_softmax_temperature,
            noise_scale=profile.sampler.noise_scale,
        )
        self.assertEqual(observed.proxy_token_id, official.next_token_id)
        self.assertTrue(torch.equal(observed.token_ids, official.token_ids))
        self.assertTrue(torch.equal(observed.weights, official.weights))

    def test_content_conditioning_excludes_and_renormalizes_end_mass(self):
        profile = _profile("soft_grpo_qwen_1_5b")
        action = SourceAction(
            token_ids=torch.tensor([151649, 11, 12]),
            weights=torch.tensor([0.005, 0.7, 0.295]),
            proxy_token_id=11,
            sampler_metadata={},
        )
        result = condition_on_content(action, profile, minimum_content_support=2)
        self.assertEqual(result.status, "COMPLETE")
        self.assertAlmostEqual(result.structural_end_mass, 0.005, places=6)
        self.assertEqual(result.token_ids.tolist(), [11, 12])
        self.assertAlmostEqual(float(result.weights.sum()), 1.0, places=6)

    def test_excess_structural_mass_is_ineligible_not_silently_conditioned(self):
        profile = _profile("soft_grpo_qwen_1_5b")
        action = SourceAction(
            token_ids=torch.tensor([151649, 11, 12]),
            weights=torch.tensor([0.02, 0.7, 0.28]),
            proxy_token_id=11,
            sampler_metadata={},
        )
        result = condition_on_content(action, profile, minimum_content_support=2)
        self.assertEqual(result.status, "INELIGIBLE_CONTENT")
        self.assertIn("exceeds", result.reason)

    def test_weighted_embedding_accumulates_with_float_weights(self):
        table = torch.tensor(
            [[0.0, 0.0], [1.0, 2.0], [3.0, 6.0]], dtype=torch.bfloat16
        )
        result = weighted_embedding(
            table, torch.tensor([1, 2]), torch.tensor([0.25, 0.75])
        )
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertTrue(torch.allclose(result.float(), torch.tensor([2.5, 5.0])))

    def test_action_fingerprint_changes_with_weights(self):
        base = SourceAction(
            token_ids=torch.tensor([1, 2]),
            weights=torch.tensor([0.4, 0.6]),
            proxy_token_id=2,
            sampler_metadata={},
        )
        changed = copy.copy(base)
        object.__setattr__(changed, "weights", torch.tensor([0.5, 0.5]))
        self.assertNotEqual(action_fingerprint(base), action_fingerprint(changed))

    def test_cache_clone_and_repeat_are_storage_disjoint(self):
        cache = ((torch.arange(6).reshape(1, 2, 3), torch.ones(1, 2, 3)),)
        cloned = _clone_cache(cache)
        repeated = _repeat_cache(cache, 4)
        self.assertFalse(_cache_ptrs(cache) & _cache_ptrs(cloned))
        self.assertFalse(_cache_ptrs(cache) & _cache_ptrs(repeated))
        self.assertEqual(repeated[0][0].shape[0], 4)

    def test_dynamic_cache_clone_and_repeat_are_storage_disjoint(self):
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache()
        cache.update(
            torch.ones(1, 2, 3, 4),
            torch.full((1, 2, 3, 4), 2.0),
            0,
        )
        cloned = _clone_cache(cache)
        repeated = _repeat_cache(cache, 3)
        self.assertFalse(_cache_ptrs(cache) & _cache_ptrs(cloned))
        self.assertFalse(_cache_ptrs(cache) & _cache_ptrs(repeated))
        self.assertEqual(repeated.layers[0].keys.shape[0], 3)

    def test_one_step_closure_batches_hard_branches(self):
        class FakeModel:
            def __init__(self):
                self.embedding = SimpleNamespace(
                    weight=torch.tensor(
                        [
                            [0.0, 0.0],
                            [1.0, 0.0],
                            [0.0, 1.0],
                            [1.0, 1.0],
                        ]
                    )
                )

            def get_input_embeddings(self):
                return self.embedding

            def __call__(self, *, inputs_embeds, past_key_values, **_kwargs):
                hidden = inputs_embeds[:, -1]
                logits = torch.stack(
                    (
                        hidden[:, 0],
                        hidden[:, 1],
                        hidden[:, 0] * hidden[:, 1],
                        -hidden.sum(dim=-1),
                    ),
                    dim=-1,
                )
                return SimpleNamespace(
                    logits=logits.unsqueeze(1), past_key_values=past_key_values
                )

        cache = ((torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2)),)
        prefill = SimpleNamespace(past_key_values=cache)
        action = SimpleNamespace(
            token_ids=torch.tensor([1, 2]), weights=torch.tensor([0.4, 0.6])
        )
        result = _one_step_closure(
            FakeModel(),
            prefill,
            action,
            position=2,
            distribution_temperature=1.0,
        )
        self.assertGreater(result["js_branch_arithmetic_nats"], 0.0)
        self.assertEqual(len(result["arithmetic_distribution_sha256"]), 64)

    def test_js_divergence_controls(self):
        distribution = torch.tensor([0.2, 0.3, 0.5])
        self.assertAlmostEqual(_jensen_shannon(distribution, distribution), 0.0)
        separated = _jensen_shannon(
            torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
        )
        self.assertAlmostEqual(separated, math.log(2.0), places=6)

    def test_frozen_profiles_include_both_structural_end_tokens(self):
        self.assertEqual(
            _profile("latent_grpo_llama_1b").structural_end_token_id, 524
        )
        self.assertEqual(
            _profile("soft_grpo_qwen_1_5b").structural_end_token_id, 151649
        )


if __name__ == "__main__":
    unittest.main()
