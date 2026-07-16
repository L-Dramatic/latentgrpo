import unittest

import torch

from research.behavioral_geometry.p1_source_sampler_contract import (
    SourceLatentSamplerConfig,
    source_style_latent_action,
)


class SourceSamplerContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.logits = torch.tensor([3.2, 2.4, 1.7, 0.8, -0.3, -1.1], dtype=torch.float64)
        self.embedding_table = torch.arange(24, dtype=torch.float64).reshape(6, 4) / 10.0

    def test_narrow_top_p_still_admits_raw_top_k_and_no_other_token(self):
        result = source_style_latent_action(
            self.logits,
            self.embedding_table,
            SourceLatentSamplerConfig(max_topk=3, top_p=0.01, latent_end_token_id=5),
            generator=torch.Generator().manual_seed(11),
        )

        expected_raw_ids = torch.topk(self.logits.float(), k=3).indices
        self.assertTrue(torch.all(result.top_p_mask[expected_raw_ids]))
        self.assertEqual(int(result.top_p_mask.sum().item()), 3)
        self.assertTrue(torch.all(torch.isin(result.mixture_token_ids, expected_raw_ids)))

    def test_gumbel_scores_and_mixture_match_source_formula_under_fixed_seed(self):
        config = SourceLatentSamplerConfig(
            max_topk=3,
            top_p=0.71,
            gumbel_softmax_temperature=0.83,
            noise_scale=0.7,
            use_one_sided_gumbel_noise=True,
            latent_end_token_id=5,
        )
        seed = 314159
        result = source_style_latent_action(
            self.logits,
            self.embedding_table,
            config,
            generator=torch.Generator().manual_seed(seed),
        )

        logits = self.logits.float()
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        sorted_mask = (torch.cumsum(sorted_probs, dim=-1) - sorted_probs) < config.top_p
        sorted_mask[: config.max_topk] = True
        mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(0, sorted_indices, sorted_mask)
        expected_noise = -torch.empty_like(logits).exponential_(
            generator=torch.Generator().manual_seed(seed)
        ).log()
        expected_noise = expected_noise.clamp(-1.5, 3.0) + 1.5
        expected_noise = config.noise_scale * expected_noise
        expected_scores = torch.log_softmax(logits, dim=-1).masked_fill(~mask, float("-inf")) + expected_noise
        expected_scores_topk, expected_ids = torch.topk(expected_scores, k=config.max_topk)
        expected_probs = torch.softmax(
            expected_scores_topk / config.gumbel_softmax_temperature, dim=-1
        )

        self.assertTrue(result.used_noisy_branch)
        torch.testing.assert_close(result.gumbel_noise, expected_noise)
        torch.testing.assert_close(result.selection_scores, expected_scores)
        torch.testing.assert_close(result.mixture_token_ids, expected_ids)
        torch.testing.assert_close(result.mixture_probs, expected_probs)
        torch.testing.assert_close(
            result.proposed_embedding,
            expected_probs.to(torch.float64) @ self.embedding_table[expected_ids],
        )

    def test_raw_524_top_one_forces_raw_top_k_even_if_gumbel_proxy_would_differ(self):
        logits = torch.full((525,), -12.0)
        logits[524] = 4.0
        logits[2] = 3.9
        logits[8] = 3.8
        embedding_table = torch.arange(525 * 2, dtype=torch.float32).reshape(525, 2)
        result = source_style_latent_action(
            logits,
            embedding_table,
            SourceLatentSamplerConfig(max_topk=3, latent_end_token_id=524),
            generator=torch.Generator().manual_seed(7),
        )

        self.assertFalse(result.used_noisy_branch)
        self.assertEqual(result.proxy, 524)
        torch.testing.assert_close(result.mixture_token_ids, result.raw_topk_token_ids)
        torch.testing.assert_close(result.mixture_probs, result.raw_topk_probs)

    def test_nonlatent_mode_and_explicit_noise_off_both_use_raw_top_k(self):
        for config in (
            SourceLatentSamplerConfig(max_topk=3, latent_mode=False, latent_end_token_id=5),
            SourceLatentSamplerConfig(
                max_topk=3, add_noise_gumbel_softmax=False, latent_end_token_id=5
            ),
        ):
            with self.subTest(config=config):
                result = source_style_latent_action(
                    self.logits,
                    self.embedding_table,
                    config,
                    generator=torch.Generator().manual_seed(9),
                )
                self.assertFalse(result.used_noisy_branch)
                torch.testing.assert_close(result.mixture_token_ids, result.raw_topk_token_ids)
                torch.testing.assert_close(result.mixture_probs, result.raw_topk_probs)

    def test_temperature_is_audit_distribution_not_the_raw_fallback_mixture_law(self):
        raw_config = SourceLatentSamplerConfig(
            max_topk=3, add_noise_gumbel_softmax=False, temperature=1.0, latent_end_token_id=5
        )
        cooled_config = SourceLatentSamplerConfig(
            max_topk=3, add_noise_gumbel_softmax=False, temperature=0.2, latent_end_token_id=5
        )
        raw = source_style_latent_action(self.logits, self.embedding_table, raw_config)
        cooled = source_style_latent_action(self.logits, self.embedding_table, cooled_config)

        torch.testing.assert_close(raw.mixture_token_ids, cooled.mixture_token_ids)
        torch.testing.assert_close(raw.mixture_probs, cooled.mixture_probs)
        self.assertFalse(
            torch.allclose(raw.original_temperature_topk_probs, cooled.original_temperature_topk_probs)
        )


if __name__ == "__main__":
    unittest.main()
