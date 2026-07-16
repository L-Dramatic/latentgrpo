import subprocess
import unittest
from pathlib import Path

import torch

from research.behavioral_geometry.p1_official_sampler_replay import (
    OFFICIAL_SAMPLER_PATH,
    PINNED_OFFICIAL_SOURCE_COMMIT,
    replay_pinned_sampler_on_fake_request,
)
from research.behavioral_geometry.p1_source_sampler_contract import (
    SourceLatentSamplerConfig,
    source_style_latent_action,
)


class OfficialSamplerReplayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.logits = torch.tensor([3.2, 2.4, 1.7, 0.8, -0.3, -1.1], dtype=torch.float64)
        self.embedding_table = torch.arange(24, dtype=torch.float64).reshape(6, 4) / 10.0

    def test_replay_targets_the_pinned_official_source_checkout(self):
        official_root = OFFICIAL_SAMPLER_PATH.parents[4]
        observed = subprocess.check_output(
            ["git", "-C", str(official_root), "rev-parse", "HEAD"], text=True
        ).strip()
        self.assertEqual(observed, PINNED_OFFICIAL_SOURCE_COMMIT)

    def test_noisy_fake_request_replays_unmodified_source_sampler(self):
        config = SourceLatentSamplerConfig(
            max_topk=3,
            top_p=0.71,
            temperature=0.9,
            gumbel_softmax_temperature=0.83,
            noise_scale=0.7,
            use_one_sided_gumbel_noise=True,
            latent_end_token_id=5,
        )
        seed = 808
        torch.manual_seed(seed)
        source = replay_pinned_sampler_on_fake_request(self.logits, config)
        contract = source_style_latent_action(
            self.logits,
            self.embedding_table,
            config,
            generator=torch.Generator().manual_seed(seed),
        )

        self.assertTrue(contract.used_noisy_branch)
        self.assertEqual(source.next_token_id, contract.proxy)
        torch.testing.assert_close(source.topk_indices, contract.mixture_token_ids)
        torch.testing.assert_close(source.topk_probs, contract.mixture_probs)
        torch.testing.assert_close(source.topk_original_indices, contract.raw_topk_token_ids)
        torch.testing.assert_close(
            source.topk_original_probs, contract.original_temperature_topk_probs
        )
        expected_scores = torch.topk(contract.selection_scores, k=config.max_topk).values
        torch.testing.assert_close(source.topk_gumbels, expected_scores)

    def test_raw_524_fallback_and_nonlatent_fallback_replay_official_source(self):
        cases = []
        end_logits = torch.full((525,), -12.0)
        end_logits[524] = 4.0
        end_logits[2] = 3.9
        end_logits[8] = 3.8
        cases.append(
            (end_logits, SourceLatentSamplerConfig(max_topk=3, latent_end_token_id=524))
        )
        cases.append(
            (
                self.logits,
                SourceLatentSamplerConfig(max_topk=3, latent_mode=False, latent_end_token_id=5),
            )
        )

        for logits, config in cases:
            with self.subTest(config=config):
                torch.manual_seed(77)
                source = replay_pinned_sampler_on_fake_request(logits, config)
                table = torch.zeros((logits.numel(), 1), dtype=torch.float32)
                contract = source_style_latent_action(
                    logits,
                    table,
                    config,
                    generator=torch.Generator().manual_seed(77),
                )
                self.assertFalse(contract.used_noisy_branch)
                self.assertEqual(source.next_token_id, contract.proxy)
                torch.testing.assert_close(source.topk_indices, contract.raw_topk_token_ids)
                torch.testing.assert_close(source.topk_probs, contract.raw_topk_probs)

    def test_official_noisy_fallback_is_literal_524_not_a_general_configured_end_id(self):
        # This is a source-audit guard, not a legal platform configuration.
        # sampler.py tests ``raw_top1 != 524`` literally, whereas the scheduler
        # later consults sampling_params.latent_end_token_id.  A generic adapter
        # must reject this mismatch instead of treating id 5 as source-equivalent.
        logits = torch.full((525,), -12.0)
        logits[5] = 4.0
        logits[2] = 3.9
        logits[8] = 3.8
        config = SourceLatentSamplerConfig(max_topk=3, latent_end_token_id=5)
        torch.manual_seed(2718)
        source = replay_pinned_sampler_on_fake_request(logits, config)
        table = torch.zeros((525, 1), dtype=torch.float32)
        generalized_contract = source_style_latent_action(
            logits,
            table,
            config,
            generator=torch.Generator().manual_seed(2718),
        )

        self.assertFalse(generalized_contract.used_noisy_branch)
        self.assertFalse(torch.allclose(source.topk_probs, generalized_contract.raw_topk_probs))


if __name__ == "__main__":
    unittest.main()
