import math
import unittest

import torch

from research.behavioral_geometry.p1_forward_kl_contract import (
    ContinuationEndpoint,
    audit_endpoint_pair,
    paired_path_kl,
)


class ForwardKLContractTest(unittest.TestCase):
    def test_identity_logits_have_exact_zero_path_kl(self):
        logits = torch.tensor(
            [[1.2, -0.3, 0.7], [-0.5, 0.2, 1.1]], dtype=torch.float64
        )
        result = paired_path_kl(logits, logits.clone())
        torch.testing.assert_close(result.per_step, torch.zeros(2, dtype=torch.float64))
        self.assertEqual(result.total, 0.0)

    def test_directional_path_kl_is_asymmetric_and_nonnegative(self):
        left = torch.tensor([[2.0, 0.0, -1.0]], dtype=torch.float64)
        right = torch.tensor([[-0.2, 0.6, 1.3]], dtype=torch.float64)
        forward = paired_path_kl(left, right)
        reverse = paired_path_kl(right, left)
        self.assertGreater(forward.total, 0.0)
        self.assertGreater(reverse.total, 0.0)
        self.assertNotAlmostEqual(forward.total, reverse.total)

    def test_endpoint_mismatch_is_extended_real_infinite(self):
        audit = audit_endpoint_pair(
            ContinuationEndpoint.NATURAL_VISIBLE,
            ContinuationEndpoint.LATENT_TIMEOUT,
        )
        self.assertFalse(audit.same_atom)
        self.assertTrue(math.isinf(audit.endpoint_kl))
        self.assertFalse(audit.visible_continuation_defined)

    def test_matching_timeout_has_no_visible_continuation(self):
        audit = audit_endpoint_pair(
            ContinuationEndpoint.LATENT_TIMEOUT,
            ContinuationEndpoint.LATENT_TIMEOUT,
        )
        self.assertEqual(audit.endpoint_kl, 0.0)
        self.assertFalse(audit.visible_continuation_defined)
        self.assertFalse(audit.scientific_visible_pair)

    def test_forced_boundary_requires_explicit_control_authorization(self):
        forced = ContinuationEndpoint.FORCED_VISIBLE_CONTROL
        with self.assertRaisesRegex(ValueError, "not scientific"):
            audit_endpoint_pair(forced, forced)
        audit = audit_endpoint_pair(forced, forced, allow_forced_control=True)
        self.assertTrue(audit.visible_continuation_defined)
        self.assertFalse(audit.scientific_visible_pair)


if __name__ == "__main__":
    unittest.main()
