import math
import unittest

from research.policy_conditional_mixture_closure.sequence_gate import (
    categorical_kl,
    first_token_marginal,
    run_sequence_closure_gate,
    total_variation,
)


class PolicyConditionalMixtureClosureGateTest(unittest.TestCase):
    def test_one_step_match_does_not_imply_sequence_match(self):
        result = run_sequence_closure_gate()
        self.assertAlmostEqual(result.initial_one_step_kl_nats, 0.0)
        self.assertAlmostEqual(result.static_student_sequence_tv, 0.5)
        self.assertAlmostEqual(result.static_student_cross_branch_mass, 0.5)
        self.assertFalse(result.one_step_closure_implies_sequence_closure)

    def test_posterior_updated_conditionals_recover_teacher(self):
        result = run_sequence_closure_gate()
        self.assertAlmostEqual(result.posterior_updated_student_sequence_tv, 0.0)

    def test_first_token_marginal(self):
        marginal = first_token_marginal({"00": 0.5, "11": 0.5})
        self.assertEqual(marginal, {"0": 0.5, "1": 0.5})

    def test_distribution_metrics(self):
        left = {"a": 0.5, "b": 0.5}
        right = {"a": 0.25, "b": 0.75}
        self.assertAlmostEqual(total_variation(left, right), 0.25)
        self.assertGreater(categorical_kl(left, right), 0.0)
        self.assertTrue(math.isinf(categorical_kl({"a": 1.0}, {"b": 1.0})))

    def test_invalid_distributions_fail_closed(self):
        with self.assertRaises(ValueError):
            total_variation({}, {"a": 1.0})
        with self.assertRaises(ValueError):
            categorical_kl({"a": 0.4}, {"a": 1.0})
        with self.assertRaises(ValueError):
            first_token_marginal({"": 1.0})


if __name__ == "__main__":
    unittest.main()

