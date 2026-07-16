import unittest

from research.coupled_group_exploration.bernoulli_gate import (
    antithetic_joint,
    evaluate_coupled_group_gate,
    iid_joint,
)


class CoupledGroupExplorationGateTest(unittest.TestCase):
    def test_joint_laws_are_normalized(self):
        for probability in (0.1, 0.3, 0.5, 0.7, 0.9):
            self.assertAlmostEqual(
                sum(item.probability for item in iid_joint(probability)), 1.0
            )
            self.assertAlmostEqual(
                sum(item.probability for item in antithetic_joint(probability)),
                1.0,
            )

    def test_antithetic_members_preserve_policy_marginals(self):
        for probability in (0.1, 0.3, 0.5, 0.7, 0.9):
            gate = evaluate_coupled_group_gate(probability)
            self.assertAlmostEqual(gate.first_marginal_antithetic, probability)
            self.assertAlmostEqual(gate.second_marginal_antithetic, probability)

    def test_raw_score_estimator_remains_unbiased(self):
        for probability in (0.1, 0.3, 0.5, 0.7, 0.9):
            gate = evaluate_coupled_group_gate(probability)
            self.assertAlmostEqual(gate.iid_raw.expectation, gate.true_gradient)
            self.assertAlmostEqual(
                gate.antithetic_raw.expectation, gate.true_gradient
            )

    def test_iid_leave_one_out_is_unbiased(self):
        for probability in (0.1, 0.3, 0.5, 0.7, 0.9):
            gate = evaluate_coupled_group_gate(probability)
            self.assertAlmostEqual(
                gate.iid_leave_one_out.expectation, gate.true_gradient
            )

    def test_antithetic_leave_one_out_is_biased(self):
        gate = evaluate_coupled_group_gate(0.3)
        self.assertAlmostEqual(gate.true_gradient, 0.21)
        self.assertAlmostEqual(gate.antithetic_leave_one_out.expectation, 0.3)
        self.assertAlmostEqual(
            gate.antithetic_leave_one_out_relative_bias,
            0.3 / 0.21 - 1.0,
        )

    def test_more_informative_groups_do_not_imply_unbiased_gradients(self):
        gate = evaluate_coupled_group_gate(0.3)
        self.assertAlmostEqual(gate.iid_informative_probability, 0.42)
        self.assertAlmostEqual(gate.antithetic_informative_probability, 0.6)
        self.assertGreater(gate.antithetic_leave_one_out_bias, 0.0)

    def test_antithetic_raw_estimator_can_reduce_variance(self):
        gate = evaluate_coupled_group_gate(0.3)
        self.assertAlmostEqual(gate.raw_variance_ratio, (1.0 - 0.6) / 0.7)
        self.assertLess(gate.raw_variance_ratio, 1.0)

    def test_invalid_probability_fails_closed(self):
        for probability in (0.0, 1.0, -0.1, 1.1):
            with self.assertRaises(ValueError):
                evaluate_coupled_group_gate(probability)


if __name__ == "__main__":
    unittest.main()
