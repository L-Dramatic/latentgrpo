import math
import unittest

from research.observable_marginal_policy.exact_identity_gate import (
    marginal_logits,
    normalized_responsibility_entropy,
    off_source_mass,
    ompi_likelihood_gradient,
    ompi_loss,
    path_candidate_mutual_information,
    responsibility_ess,
    run_identity_gate,
    softmax,
    tpo_candidate_loss,
    tpo_on_marginal_likelihood_gradient,
)


class ObservableMarginalPolicyGateTest(unittest.TestCase):
    def setUp(self):
        self.current = (
            (-0.3, -1.2, -2.1),
            (-0.5, -0.8, -1.9),
            (-1.4, -0.7, -0.6),
        )
        self.old = tuple(
            tuple(value - 0.1 for value in row) for row in self.current
        )
        self.rewards = (1.0, 0.0, -0.5)

    def test_ompi_loss_is_tpo_on_empirical_marginal_logits(self):
        ompi = ompi_loss(self.current, self.old, self.rewards, beta=0.8)
        composed = tpo_candidate_loss(
            marginal_logits(self.current),
            marginal_logits(self.old),
            self.rewards,
            beta=0.8,
        )
        self.assertAlmostEqual(ompi, composed, places=14)

    def test_ompi_responsibility_gradient_is_composed_chain_rule(self):
        ompi = ompi_likelihood_gradient(
            self.current, self.old, self.rewards, beta=0.8
        )
        composed = tpo_on_marginal_likelihood_gradient(
            self.current, self.old, self.rewards, beta=0.8
        )
        for ompi_row, composed_row in zip(ompi, composed):
            for ompi_value, composed_value in zip(ompi_row, composed_row):
                self.assertAlmostEqual(ompi_value, composed_value, places=14)

    def test_report_gate_metrics_false_pass_latent_irrelevance(self):
        base = softmax((-0.2, -0.9, -1.8, -2.4))
        matrix = tuple(tuple(math.log(value) for value in base) for _ in range(4))

        self.assertEqual(responsibility_ess(matrix), (4.0, 4.0, 4.0, 4.0))
        for entropy in normalized_responsibility_entropy(matrix):
            self.assertAlmostEqual(entropy, 1.0)
        for mass in off_source_mass(matrix, (0, 1, 2, 3)):
            self.assertAlmostEqual(mass, 0.75)
        self.assertAlmostEqual(path_candidate_mutual_information(matrix), 0.0)

        marginal_probability = softmax(marginal_logits(matrix))
        for observed, expected in zip(marginal_probability, base):
            self.assertAlmostEqual(observed, expected)

    def test_path_sensitivity_control_detects_nontrivial_paths(self):
        path_dependent = (
            (math.log(0.70), math.log(0.25), math.log(0.05)),
            (math.log(0.55), math.log(0.40), math.log(0.05)),
            (math.log(0.10), math.log(0.45), math.log(0.45)),
        )
        self.assertGreater(path_candidate_mutual_information(path_dependent), 0.05)

    def test_frozen_gate_records_both_failures(self):
        result = run_identity_gate()
        self.assertTrue(result.standalone_optimizer_identity_passes)
        self.assertTrue(result.report_gate_1a_would_false_pass)

    def test_invalid_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            marginal_logits(())
        with self.assertRaises(ValueError):
            marginal_logits(((0.0, 1.0), (0.0,)))
        with self.assertRaises(ValueError):
            off_source_mass(((0.0, 1.0), (1.0, 0.0)), (0,))
        with self.assertRaises(ValueError):
            tpo_candidate_loss((0.0,), (0.0,), (1.0,), beta=0.0)


if __name__ == "__main__":
    unittest.main()

