import unittest

from research.coordinate_invariance.statistics import bootstrap_mean


class BootstrapStatisticsTest(unittest.TestCase):
    def test_constant_values_have_degenerate_interval(self):
        estimate = bootstrap_mean(
            [0.25] * 20, bootstrap_samples=200, seed=1
        )

        self.assertEqual(estimate.value, 0.25)
        self.assertEqual(estimate.ci_low, 0.25)
        self.assertEqual(estimate.ci_high, 0.25)
        self.assertEqual(estimate.count, 20)

    def test_seeded_bootstrap_is_reproducible(self):
        first = bootstrap_mean(range(10), bootstrap_samples=500, seed=7)
        second = bootstrap_mean(range(10), bootstrap_samples=500, seed=7)

        self.assertEqual(first, second)
        self.assertLessEqual(first.ci_low, first.value)
        self.assertGreaterEqual(first.ci_high, first.value)


if __name__ == "__main__":
    unittest.main()

