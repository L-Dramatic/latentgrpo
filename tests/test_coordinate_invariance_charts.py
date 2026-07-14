import math
import unittest

import torch

from research.coordinate_invariance import (
    AffineChart,
    SinhChart,
    add_isotropic_noise_in_chart,
    interpolate_in_chart,
    nearest_neighbor_in_chart,
)


class CoordinateChartTest(unittest.TestCase):
    def setUp(self):
        self.dtype = torch.float64

    def test_identity_round_trip_is_exact(self):
        chart = AffineChart.identity(4, dtype=self.dtype)
        native = torch.tensor(
            [[0.25, -1.0, 3.0, 0.0], [1.5, 2.0, -0.5, 4.0]],
            dtype=self.dtype,
        )

        diagnostics = chart.diagnose(native)

        self.assertTrue(diagnostics.all_finite)
        self.assertEqual(diagnostics.max_abs_round_trip_error, 0.0)
        self.assertEqual(diagnostics.relative_l2_round_trip_error, 0.0)

    def test_affine_chart_matches_requested_condition_number(self):
        chart = AffineChart.with_condition_number(
            6, 10.0, seed=17, dtype=self.dtype, bias_scale=0.25
        )
        native = torch.randn(8, 6, dtype=self.dtype)

        diagnostics = chart.diagnose(native)

        self.assertAlmostEqual(chart.condition_number, 10.0, places=10)
        self.assertLess(diagnostics.max_abs_round_trip_error, 1e-12)
        self.assertLess(diagnostics.relative_l2_round_trip_error, 1e-12)

    def test_high_precision_chart_recovers_float32_values_before_consumption(self):
        chart = AffineChart.with_condition_number(
            8,
            20.0,
            seed=21,
            dtype=torch.float64,
            compute_dtype=torch.float64,
        )
        native = torch.randn(16, 8, dtype=torch.float32)

        reconstructed = chart.decode(chart.encode(native)).to(dtype=native.dtype)

        self.assertTrue(torch.equal(reconstructed, native))

    def test_singular_affine_chart_is_rejected(self):
        matrix = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=self.dtype)
        with self.assertRaisesRegex(ValueError, "full rank"):
            AffineChart(matrix)

    def test_orthogonal_chart_preserves_euclidean_neighbor(self):
        query = torch.tensor([0.2, -0.3, 0.7], dtype=self.dtype)
        candidates = torch.tensor(
            [[0.1, -0.2, 0.6], [1.0, 0.0, 0.0], [-0.5, 0.5, 1.0]],
            dtype=self.dtype,
        )
        identity = AffineChart.identity(3, dtype=self.dtype)
        orthogonal = AffineChart.random_orthogonal(3, seed=9, dtype=self.dtype)

        native_result = nearest_neighbor_in_chart(query, candidates, identity)
        rotated_result = nearest_neighbor_in_chart(query, candidates, orthogonal)

        self.assertEqual(native_result.index, rotated_result.index)
        torch.testing.assert_close(
            native_result.all_distances,
            rotated_result.all_distances,
            atol=1e-12,
            rtol=1e-12,
        )

    def test_anisotropic_chart_can_flip_euclidean_neighbor(self):
        query = torch.zeros(2, dtype=self.dtype)
        candidates = torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=self.dtype)
        identity = AffineChart.identity(2, dtype=self.dtype)
        anisotropic = AffineChart(
            torch.diag(torch.tensor([3.0, 0.25], dtype=self.dtype)),
            name="anisotropic-control",
        )

        native_result = nearest_neighbor_in_chart(query, candidates, identity)
        charted_result = nearest_neighbor_in_chart(query, candidates, anisotropic)

        self.assertEqual(native_result.index, 0)
        self.assertEqual(charted_result.index, 1)

    def test_affine_interpolation_is_coordinate_equivariant(self):
        left = torch.tensor([-2.0, 1.0, 0.5], dtype=self.dtype)
        right = torch.tensor([1.0, -0.5, 3.0], dtype=self.dtype)
        chart = AffineChart.with_condition_number(
            3, 20.0, seed=4, dtype=self.dtype, bias_scale=0.5
        )
        alpha = 0.35

        interpolated = interpolate_in_chart(left, right, chart, alpha=alpha)
        expected = (1.0 - alpha) * left + alpha * right

        torch.testing.assert_close(interpolated, expected, atol=1e-12, rtol=1e-12)

    def test_nonlinear_interpolation_is_chart_dependent(self):
        left = torch.tensor([-2.0, 0.2], dtype=self.dtype)
        right = torch.tensor([1.0, -0.4], dtype=self.dtype)
        chart = SinhChart(2, scale=[0.8, 0.4], dtype=self.dtype)

        interpolated = interpolate_in_chart(left, right, chart, alpha=0.5)
        native_midpoint = 0.5 * (left + right)

        self.assertGreater(
            torch.linalg.vector_norm(interpolated - native_midpoint).item(), 1e-2
        )

    def test_chart_isotropic_noise_transports_to_different_native_noise(self):
        native = torch.zeros(2, dtype=self.dtype)
        identity = AffineChart.identity(2, dtype=self.dtype)
        anisotropic = AffineChart(
            torch.diag(torch.tensor([2.0, 0.5], dtype=self.dtype)),
            name="anisotropic-noise-control",
        )
        identity_generator = torch.Generator(device="cpu").manual_seed(123)
        chart_generator = torch.Generator(device="cpu").manual_seed(123)

        identity_sample = add_isotropic_noise_in_chart(
            native,
            identity,
            standard_deviation=0.2,
            generator=identity_generator,
        )
        chart_sample = add_isotropic_noise_in_chart(
            native,
            anisotropic,
            standard_deviation=0.2,
            generator=chart_generator,
        )

        self.assertFalse(torch.allclose(identity_sample, chart_sample))
        torch.testing.assert_close(
            chart_sample,
            identity_sample / torch.tensor([2.0, 0.5], dtype=self.dtype),
            atol=1e-12,
            rtol=1e-12,
        )

    def test_sinh_chart_round_trip_and_jacobian(self):
        chart = SinhChart(3, scale=[0.2, 0.5, 1.0], dtype=self.dtype)
        native = torch.tensor(
            [[-1.2, 0.0, 0.5], [0.3, -0.8, 1.1]], dtype=self.dtype
        )

        diagnostics = chart.diagnose(native)
        log_determinant = chart.log_abs_det_jacobian(native)
        local_condition = chart.local_condition_number(native)

        self.assertLess(diagnostics.max_abs_round_trip_error, 1e-12)
        self.assertEqual(log_determinant.shape, torch.Size([2]))
        self.assertTrue((log_determinant >= -1e-14).all())
        self.assertTrue((local_condition >= 1.0).all())

    def test_affine_log_determinant_matches_matrix(self):
        matrix = torch.diag(torch.tensor([2.0, 3.0, 0.5], dtype=self.dtype))
        chart = AffineChart(matrix)
        native = torch.zeros(5, 3, dtype=self.dtype)

        values = chart.log_abs_det_jacobian(native)

        expected = math.log(3.0)
        torch.testing.assert_close(
            values,
            torch.full((5,), expected, dtype=self.dtype),
            atol=1e-12,
            rtol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
