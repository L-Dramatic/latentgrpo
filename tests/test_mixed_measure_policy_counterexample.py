import math
import unittest

from research.mixed_measure_policy.counterexample import (
    clipped_gumbel_log_measure,
    clipped_gumbel_masses,
    moving_atom_dominance,
    standard_gumbel_log_density,
    two_token_weight_atoms,
)


class MixedMeasureCounterexampleTest(unittest.TestCase):
    def test_official_clip_bounds_create_positive_boundary_masses(self):
        masses = clipped_gumbel_masses()
        self.assertGreater(masses.lower_mass, 0.0)
        self.assertGreater(masses.upper_mass, 0.0)
        self.assertLess(masses.boundary_mass, 1.0)
        self.assertAlmostEqual(masses.lower_mass, math.exp(-math.exp(1.5)))

    def test_two_clipped_noises_create_three_weight_atoms(self):
        masses = clipped_gumbel_masses()
        atoms = two_token_weight_atoms(logit_gap=0.2)
        self.assertEqual(len(atoms), 3)
        self.assertTrue(all(0.0 < atom.location < 1.0 for atom in atoms))
        self.assertAlmostEqual(
            sum(atom.mass for atom in atoms),
            masses.boundary_mass**2,
            places=15,
        )

    def test_generic_policy_update_moves_all_atoms(self):
        audit = moving_atom_dominance(0.2, 0.37)
        self.assertTrue(audit.has_bidirectional_singularity)
        self.assertGreater(audit.old_only_mass, 0.0)
        self.assertGreater(audit.new_only_mass, 0.0)
        self.assertEqual(
            {atom.location for atom in audit.old_atoms}
            & {atom.location for atom in audit.new_atoms},
            set(),
        )

    def test_no_update_preserves_atom_support(self):
        audit = moving_atom_dominance(-0.4, -0.4)
        self.assertEqual(audit.old_only_mass, 0.0)
        self.assertEqual(audit.new_only_mass, 0.0)

    def test_boundary_mass_is_not_interior_density(self):
        lower = clipped_gumbel_log_measure(-1.5)
        upper = clipped_gumbel_log_measure(3.0)
        interior = clipped_gumbel_log_measure(0.0)
        self.assertEqual(lower.component, "lower_atom")
        self.assertEqual(upper.component, "upper_atom")
        self.assertEqual(interior.component, "interior_density")
        self.assertNotAlmostEqual(
            lower.log_value,
            standard_gumbel_log_density(-1.5),
        )
        self.assertNotAlmostEqual(
            upper.log_value,
            standard_gumbel_log_density(3.0),
        )

    def test_outside_clipped_support_has_zero_measure(self):
        result = clipped_gumbel_log_measure(3.1)
        self.assertEqual(result.component, "outside_support")
        self.assertEqual(result.log_value, -math.inf)

    def test_invalid_parameters_fail_closed(self):
        with self.assertRaises(ValueError):
            clipped_gumbel_masses(1.0, 1.0)
        with self.assertRaises(ValueError):
            two_token_weight_atoms(0.0, temperature=0.0)


if __name__ == "__main__":
    unittest.main()
