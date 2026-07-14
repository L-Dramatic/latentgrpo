import json
import unittest
from pathlib import Path

from research.coordinate_invariance.toy_gate import load_config, run_toy_gate


class ToyCoordinateGateTest(unittest.TestCase):
    def test_preregistered_toy_contract_passes(self):
        config_path = (
            Path(__file__).parents[1]
            / "research"
            / "coordinate_invariance"
            / "configs"
            / "toy_contract_v1.json"
        )
        config = load_config(config_path)

        result = run_toy_gate(config)

        self.assertEqual(result["status"], "pass")
        self.assertFalse(result["scientific_evidence"])
        self.assertTrue(all(result["checks"].values()))
        self.assertGreater(result["metrics"]["anisotropic_neighbor_flip_rate"], 0.0)
        self.assertEqual(result["metrics"]["orthogonal_neighbor_flip_rate"], 0.0)
        json.dumps(result, sort_keys=True)


if __name__ == "__main__":
    unittest.main()

