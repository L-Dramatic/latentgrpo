import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from research.latent_reasoning_contract_benchmark.schema import (
    LAYER_IDS,
    load_manifest,
    validate_manifest,
)
from research.latent_reasoning_contract_benchmark.source_preflight import (
    run_preflight,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    ROOT / "research" / "latent_reasoning_contract_benchmark" / "SOURCE_MANIFEST.json"
)


class LatentReasoningContractBenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.manifest = load_manifest(MANIFEST_PATH)

    def test_manifest_declares_four_distinct_families_and_two_stochastic_methods(self):
        methods = self.manifest["methods"]
        self.assertEqual(len(methods), 4)
        self.assertEqual(len({method["family"] for method in methods}), 4)
        self.assertEqual(sum(method["stochastic"] for method in methods), 2)

    def test_every_method_declares_ordered_layers_with_honest_policy_na(self):
        for method in self.manifest["methods"]:
            layers = method["contract_layers"]
            self.assertEqual(tuple(layer["id"] for layer in layers), LAYER_IDS)
            policy = next(layer for layer in layers if layer["id"] == "policy_measure")
            if method["stochastic"]:
                self.assertEqual(policy["status"], "source_anchored")
            else:
                self.assertEqual(policy["status"], "not_applicable")
                self.assertTrue(policy["reason"])

    def test_stochastic_method_cannot_hide_policy_measure_as_not_applicable(self):
        mutated = copy.deepcopy(self.manifest)
        stochastic = next(method for method in mutated["methods"] if method["stochastic"])
        policy = next(
            layer for layer in stochastic["contract_layers"] if layer["id"] == "policy_measure"
        )
        policy.update(status="not_applicable", evidence=[], reason="hidden")
        with self.assertRaisesRegex(ValueError, "stochastic but omits policy_measure"):
            validate_manifest(mutated)

    def test_collided_broad_claim_cannot_be_reenabled(self):
        mutated = copy.deepcopy(self.manifest)
        mutated["scope"]["broad_mechanism_claim"] = "first_mechanism_audit"
        with self.assertRaisesRegex(ValueError, "collided broad mechanism claim"):
            validate_manifest(mutated)

    def test_preflight_fails_closed_on_source_commit_drift(self):
        original = subprocess.check_output

        def drifted(command, *args, **kwargs):
            if command[-2:] == ["rev-parse", "HEAD"]:
                return "0" * 40 + "\n"
            return original(command, *args, **kwargs)

        with mock.patch("subprocess.check_output", side_effect=drifted):
            report = run_preflight(ROOT, MANIFEST_PATH)
        self.assertEqual(report["decision"], "HOLD_GATE_MINUS_ONE")
        self.assertFalse(report["criteria"]["all_sources_pinned_and_verified"])

    def test_manifest_rejects_workspace_escape(self):
        mutated = copy.deepcopy(self.manifest)
        mutated["methods"][0]["source"]["path"] = "../other-checkout"
        with self.assertRaisesRegex(ValueError, "inside the workspace"):
            validate_manifest(mutated)

    def test_manifest_round_trip_is_stable_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(self.manifest, indent=2) + "\n", encoding="utf-8")
            reloaded = load_manifest(path)
        self.assertEqual(reloaded, self.manifest)


if __name__ == "__main__":
    unittest.main()
