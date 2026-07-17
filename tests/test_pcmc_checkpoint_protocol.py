import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from research.policy_conditional_mixture_closure.checkpoint_protocol import (
    METHODS,
    action_seed,
    audit_as_dict,
    checkpoint_profiles,
    continuation_seed,
    example_ids,
    load_protocol,
    selected_confirmation_ids,
    split_example_ids,
    validate_protocol,
)
from research.policy_conditional_mixture_closure.gate_a_analysis import (
    a0_record_key,
    a1_record_key,
    summarize_gate_a,
)
from research.policy_conditional_mixture_closure.task_manifest import (
    RecordStore,
    build_a0_manifest,
    build_a1_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "research"
    / "policy_conditional_mixture_closure"
    / "configs"
    / "pcmc_gate_ab_v1.json"
)


def _protocol():
    return load_protocol(CONFIG)


def _a0_records(protocol, *, flat_gap=False):
    records = []
    splits = split_example_ids(protocol)
    protocol_hash = validate_protocol(protocol).protocol_sha256
    for checkpoint_index, profile in enumerate(checkpoint_profiles(protocol)):
        for example_index, example_id in enumerate(example_ids(protocol)):
            gap = 0.001 if flat_gap else 0.001 + (example_index % 100) * 0.001
            records.append(
                {
                    "record_type": "a0_event",
                    "key": a0_record_key(profile.checkpoint_id, example_id),
                    "status": "COMPLETE",
                    "protocol_id": protocol["protocol_id"],
                    "protocol_sha256": protocol_hash,
                    "checkpoint_id": profile.checkpoint_id,
                    "example_id": example_id,
                    "dataset_partition": splits[example_id],
                    "action_seed": action_seed(
                        protocol, checkpoint_index, example_index
                    ),
                    "js_branch_arithmetic_nats": gap,
                    "weight_entropy_normalized": 0.55 + (example_index % 3) * 0.01,
                    "maximum_weight": 0.45 - (example_index % 3) * 0.01,
                    "effective_support": 3.0 + (example_index % 5) * 0.1,
                    "structural_end_mass": 0.0,
                    "content_support_size": 5,
                    "prompt_token_count": 40 + (example_index % 37),
                    "latent_step_index": 0,
                }
            )
    return records


def _a1_records(protocol, a0_records, *, strong):
    records = []
    splits = split_example_ids(protocol)
    protocol_hash = validate_protocol(protocol).protocol_sha256
    example_index = {
        example_id: index for index, example_id in enumerate(example_ids(protocol))
    }
    for checkpoint_index, profile in enumerate(checkpoint_profiles(protocol)):
        selection = selected_confirmation_ids(
            protocol, a0_records, checkpoint_id=profile.checkpoint_id
        )
        high = set(selection["selected_high"])
        selected = selection["selected_high"] + selection["selected_low"]
        for example_id in selected:
            for method in METHODS:
                for replicate_index in range(8):
                    if not strong or example_id not in high:
                        correct = int(replicate_index < 4)
                    elif method == "randomized_hard":
                        correct = 1
                    elif method == "arithmetic":
                        correct = int(replicate_index < example_index[example_id] % 2)
                    else:
                        correct = int(replicate_index < 4)
                    key = a1_record_key(
                        profile.checkpoint_id,
                        example_id,
                        method,
                        replicate_index,
                    )
                    records.append(
                        {
                            "record_type": "a1_continuation",
                            "key": key,
                            "status": "COMPLETE",
                            "protocol_id": protocol["protocol_id"],
                            "protocol_sha256": protocol_hash,
                            "checkpoint_id": profile.checkpoint_id,
                            "example_id": example_id,
                            "dataset_partition": splits[example_id],
                            "method": method,
                            "replicate_index": replicate_index,
                            "continuation_seed": continuation_seed(
                                protocol,
                                checkpoint_index,
                                example_index[example_id],
                                replicate_index,
                            ),
                            "correct": correct,
                            "response_sha256": hashlib.sha256(
                                key.encode("utf-8")
                            ).hexdigest(),
                        }
                    )
    return records


class PcmcCheckpointProtocolTest(unittest.TestCase):
    def test_frozen_protocol_lints_and_counts_work(self):
        protocol = _protocol()
        audit = audit_as_dict(protocol)
        self.assertEqual(audit["example_count"], 500)
        self.assertEqual(audit["calibration_count"], 250)
        self.assertEqual(audit["confirmation_count"], 250)
        self.assertEqual(audit["maximum_a0_records"], 1000)
        self.assertEqual(audit["maximum_a1_records"], 8192)

    def test_label_blind_split_is_exact_and_deterministic(self):
        protocol = _protocol()
        first = split_example_ids(protocol)
        second = split_example_ids(protocol)
        self.assertEqual(first, second)
        self.assertEqual(sum(value == "calibration" for value in first.values()), 250)
        self.assertEqual(sum(value == "confirmation" for value in first.values()), 250)

    def test_action_and_continuation_seeds_are_globally_unique(self):
        protocol = _protocol()
        actions = set()
        continuations = set()
        for checkpoint_index in range(2):
            for example_index in range(500):
                actions.add(action_seed(protocol, checkpoint_index, example_index))
                for replicate_index in range(8):
                    continuations.add(
                        continuation_seed(
                            protocol,
                            checkpoint_index,
                            example_index,
                            replicate_index,
                        )
                    )
        self.assertEqual(len(actions), 1000)
        self.assertEqual(len(continuations), 8000)
        self.assertFalse(actions & continuations)

    def test_complete_a0_signal_authorizes_only_a1(self):
        protocol = _protocol()
        summary = summarize_gate_a(
            protocol, _a0_records(protocol), completed_stage="A0"
        )
        self.assertEqual(summary["overall_decision"], "ADVANCE_TO_A1_COLLECTION")
        self.assertEqual(summary["authorization"], "A1_CAUSAL_CONTINUATION")

    def test_flat_a0_signal_is_killed_before_continuations(self):
        protocol = _protocol()
        summary = summarize_gate_a(
            protocol, _a0_records(protocol, flat_gap=True), completed_stage="A0"
        )
        self.assertEqual(summary["overall_decision"], "KILL_PCMC_GATE_A0")
        self.assertEqual(summary["authorization"], "NONE")

    def test_incomplete_a0_is_operationally_blocked_not_scientifically_killed(self):
        protocol = _protocol()
        summary = summarize_gate_a(
            protocol, _a0_records(protocol)[:-1], completed_stage="A0"
        )
        self.assertEqual(summary["overall_decision"], "BLOCKED_OPERATIONAL")

    def test_strong_complete_a1_signal_authorizes_only_b0(self):
        protocol = _protocol()
        a0 = _a0_records(protocol)
        records = a0 + _a1_records(protocol, a0, strong=True)
        summary = summarize_gate_a(
            protocol,
            records,
            completed_stage="A1",
            bootstrap_replicates=200,
        )
        self.assertEqual(summary["overall_decision"], "ADVANCE_TO_B0_ORACLE")
        self.assertEqual(summary["authorization"], "B0_CONSTRAINED_ORACLE")

    def test_null_complete_a1_signal_is_killed(self):
        protocol = _protocol()
        a0 = _a0_records(protocol)
        records = a0 + _a1_records(protocol, a0, strong=False)
        summary = summarize_gate_a(
            protocol,
            records,
            completed_stage="A1",
            bootstrap_replicates=200,
        )
        self.assertEqual(summary["overall_decision"], "KILL_PCMC_GATE_A1")

    def test_unselected_optional_a1_work_fails_closed(self):
        protocol = _protocol()
        a0 = _a0_records(protocol)
        record = _a1_records(protocol, a0, strong=True)[0]
        record = dict(record)
        record["example_id"] = "math500:000"
        record["dataset_partition"] = split_example_ids(protocol)["math500:000"]
        record["key"] = a1_record_key(
            record["checkpoint_id"],
            record["example_id"],
            record["method"],
            record["replicate_index"],
        )
        profile_index = [
            profile.checkpoint_id for profile in checkpoint_profiles(protocol)
        ].index(record["checkpoint_id"])
        record["continuation_seed"] = continuation_seed(
            protocol, profile_index, 0, record["replicate_index"]
        )
        with self.assertRaisesRegex(ValueError, "optional or unselected"):
            summarize_gate_a(
                protocol, a0 + [record], completed_stage="A0"
            )

    def test_task_manifests_are_deterministic_and_pair_method_seeds(self):
        protocol = _protocol()
        first = build_a0_manifest(protocol)
        second = build_a0_manifest(protocol)
        self.assertEqual(first, second)
        self.assertEqual(first["task_count"], 1000)

        a0 = _a0_records(protocol)
        a1 = build_a1_manifest(protocol, a0)
        self.assertLessEqual(a1["task_count"], 8192)
        seeds_by_unit = {}
        for task in a1["tasks"]:
            unit = (
                task["checkpoint_id"],
                task["example_id"],
                task["replicate_index"],
            )
            seeds_by_unit.setdefault(unit, set()).add(task["continuation_seed"])
        self.assertTrue(seeds_by_unit)
        self.assertTrue(all(len(values) == 1 for values in seeds_by_unit.values()))

    def test_a1_manifest_refuses_a0_that_failed(self):
        protocol = _protocol()
        with self.assertRaisesRegex(ValueError, "did not authorize"):
            build_a1_manifest(protocol, _a0_records(protocol, flat_gap=True))

    def test_record_store_is_immutable_resumable_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = RecordStore(root)
            record = {"key": "unit-1", "value": 3}
            self.assertEqual(store.put(record), "CREATED")
            self.assertEqual(store.put(record), "RESUMED")
            with self.assertRaisesRegex(ValueError, "immutable record mismatch"):
                store.put({"key": "unit-1", "value": 4})
            destination = root / "records.jsonl"
            first = store.compact(destination)
            second = store.compact(destination)
            self.assertEqual(first["records_sha256"], second["records_sha256"])
            self.assertEqual(
                json.loads(destination.read_text(encoding="utf-8")), record
            )


if __name__ == "__main__":
    unittest.main()
