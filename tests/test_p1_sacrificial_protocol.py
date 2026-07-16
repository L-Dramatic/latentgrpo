import tempfile
import unittest
from pathlib import Path

from research.behavioral_geometry.p1_sacrificial_protocol import (
    action_record_key,
    action_seed,
    cumulative_at_horizons,
    history_seed,
    load_protocol,
    path_record_key,
    summarize_discovery,
    validate_protocol,
)
from research.behavioral_geometry.p1_sacrificial_discovery import (
    append_record,
    load_record_map,
    write_json_atomic,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "research" / "behavioral_geometry" / "configs" / "p1_sacrificial_discovery_v1.json"


def _protocol():
    return load_protocol(CONFIG)


def _records(protocol, *, flip_prompts: int, late: bool):
    records = []
    prompt_count = len(protocol["prompts"])
    action_count = len(protocol["action_seeds"])
    history_count = len(protocol["forward_history_seeds"])
    for prompt_index, prompt in enumerate(protocol["prompts"]):
        prompt_id = prompt["prompt_id"]
        records.append(
            {
                "record_type": "action",
                "key": action_record_key(prompt_id, None),
                "prompt_id": prompt_id,
                "role": "reference",
                "endpoint": "NATURAL_VISIBLE",
                "status": "COMPLETE",
            }
        )
        for action_index in range(action_count):
            records.append(
                {
                    "record_type": "action",
                    "key": action_record_key(prompt_id, action_index),
                    "prompt_id": prompt_id,
                    "role": "candidate",
                    "action_index": action_index,
                    "endpoint": "NATURAL_VISIBLE",
                    "status": "COMPLETE",
                }
            )
            for direction in ("forward", "reverse"):
                for history_index in range(history_count):
                    if prompt_index < flip_prompts:
                        # At H8 action 0 is lower; by H64 it is higher. The
                        # same relation holds in every leave-one subset.
                        early = 0.001 + action_index * 0.001
                        late_value = 0.020 - action_index * 0.003
                    else:
                        early = 0.001 + action_index * 0.001
                        late_value = early + (0.010 if late else 0.00005)
                    per_step = [early / 8.0] * 8 + [(late_value - early) / 56.0] * 56
                    records.append(
                        {
                            "record_type": "path",
                            "key": path_record_key(
                                prompt_id, action_index, direction, history_index
                            ),
                            "prompt_id": prompt_id,
                            "action_index": action_index,
                            "direction": direction,
                            "history_index": history_index,
                            "status": "COMPLETE",
                            "per_step_kl": per_step,
                        }
                    )
    return records


class SacrificialProtocolTest(unittest.TestCase):
    def test_frozen_protocol_lints_and_has_global_seed_uniqueness(self):
        protocol = _protocol()
        audit = validate_protocol(protocol)
        self.assertEqual(audit.expected_action_records, 40)
        self.assertEqual(audit.expected_path_records, 256)
        seeds = []
        for prompt_index in range(8):
            for action_index in range(4):
                seeds.append(action_seed(protocol, prompt_index, action_index))
            for history_index in range(4):
                seeds.append(
                    history_seed(
                        protocol,
                        direction="forward",
                        prompt_index=prompt_index,
                        history_index=history_index,
                    )
                )
                for action_index in range(4):
                    seeds.append(
                        history_seed(
                            protocol,
                            direction="reverse",
                            prompt_index=prompt_index,
                            action_index=action_index,
                            history_index=history_index,
                        )
                    )
        self.assertEqual(len(seeds), len(set(seeds)))

    def test_absorbing_cumulative_kl_stays_flat_after_eos(self):
        result = cumulative_at_horizons([0.1, 0.2], [1, 3, 8, 64])
        self.assertEqual(result, {"1": 0.1, "3": 0.30000000000000004, "8": 0.30000000000000004, "64": 0.30000000000000004})

    def test_complete_strong_synthetic_signal_passes_go_gate(self):
        protocol = _protocol()
        summary = summarize_discovery(
            protocol, _records(protocol, flip_prompts=2, late=True), run_complete=True
        )
        self.assertEqual(summary["decision"], "GO-REWRITE-METHOD-CONTRACT")
        self.assertGreaterEqual(len(summary["robust_forward_flip_prompts"]), 2)

    def test_flat_complete_signal_is_killed(self):
        protocol = _protocol()
        records = _records(protocol, flip_prompts=0, late=False)
        summary = summarize_discovery(protocol, records, run_complete=True)
        self.assertEqual(summary["decision"], "KILL")

    def test_incomplete_records_hold_without_optional_sample_growth(self):
        protocol = _protocol()
        records = _records(protocol, flip_prompts=2, late=True)[:-1]
        summary = summarize_discovery(protocol, records, run_complete=False)
        self.assertEqual(summary["decision"], "HOLD-INSUFFICIENT")

    def test_jsonl_resume_helpers_preserve_completed_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records_path = root / "records.jsonl"
            manifest_path = root / "manifest.json"
            record_map = {}
            record = {"key": "unit-1", "record_type": "path", "status": "COMPLETE"}
            append_record(records_path, record_map, record)
            self.assertEqual(load_record_map(records_path), {"unit-1": record})
            with self.assertRaisesRegex(ValueError, "overwrite"):
                append_record(records_path, record_map, record)
            write_json_atomic(manifest_path, {"status": "RUNNING", "completed": 1})
            self.assertIn('"completed": 1', manifest_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
