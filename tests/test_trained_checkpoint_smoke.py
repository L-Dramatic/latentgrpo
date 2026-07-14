import pytest

from research.policy_contract_audit.trained_checkpoint_smoke import (
    summarize_gate_status,
    tokenizer_id_diagnostics,
    validate_config,
)


class TinyTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def get_vocab(self):
        return self.vocab

    def __len__(self):
        return len(self.vocab)


def valid_config():
    return {
        "experiment_name": "smoke",
        "checkpoints": [
            {
                "name": "model",
                "id": "owner/model",
                "revision": "a" * 40,
                "local_dir": "model",
                "weight_file": "model.safetensors",
                "expected_weight_bytes": 1,
                "expected_weight_sha256": "b" * 64,
            }
        ],
        "runtime": {"preferred_device": "cuda", "max_prompt_tokens": 8},
        "gates": {"repeat_max_abs_error_max": 0.0},
    }


def test_validate_config_accepts_pinned_checkpoint():
    validate_config(valid_config())


def test_validate_config_rejects_mutable_revision():
    config = valid_config()
    config["checkpoints"][0]["revision"] = "main"
    with pytest.raises(ValueError, match="full revision"):
        validate_config(config)


def test_tokenizer_diagnostics_preserve_packaging_warning():
    diagnostics = tokenizer_id_diagnostics(
        TinyTokenizer({"a": 0, "b": 2, "extra": 3}), embedding_rows=3
    )
    assert diagnostics["tokenizer_max_id"] == 3
    assert diagnostics["out_of_range_token_ids"] == [3]


def test_gate_summary_requires_every_control():
    assert summarize_gate_status({"a": True, "b": True}) == "pass"
    assert summarize_gate_status({"a": True, "b": False}) == "fail"
