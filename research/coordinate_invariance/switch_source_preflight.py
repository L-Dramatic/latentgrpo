from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_blob(source_directory: Path, commit: str, relative: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(source_directory), "show", f"{commit}:{relative}"],
        check=True,
        capture_output=True,
    ).stdout


def _canonical_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _require(source: str, fragment: str, label: str) -> bool:
    if fragment not in source:
        raise ValueError(f"missing source contract: {label}")
    return True


def run(
    config: dict[str, Any],
    source_directory: Path,
    metadata_directory: Path,
) -> dict[str, Any]:
    source_directory = source_directory.resolve()
    metadata_directory = metadata_directory.resolve()
    source_config = config["source"]
    commit = subprocess.run(
        ["git", "-C", str(source_directory), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    gates: dict[str, bool] = {
        "source_commit": commit == source_config["commit"],
    }
    source_hashes: dict[str, str] = {}
    for relative, expected in source_config["files"].items():
        payload = _git_blob(source_directory, commit, relative)
        actual = hashlib.sha256(payload).hexdigest()
        source_hashes[relative] = actual
        gates[f"source_hash:{relative}"] = actual == expected
        worktree_payload = (source_directory / relative).read_bytes().replace(
            b"\r\n", b"\n"
        )
        gates[f"source_worktree_matches_blob:{relative}"] = (
            worktree_payload == payload
        )

    metadata_hashes: dict[str, str] = {}
    for relative, expected in config["adapter"]["metadata_files"].items():
        actual = _sha256(metadata_directory / relative)
        metadata_hashes[relative] = actual
        gates[f"metadata_hash:{relative}"] = actual == expected

    tokenizer_json = json.loads(
        (metadata_directory / "tokenizer.json").read_text(encoding="utf-8")
    )
    added_tokens = {
        row["content"]: int(row["id"])
        for row in tokenizer_json.get("added_tokens", [])
    }
    token_config = config["tokenizer"]
    token_ids = {
        "swi_start_id": added_tokens.get("<swi>"),
        "swi_end_id": added_tokens.get("</swi>"),
        "latent_id": added_tokens.get("<latent>"),
        "eos_id": added_tokens.get("<|im_end|>"),
    }
    for name, value in token_ids.items():
        gates[f"token_id:{name}"] = value == int(token_config[name])
    all_ids = list(tokenizer_json["model"]["vocab"].values()) + list(
        added_tokens.values()
    )
    gates["tokenizer_length"] = max(all_ids) + 1 == int(
        token_config["expected_length"]
    )

    adapter_config = json.loads(
        (metadata_directory / "adapter_config.json").read_text(encoding="utf-8")
    )
    gates["peft_version"] = adapter_config.get("peft_version") == config["adapter"][
        "peft_version"
    ]
    gates["adapter_base_family"] = "Qwen3-8B" in str(
        adapter_config.get("base_model_name_or_path", "")
    )

    model_source = (source_directory / "src/model/coconut_swi_model.py").read_text(
        encoding="utf-8"
    )
    latent_eval_source = (source_directory / "scripts/eval_latent.py").read_text(
        encoding="utf-8"
    )
    generic_eval_source = (source_directory / "scripts/eval_math500.py").read_text(
        encoding="utf-8"
    )
    launcher_source = (
        source_directory / "scripts/train_phase3_grpo.sh"
    ).read_text(encoding="utf-8")
    card = (metadata_directory / "README.md").read_text(encoding="utf-8")

    gates["dedicated_latent_driver"] = _require(
        latent_eval_source,
        "coconut_model.generate(",
        "dedicated eval must call CoconutSwiModel.generate",
    )
    gates["generic_eval_is_plain_generation"] = _require(
        generic_eval_source,
        "output_ids = model.generate(",
        "generic MATH-500 eval must remain classified as plain generation",
    )
    gates["latent_noise_disabled"] = _require(
        launcher_source,
        "LATENT_NOISE_SCALE=${LATENT_NOISE_SCALE:-0.0}",
        "paper-final latent noise default",
    )
    gates["latent_replay_disabled"] = _require(
        launcher_source,
        "LATENT_REPLAY=${LATENT_REPLAY:-0}",
        "paper-final latent replay default",
    )
    gates["forced_end_is_not_sampled"] = all(
        (
            _require(
                model_source,
                "response_ids_full.append(self.swi_end_id)",
                "forced end insertion",
            ),
            _require(
                model_source,
                "sampled_mask.append(0)  # forced, not sampled",
                "forced end policy mask",
            ),
        )
    )
    gates["card_minimum_dwell_4"] = _require(
        card, "| K_min (inference)| 4 |", "released checkpoint K_min"
    )
    gates["card_requires_latent_loop"] = _require(
        card,
        "will **not** perform the hidden-state recurrence",
        "naive generation warning",
    )

    return {
        "experiment_name": config["experiment_name"],
        "status": "pass" if all(gates.values()) else "fail",
        "evidence_level": "source and release-metadata contract only",
        "config_sha256": _canonical_hash(config),
        "source_commit": commit,
        "source_hashes": source_hashes,
        "metadata_hashes": metadata_hashes,
        "token_ids": token_ids,
        "gates": gates,
        "observations": {
            "paper_final_minimum_latent_dwell": 4,
            "paper_final_latent_noise_enabled": False,
            "paper_final_latent_replay_enabled": False,
            "valid_latent_driver": "scripts/eval_latent.py",
            "generic_eval_math500_executes_latent_recurrence": False,
            "rl_source_samples_swi_end_token": False,
            "rl_source_exit_decision": (
                "argmax over latent-step logits after minimum dwell, followed by "
                "a forced </swi> token with sampled_mask=0"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source-directory", required=True, type=Path)
    parser.add_argument("--metadata-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = run(config, args.source_directory, args.metadata_directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
