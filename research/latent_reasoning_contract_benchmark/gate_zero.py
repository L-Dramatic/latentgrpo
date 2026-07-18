"""Unified zero-training source-native fixture gate for four latent methods."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from research.behavioral_geometry.p1_official_sampler_replay import (
    replay_pinned_sampler_on_fake_request,
)
from research.behavioral_geometry.p1_source_sampler_contract import (
    SourceLatentSamplerConfig,
    source_style_latent_action,
)
from research.policy_conditional_mixture_closure.checkpoint_adapters import (
    sample_source_action,
)
from research.policy_conditional_mixture_closure.checkpoint_protocol import (
    checkpoint_profiles,
    load_protocol,
)
from research.policy_conditional_mixture_closure.soft_official_sampler_replay import (
    replay_pinned_soft_sampler,
)

from .deterministic_fixtures import run_coconut_fixture, run_codi_fixture
from .source_preflight import run_preflight


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("configs") / "gate_zero_fixtures_v1.json"
SOURCE_MANIFEST = Path(__file__).with_name("SOURCE_MANIFEST.json")
PCMC_CONFIG = (
    ROOT
    / "research"
    / "policy_conditional_mixture_closure"
    / "configs"
    / "pcmc_gate_ab_v1.json"
)


def _canonical_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _latent_fixture(config: dict[str, Any], tolerance: float) -> dict[str, Any]:
    logits = torch.tensor(config["logits"], dtype=torch.float64)
    hidden_size = int(config["embedding_hidden_size"])
    table = torch.arange(logits.numel() * hidden_size, dtype=torch.float64)
    table = table.reshape(logits.numel(), hidden_size) / 10.0
    sampler = SourceLatentSamplerConfig(
        max_topk=int(config["max_topk"]),
        top_p=float(config["top_p"]),
        temperature=float(config["temperature"]),
        gumbel_softmax_temperature=float(config["gumbel_softmax_temperature"]),
        noise_scale=float(config["noise_scale"]),
        use_one_sided_gumbel_noise=True,
        latent_end_token_id=int(config["latent_end_token_id"]),
    )
    seed = int(config["seed"])
    torch.manual_seed(seed)
    official = replay_pinned_sampler_on_fake_request(logits, sampler)
    independent = source_style_latent_action(
        logits, table, sampler, generator=torch.Generator().manual_seed(seed)
    )
    torch.manual_seed(seed)
    repeated = replay_pinned_sampler_on_fake_request(logits, sampler)
    ids_equal = torch.equal(official.topk_indices, independent.mixture_token_ids)
    weights_error = float((official.topk_probs - independent.mixture_probs).abs().max())
    normalization_error = abs(float(official.topk_probs.sum()) - 1.0)
    deterministic = torch.equal(official.topk_indices, repeated.topk_indices) and torch.equal(
        official.topk_probs, repeated.topk_probs
    )
    return {
        "method_id": "latent_grpo_llama_1b",
        "source_equivalent": ids_equal and weights_error <= tolerance,
        "replay_deterministic": bool(deterministic),
        "reconstruction_max_abs": weights_error,
        "normalization_error": normalization_error,
        "executed_latent_steps": 1,
        "proxy_token_id": official.next_token_id,
    }


def _soft_fixture(config: dict[str, Any], tolerance: float) -> dict[str, Any]:
    protocol = load_protocol(PCMC_CONFIG)
    profile = next(
        profile
        for profile in checkpoint_profiles(protocol)
        if profile.checkpoint_id == config["checkpoint_id"]
    )
    logits = torch.linspace(
        float(config["logit_min"]),
        float(config["logit_max"]),
        int(config["vocabulary_size"]),
    )
    seed = int(config["seed"])
    torch.manual_seed(seed)
    independent = sample_source_action(logits, profile)
    torch.manual_seed(seed)
    official = replay_pinned_soft_sampler(
        logits,
        top_p=profile.sampler.top_p,
        top_k=profile.sampler.top_k,
        max_topk=profile.sampler.max_topk,
        temperature=profile.sampler.temperature,
        gumbel_softmax_temperature=profile.sampler.gumbel_softmax_temperature,
        noise_scale=profile.sampler.noise_scale,
    )
    torch.manual_seed(seed)
    repeated = replay_pinned_soft_sampler(
        logits,
        top_p=profile.sampler.top_p,
        top_k=profile.sampler.top_k,
        max_topk=profile.sampler.max_topk,
        temperature=profile.sampler.temperature,
        gumbel_softmax_temperature=profile.sampler.gumbel_softmax_temperature,
        noise_scale=profile.sampler.noise_scale,
    )
    ids_equal = torch.equal(independent.token_ids, official.token_ids)
    weights_error = float((independent.weights - official.weights).abs().max())
    normalization_error = abs(float(official.weights.sum()) - 1.0)
    deterministic = torch.equal(official.token_ids, repeated.token_ids) and torch.equal(
        official.weights, repeated.weights
    )
    return {
        "method_id": "soft_grpo_qwen_1_5b",
        "source_equivalent": ids_equal and weights_error <= tolerance,
        "replay_deterministic": bool(deterministic),
        "reconstruction_max_abs": weights_error,
        "normalization_error": normalization_error,
        "executed_latent_steps": 1,
        "proxy_token_id": official.next_token_id,
    }


def run_gate_zero(root: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Gate 0 schema_version must be 1")
    manifest_sha = hashlib.sha256(SOURCE_MANIFEST.read_bytes()).hexdigest()
    if manifest_sha != config["source_manifest_sha256"]:
        raise ValueError("Gate 0 source manifest hash drift")
    source_report = run_preflight(root, SOURCE_MANIFEST)
    if source_report["decision"] != "PASS_GATE_MINUS_ONE":
        raise RuntimeError("Gate -1 source readiness no longer passes")

    tolerance = float(config["tolerance"])
    methods = [
        asdict(run_coconut_fixture(tolerance)),
        asdict(run_codi_fixture(tolerance)),
        _latent_fixture(config["latent_grpo"], tolerance),
        _soft_fixture(config["soft_grpo"], tolerance),
    ]
    decision = config["decision"]
    if decision.get("require_source_equivalence") is not True:
        raise ValueError("Gate 0 must require source equivalence")
    if decision.get("require_deterministic_replay") is not True:
        raise ValueError("Gate 0 must require deterministic replay")
    controls = {
        "required_method_count": len(methods) == int(decision["required_method_count"]),
        "all_source_equivalent": all(method["source_equivalent"] for method in methods),
        "all_replays_deterministic": all(
            method["replay_deterministic"] for method in methods
        ),
        "all_reconstruction_errors_bounded": all(
            method["reconstruction_max_abs"]
            <= float(decision["maximum_reconstruction_error"])
            for method in methods
        ),
        "all_normalization_errors_bounded": all(
            method.get("normalization_error", 0.0)
            <= float(decision["maximum_normalization_error"])
            for method in methods
        ),
    }
    passed = all(controls.values())
    return {
        "protocol_id": config["protocol_id"],
        "config_file_sha256": _file_sha256(config_path),
        "config_canonical_sha256": _canonical_sha256(config),
        "source_manifest_sha256": manifest_sha,
        "implementation_sha256": {
            "deterministic_fixtures.py": _file_sha256(
                Path(__file__).with_name("deterministic_fixtures.py")
            ),
            "gate_zero.py": _file_sha256(Path(__file__)),
        },
        "stage": "gate_zero_source_native_fixtures",
        "gpu_used": False,
        "training_used": False,
        "decision": "PASS_GATE_ZERO" if passed else "HOLD_GATE_ZERO",
        "controls": controls,
        "methods": methods,
        "scientific_evidence": "none; fixture conformance only",
        "next_required_action": (
            "freeze checkpoint-state pilot manifests before any effect inspection"
            if passed
            else "repair only source-equivalence or reconstruction controls"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_gate_zero(args.workspace, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "controls": report["controls"]}))
    return 0 if report["decision"] == "PASS_GATE_ZERO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
