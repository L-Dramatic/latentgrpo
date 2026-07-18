"""Independent cross-evidence audit for the frozen LRC-Bench thesis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from research.policy_conditional_mixture_closure.checkpoint_protocol import (
    load_protocol,
)
from research.policy_conditional_mixture_closure.gate_a_analysis import (
    load_jsonl,
    summarize_gate_a,
    write_json_atomic,
)
from research.policy_conditional_mixture_closure.posthoc_a0_diagnostics import (
    describe_a0,
)

from .verify_deterministic_confirmation import verify_confirmation


ROOT = Path(__file__).resolve().parents[2]

EXPECTED_SHA256 = {
    "lrc_claim_contract": "de06e3a0108b9db6f048a6e151c2c95baf34ae22537871f1ddfece92557afbde",
    "deterministic_config": "b34b613681bdd3629f6fcefdac53f874be508fd28fe227fbe442cb720520e3b4",
    "deterministic_records": "1d23ce1717d1c5b39331f61ebfa7e3d87bd2da5613e5e2e1cc1680e14f18c6e5",
    "deterministic_summary": "fce0e9b759b1b4918b4e7ae435d6c6c85e2d87b138ba60ca5d700008cfa9489d",
    "deterministic_audit": "3585a3524d32a3cbbd17b2d0532acdb1780a6179a62db0181a1fd9fb3e4045dc",
    "pcmc_protocol": "67d1a6b952823337433888cbfebf7841cb3851739af57426e68fd184a4760085",
    "pcmc_records": "69d0852e9e65408427fdf2d7271138193475eef6b8d79c755669dcc48ed765eb",
    "pcmc_decision": "3b1fd475cf2cb8f2f3e23d830ef83484ea3de2816685c1ec0ac1edbc1ec77935",
    "pcmc_posthoc": "ab4697e273af79312b369c906abfcdc29fbaf23e6adaa54fb208a7b9ac7feee4",
    "pcmc_asset_preflight": "8915c21701efb9a0b867f30c795a6f7ef2779da0902eb8764234a2b2760a0d28",
    "pcmc_latent_preflight": "81d3763a5214524ea0c992ab3feeeca556a8787578330feadde915da733ddc92",
    "pcmc_soft_preflight": "8198a55456b406bbd3f9ffa6efe2f36e7afc406dc954572f7cfce751779565be",
}

KILL_CLAUSE = (
    "policy-contract findings occur only in Latent-GRPO and disappear in\n"
    "  SofT-GRPO or another stochastic family;"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _paths(root: Path) -> dict[str, Path]:
    lrc = root / "research" / "latent_reasoning_contract_benchmark"
    lrc_artifacts = root / "artifacts" / "latent_reasoning_contract_benchmark"
    pcmc = root / "research" / "policy_conditional_mixture_closure"
    pcmc_artifacts = root / "artifacts" / "pcmc_gate"
    return {
        "lrc_claim_contract": lrc / "CLAIM_CONTRACT.md",
        "deterministic_config": lrc / "configs" / "deterministic_intervention_confirmation_v1.json",
        "deterministic_records": lrc_artifacts / "deterministic_intervention_confirmation_v1_records.jsonl",
        "deterministic_summary": lrc_artifacts / "deterministic_intervention_confirmation_v1.json",
        "deterministic_audit": lrc_artifacts / "deterministic_intervention_confirmation_v1_audit.json",
        "pcmc_protocol": pcmc / "configs" / "pcmc_gate_ab_v1.json",
        "pcmc_records": pcmc_artifacts / "local_a0" / "a0_records.jsonl",
        "pcmc_decision": pcmc_artifacts / "local_a0" / "a0_decision.json",
        "pcmc_posthoc": pcmc_artifacts / "local_a0" / "posthoc_a0_diagnostics.json",
        "pcmc_asset_preflight": pcmc_artifacts / "asset_preflight.json",
        "pcmc_latent_preflight": pcmc_artifacts / "local_preflight_latent_grpo_llama_1b.json",
        "pcmc_soft_preflight": pcmc_artifacts / "local_preflight_soft_grpo_qwen_1_5b.json",
    }


def audit_current_thesis(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    paths = _paths(root)
    observed_hashes = {name: _sha256(path) for name, path in paths.items()}
    hash_controls = {
        name: observed_hashes[name] == expected
        for name, expected in EXPECTED_SHA256.items()
    }

    deterministic_recomputed = verify_confirmation(
        root,
        paths["deterministic_config"],
        paths["deterministic_records"],
        paths["deterministic_summary"],
    )
    deterministic_stored = _load_json(paths["deterministic_audit"])

    protocol = load_protocol(paths["pcmc_protocol"])
    records = load_jsonl(paths["pcmc_records"])
    pcmc_decision_recomputed = summarize_gate_a(
        protocol, records, completed_stage="A0"
    )
    pcmc_posthoc_recomputed = describe_a0(protocol, records)
    pcmc_decision_stored = _load_json(paths["pcmc_decision"])
    pcmc_posthoc_stored = _load_json(paths["pcmc_posthoc"])
    asset_preflight = _load_json(paths["pcmc_asset_preflight"])
    latent_preflight = _load_json(paths["pcmc_latent_preflight"])
    soft_preflight = _load_json(paths["pcmc_soft_preflight"])

    latent_a0 = pcmc_decision_recomputed["checkpoint_results"][
        "latent_grpo_llama_1b"
    ]["a0"]
    soft_a0 = pcmc_decision_recomputed["checkpoint_results"][
        "soft_grpo_qwen_1_5b"
    ]["a0"]
    latent_posthoc = pcmc_posthoc_recomputed["checkpoint_results"][
        "latent_grpo_llama_1b"
    ]
    soft_posthoc = pcmc_posthoc_recomputed["checkpoint_results"][
        "soft_grpo_qwen_1_5b"
    ]

    contract_text = paths["lrc_claim_contract"].read_text(encoding="utf-8")
    source_replay_pass = (
        latent_preflight.get("status") == "PASS_ENGINEERING_PREFLIGHT"
        and soft_preflight.get("status") == "PASS_ENGINEERING_PREFLIGHT"
        and all(
            item.get("official_sampler_replay_match") is True
            for item in soft_preflight.get("task_results", [])
        )
        and len(soft_preflight.get("task_results", [])) == 4
    )
    controls = {
        "all_frozen_hashes_match": all(hash_controls.values()),
        "deterministic_confirmation_recomputes": (
            deterministic_recomputed["decision"] == "PASS_CONFIRMATION_AUDIT"
            and deterministic_recomputed == deterministic_stored
        ),
        "pcmc_decision_recomputes": pcmc_decision_recomputed == pcmc_decision_stored,
        "pcmc_posthoc_recomputes": pcmc_posthoc_recomputed == pcmc_posthoc_stored,
        "exact_record_coverage": len(records) == 1000,
        "assets_and_source_replay_pass": (
            asset_preflight.get("status") == "PASS" and source_replay_pass
        ),
        "frozen_kill_clause_present": KILL_CLAUSE in contract_text,
    }
    integrity_pass = all(controls.values())

    family_specific = (
        latent_a0["decision"] == "ADVANCE_A1"
        and soft_a0["decision"] == "KILL_A0"
        and pcmc_decision_recomputed["overall_decision"] == "KILL_PCMC_GATE_A0"
    )
    soft_effectively_discrete = (
        soft_posthoc["fraction_js_at_least_0_005"] == 0
        and soft_posthoc["maximum_weight"]["quantiles"]["q50"] >= 0.999
        and soft_posthoc["effective_support"]["quantiles"]["q50"] <= 1.001
        and soft_posthoc["top_token_disagreement_rate"] == 0
    )
    kill_triggered = integrity_pass and family_specific
    if not integrity_pass:
        decision = "HOLD_LRC_EVIDENCE_INTEGRITY"
    elif kill_triggered:
        decision = "KILL_LRC_CURRENT_THESIS"
    else:
        decision = "HOLD_LRC_THESIS_NOT_ESTABLISHED"

    return {
        "audit_id": "lrc-cross-family-thesis-audit-v1-20260718",
        "decision": decision,
        "authorization": "NONE",
        "training_used": False,
        "integrity_pass": integrity_pass,
        "controls": controls,
        "hash_controls": hash_controls,
        "observed_sha256": observed_hashes,
        "deterministic_evidence": {
            "decision": deterministic_recomputed["decision"],
            "record_count": deterministic_recomputed["record_count"],
            "interpretation": "reproducible intervention sensitivity; not unique paper thesis",
        },
        "stochastic_evidence": {
            "record_count": len(records),
            "overall_decision": pcmc_decision_recomputed["overall_decision"],
            "family_specific": family_specific,
            "soft_effectively_discrete": soft_effectively_discrete,
            "latent_grpo": {
                "a0_decision": latent_a0["decision"],
                "calibration_q75_js_nats": latent_a0["high_threshold"],
                "median_js_nats": latent_posthoc["js_nats"]["quantiles"]["q50"],
                "median_maximum_weight": latent_posthoc["maximum_weight"]["quantiles"]["q50"],
                "median_effective_support": latent_posthoc["effective_support"]["quantiles"]["q50"],
                "top_token_disagreement_rate": latent_posthoc["top_token_disagreement_rate"],
            },
            "soft_grpo": {
                "a0_decision": soft_a0["decision"],
                "calibration_q75_js_nats": soft_a0["high_threshold"],
                "median_js_nats": soft_posthoc["js_nats"]["quantiles"]["q50"],
                "median_maximum_weight": soft_posthoc["maximum_weight"]["quantiles"]["q50"],
                "median_effective_support": soft_posthoc["effective_support"]["quantiles"]["q50"],
                "top_token_disagreement_rate": soft_posthoc["top_token_disagreement_rate"],
            },
        },
        "permanent_kill_condition": {
            "clause": "policy-contract findings occur only in Latent-GRPO and disappear in SofT-GRPO or another stochastic family",
            "triggered": kill_triggered,
        },
        "next_action": "archive current thesis; collision-audit a distinct method candidate",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_current_thesis(args.workspace)
    write_json_atomic(args.output, report)
    print(json.dumps({"decision": report["decision"], "controls": report["controls"]}))
    return 0 if report["integrity_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
