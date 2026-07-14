from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .analysis import continuation_rank_report


def _rank_groups(
    rows: list[dict[str, Any]],
    *,
    prefix_horizons: list[int],
    top_fraction: float,
    screen_fraction: float,
) -> dict[str, dict[str, dict[str, float | int]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        family = str(row["family"])
        groups.setdefault(family, []).append(row)
        groups.setdefault(f"{family}:{float(row['level']):g}", []).append(row)
    return {
        name: continuation_rank_report(
            members,
            prefix_horizons=prefix_horizons,
            top_fraction=top_fraction,
            predictor_screen_fraction=screen_fraction,
        )
        for name, members in sorted(groups.items())
        if len(members) >= 2
    }


def reanalyze(
    source_path: Path,
    *,
    prefix_horizons: list[int],
) -> dict[str, Any]:
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    rows = source.get("candidates")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("source artifact does not contain candidate rows")
    source_config = source.get("config", {})
    top_fraction = float(source_config.get("ranking_top_fraction", 0.2))
    screen_fraction = float(source_config.get("ranking_screen_fraction", 0.5))
    reference_text = [
        text
        for prompt in source.get("prompts", [])
        for text in prompt.get("reference_text", [])
        if isinstance(text, str)
    ]
    fixed_hash_prefix = all(text.startswith("###") for text in reference_text)
    return {
        "schema_version": 1,
        "analysis_name": f"{source.get('experiment_name', source_path.stem)}-posthoc-prefix-audit",
        "status": "complete",
        "scientific_evidence": False,
        "evidence_level": "post-hoc exploratory confound audit",
        "source_artifact": str(source_path.as_posix()),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "prefix_horizons": prefix_horizons,
        "protocol_audit": {
            "reference_continuation_count": len(reference_text),
            "all_reference_text_starts_with_hash_delimiter": fixed_hash_prefix,
            "interpretation": (
                "H=1 is a formatting-token baseline when every continuation begins "
                "with the same delimiter. H=2 and longer prefixes are required before "
                "claiming delayed behavioral sensitivity."
            ),
        },
        "pooled_rank": continuation_rank_report(
            rows,
            prefix_horizons=prefix_horizons,
            top_fraction=top_fraction,
            predictor_screen_fraction=screen_fraction,
        ),
        "grouped_rank": _rank_groups(
            rows,
            prefix_horizons=prefix_horizons,
            top_fraction=top_fraction,
            screen_fraction=screen_fraction,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reanalyze a BCG pilot with stronger prefix-KL baselines"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prefix-horizons", type=int, nargs="+", default=[1, 2, 3]
    )
    arguments = parser.parse_args()
    result = reanalyze(
        arguments.input.resolve(), prefix_horizons=arguments.prefix_horizons
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
