from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable


def _mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("cannot average an empty collection")
    return sum(materialized) / len(materialized)


def _quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize an empty collection")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def build_complete_panel(
    records: list[dict[str, Any]],
    *,
    expected_states: int,
    expected_temperatures: list[float],
) -> tuple[list[int], list[float], dict[int, dict[float, dict[str, Any]]]]:
    temperatures = [float(value) for value in expected_temperatures]
    if len(set(temperatures)) != len(temperatures):
        raise ValueError("expected temperatures must be unique")
    panel: dict[int, dict[float, dict[str, Any]]] = {}
    for record in records:
        state = int(record["dataset_index"])
        temperature = float(record["temperature"])
        if temperature not in temperatures:
            raise ValueError(f"unexpected temperature {temperature}")
        state_records = panel.setdefault(state, {})
        if temperature in state_records:
            raise ValueError(f"duplicate state-temperature record: {state}, {temperature}")
        state_records[temperature] = record

    states = sorted(panel)
    if len(states) != expected_states:
        raise ValueError(f"expected {expected_states} states, found {len(states)}")
    for state in states:
        missing = [value for value in temperatures if value not in panel[state]]
        if missing:
            raise ValueError(f"state {state} is missing temperatures {missing}")
    expected_records = expected_states * len(temperatures)
    if len(records) != expected_records:
        raise ValueError(f"expected {expected_records} records, found {len(records)}")
    return states, temperatures, panel


def cluster_bootstrap(
    *,
    states: list[int],
    temperatures: list[float],
    panel: dict[int, dict[float, dict[str, Any]]],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    rng = random.Random(seed)
    gain_samples: list[float] = []
    positive_rate_advantage_samples: list[float] = []
    temperature_gain_samples = {temperature: [] for temperature in temperatures}

    for _ in range(replicates):
        sampled_states = [rng.choice(states) for _ in states]
        sampled_records = [
            panel[state][temperature]
            for state in sampled_states
            for temperature in temperatures
        ]
        gain_samples.append(
            _mean(record["gate_gain_difference"] for record in sampled_records)
        )
        positive_rate_advantage_samples.append(
            _mean(record["gate_exact_positive"] for record in sampled_records)
            - _mean(record["gate_surrogate_positive"] for record in sampled_records)
        )
        for temperature in temperatures:
            temperature_gain_samples[temperature].append(
                _mean(
                    panel[state][temperature]["gate_gain_difference"]
                    for state in sampled_states
                )
            )

    def interval(values: list[float]) -> dict[str, float]:
        return {
            "l95": _quantile(values, 0.025),
            "median": _quantile(values, 0.5),
            "u95": _quantile(values, 0.975),
        }

    return {
        "replicates": replicates,
        "seed": seed,
        "gain_difference": interval(gain_samples),
        "positive_rate_advantage": interval(positive_rate_advantage_samples),
        "gain_difference_by_temperature": {
            str(temperature): interval(temperature_gain_samples[temperature])
            for temperature in temperatures
        },
    }


def analyze(
    *,
    records: list[dict[str, Any]],
    config: dict[str, Any],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    expected_states = int(config["selection"]["count"])
    expected_temperatures = [
        float(value) for value in config["sampler"]["temperatures"]
    ]
    states, temperatures, panel = build_complete_panel(
        records,
        expected_states=expected_states,
        expected_temperatures=expected_temperatures,
    )
    ordered_records = [
        panel[state][temperature]
        for state in states
        for temperature in temperatures
    ]
    gain_differences = [
        float(record["gate_gain_difference"]) for record in ordered_records
    ]
    exact_positive_rate = _mean(
        record["gate_exact_positive"] for record in ordered_records
    )
    surrogate_positive_rate = _mean(
        record["gate_surrogate_positive"] for record in ordered_records
    )
    point_estimates = {
        "gain_difference_mean": _mean(gain_differences),
        "exact_positive_rate": exact_positive_rate,
        "surrogate_positive_rate": surrogate_positive_rate,
        "positive_rate_advantage": exact_positive_rate - surrogate_positive_rate,
        "gain_sign_disagreement_rate": _mean(
            record["gate_gain_sign_disagreement"] for record in ordered_records
        ),
        "gain_difference_by_temperature": {
            str(temperature): _mean(
                panel[state][temperature]["gate_gain_difference"] for state in states
            )
            for temperature in temperatures
        },
    }
    bootstrap = cluster_bootstrap(
        states=states,
        temperatures=temperatures,
        panel=panel,
        replicates=replicates,
        seed=seed,
    )
    gates = config["gates"]
    return {
        "status": "post_hoc_non_gating_robustness",
        "interpretation": (
            "Prompt-clustered bootstrap keeps every temperature for a prompt "
            "together. These flags do not replace the frozen Stage B2 gate."
        ),
        "state_count": len(states),
        "temperatures": temperatures,
        "record_count": len(ordered_records),
        "point_estimates": point_estimates,
        "prompt_cluster_bootstrap": bootstrap,
        "robustness_flags": {
            "paired_gain_mean_clears_frozen_threshold": point_estimates[
                "gain_difference_mean"
            ]
            >= float(gates["exact_minus_surrogate_gain_mean_min"]),
            "paired_gain_cluster_l95_above_zero": bootstrap["gain_difference"][
                "l95"
            ]
            > 0.0,
            "positive_rate_advantage_clears_frozen_threshold": point_estimates[
                "positive_rate_advantage"
            ]
            >= float(gates["exact_positive_rate_advantage_min"]),
            "positive_rate_advantage_cluster_l95_above_zero": bootstrap[
                "positive_rate_advantage"
            ]["l95"]
            > 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=714_504)
    args = parser.parse_args()

    records_bytes = args.records.read_bytes()
    records = load_records(args.records)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = analyze(
        records=records,
        config=config,
        replicates=args.replicates,
        seed=args.seed,
    )
    report["records_path"] = str(args.records)
    report["records_sha256"] = hashlib.sha256(records_bytes).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
