from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


@dataclass(frozen=True)
class SelectedRow:
    dataset_index: int
    selection_hash: str
    row: dict[str, Any]


def canonical_config_hash(raw: dict[str, Any]) -> str:
    payload = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def select_rows(dataset: Sequence[Any], selection: dict[str, Any]) -> list[SelectedRow]:
    count = int(selection["count"])
    key = str(selection["key"])
    salt = str(selection["salt"])
    if count < 1 or count > len(dataset):
        raise ValueError("selection count must be within the dataset")

    ranked: list[SelectedRow] = []
    for dataset_index in range(len(dataset)):
        row = dict(dataset[dataset_index])
        if key not in row:
            raise KeyError(f"selection key {key!r} is missing")
        digest = hashlib.sha256(
            (salt + "\0" + str(row[key])).encode("utf-8")
        ).hexdigest()
        ranked.append(
            SelectedRow(
                dataset_index=dataset_index,
                selection_hash=digest,
                row=row,
            )
        )
    ranked.sort(key=lambda item: (item.selection_hash, item.dataset_index))
    return ranked[:count]


def prompt_text(tokenizer: Any, problem: str, contract: dict[str, Any]) -> str:
    system_prompt = str(contract["system_prompt"])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem + "\n" + system_prompt},
    ]
    if contract["apply_model_chat_template"]:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return "\n".join(message["content"] for message in messages)


def load_model_and_tokenizer(
    raw: dict[str, Any], model_dir: Path
) -> tuple[Any, Any]:
    runtime = raw["runtime"]
    if runtime["preferred_device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("the frozen run requires CUDA but CUDA is unavailable")

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = getattr(torch, str(raw["model"]["dtype"]))
    max_memory = {
        0: f"{int(runtime['max_gpu_memory_mib'])}MiB",
        "cpu": f"{int(runtime['max_cpu_memory_gib'])}GiB",
    }
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory=max_memory,
        offload_state_dict=True,
    )
    model.eval()
    return model, tokenizer


def repeat_dynamic_cache(past_key_values: Any, repeats: int) -> DynamicCache:
    if repeats < 1:
        raise ValueError("cache repeat count must be positive")
    if not hasattr(past_key_values, "to_legacy_cache"):
        raise TypeError("Stage B requires a Transformers Cache instance")
    legacy = past_key_values.to_legacy_cache()
    repeated = tuple(
        tuple(tensor.repeat_interleave(repeats, dim=0) for tensor in layer)
        for layer in legacy
    )
    return DynamicCache.from_legacy_cache(repeated)


def quantile(values: Iterable[float], probability: float) -> float:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    if tensor.numel() == 0:
        raise ValueError("cannot summarize an empty collection")
    return float(torch.quantile(tensor, probability))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def runtime_metadata(seconds: float) -> dict[str, Any]:
    return {
        "seconds": seconds,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "datasets": __import__("datasets").__version__,
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "platform": platform.platform(),
    }
