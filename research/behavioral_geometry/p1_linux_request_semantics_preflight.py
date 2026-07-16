"""Linux source-runtime request-semantic gate for the P1 checkpoint repair.

This program executes the pinned SGLang Req and SamplingParams implementations
inside the isolated Linux runtime. It uses only the local tokenizer and
synthetic token ids; it does not load model weights, prompts from a dataset,
calibration data, or held-out data.

Run only under the pinned WSL environment:
    /home/lixingshuo/.venvs/latentgrpo-py311/bin/python -m
    research.behavioral_geometry.p1_linux_request_semantics_preflight
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoTokenizer, __version__ as transformers_version

from sglang.srt.hf_transformers_utils import attach_additional_stop_token_ids
from sglang.srt.managers.schedule_batch import (
    FINISH_MATCHED_STR,
    FINISH_MATCHED_TOKEN,
    Req,
)
from sglang.srt.sampling.sampling_params import SamplingParams


MODEL_DIR = Path("/mnt/e/LantentGRPO/_models/Latent-GRPO-Llama-1B")
SOURCE_END_ID = 524
MODEL_VOCAB_SIZE = 128256
FORBIDDEN_TOKENIZER_ONLY_ID = 128256


@dataclass
class RequestSemanticsResult:
    status: str
    runtime: dict[str, Any]
    tokenizer: dict[str, Any]
    scheduler_order: dict[str, Any]
    cases: dict[str, Any]
    failures: list[str]


def _make_req(
    tokenizer,
    *,
    rid: str,
    stop_token_ids: list[int] | None = None,
    stop: list[str] | None = None,
    ignore_eos: bool = False,
) -> Req:
    params = SamplingParams(
        max_new_tokens=8,
        stop=stop,
        stop_token_ids=stop_token_ids,
        ignore_eos=ignore_eos,
    )
    params.verify()
    params.normalize(tokenizer)
    req = Req(
        rid=rid,
        origin_input_text="synthetic",
        origin_input_ids=(int(tokenizer.bos_token_id),),
        sampling_params=params,
        eos_token_ids={int(tokenizer.eos_token_id)},
        enable_latent=True,
        latent_end_token_id=SOURCE_END_ID,
        max_topk=3,
    )
    req.tokenizer = tokenizer
    return req


def _fake_logits_output() -> SimpleNamespace:
    """Only the fields read by the unmodified Req.update_latent_info body."""

    return SimpleNamespace(
        topk_gumbels=torch.tensor([[0.1, 0.2, 0.3]], device="cuda"),
        topk_original_probs=torch.tensor([[0.7, 0.2, 0.1]], device="cuda"),
        topk_original_indices=torch.tensor([[12, 15, 19]], device="cuda"),
        topk_probs=torch.tensor([[0.7, 0.2, 0.1]], device="cuda"),
        topk_indices=torch.tensor([[12, 15, 19]], device="cuda"),
    )


def _finish_kind(req: Req) -> str | None:
    if req.finished_reason is None:
        return None
    if isinstance(req.finished_reason, FINISH_MATCHED_TOKEN):
        return "matched_token"
    if isinstance(req.finished_reason, FINISH_MATCHED_STR):
        return "matched_string"
    return type(req.finished_reason).__name__


def _run_case(req: Req, emitted_id: int, *, call_update_after_finish: bool) -> dict[str, Any]:
    """Reproduce the source decode order: append, check, then update."""

    req.output_ids.append(emitted_id)
    req.check_finished()
    finished_after_check = req.finished()
    finish_kind_after_check = _finish_kind(req)
    matched_after_check = (
        getattr(req.finished_reason, "matched", None) if req.finished_reason is not None else None
    )
    if call_update_after_finish:
        req.update_latent_info(_fake_logits_output(), 0)
    latent_mode = req.sampling_params.latent_mode
    if isinstance(latent_mode, torch.Tensor):
        latent_mode = bool(latent_mode.item())
    return {
        "emitted_id": emitted_id,
        "decoded_emitted_id": req.tokenizer.decode([emitted_id]),
        "finished_after_check": finished_after_check,
        "finish_kind_after_check": finish_kind_after_check,
        "matched_after_check": matched_after_check,
        "update_called_after_check": call_update_after_finish,
        "latent_mode_after_update": latent_mode,
        "topk_probs_after_update": req.topk_prob.detach().float().cpu().tolist(),
        "topk_ids_after_update": req.topk_idx.detach().cpu().tolist(),
    }


def run_preflight() -> RequestSemanticsResult:
    failures: list[str] = []
    runtime = {
        "torch": torch.__version__,
        "transformers": transformers_version,
        "cuda_available": torch.cuda.is_available(),
        "model_dir": str(MODEL_DIR),
    }
    if not torch.cuda.is_available():
        return RequestSemanticsResult(
            status="BLOCKED_NO_CUDA",
            runtime=runtime,
            tokenizer={},
            scheduler_order={},
            cases={},
            failures=["official latent-mode request constructor requires CUDA"],
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=True)
    attach_additional_stop_token_ids(tokenizer)
    marker_ids = tokenizer("</think>", add_special_tokens=False)["input_ids"]
    tokenizer_info = {
        "len_tokenizer": len(tokenizer),
        "model_vocab_size": MODEL_VOCAB_SIZE,
        "marker_ids": marker_ids,
        "source_end_id": SOURCE_END_ID,
        "source_end_decode": tokenizer.decode([SOURCE_END_ID]),
        "eos_token_id": tokenizer.eos_token_id,
        "all_special_ids": tokenizer.all_special_ids,
        "additional_stop_token_ids": sorted(tokenizer.additional_stop_token_ids)
        if tokenizer.additional_stop_token_ids
        else [],
        "forbidden_tokenizer_only_id": FORBIDDEN_TOKENIZER_ONLY_ID,
        "forbidden_id_decode": tokenizer.decode([FORBIDDEN_TOKENIZER_ONLY_ID]),
    }
    if marker_ids[:1] != [SOURCE_END_ID]:
        failures.append("the source end id does not match the </think> prefix")
    if SOURCE_END_ID == tokenizer.eos_token_id:
        failures.append("524 is the tokenizer EOS id")
    if SOURCE_END_ID in tokenizer.all_special_ids:
        failures.append("524 is a tokenizer special id")
    if SOURCE_END_ID in tokenizer_info["additional_stop_token_ids"]:
        failures.append("524 is an additional tokenizer stop id")

    scheduler_order = {
        "source_file": "sglang/srt/managers/scheduler_output_processor_mixin.py",
        "decode_order": ["append(next_token_id)", "check_finished()", "update_latent_info(...)"],
        "update_called_even_if_finished": True,
        "meaning": (
            "generic finish prevents another decode step, but the pinned scheduler still "
            "mutates this terminal request's latent bookkeeping once after check_finished"
        ),
    }

    cases = {
        "nonstopping_524": _run_case(
            _make_req(tokenizer, rid="nonstopping_524"),
            SOURCE_END_ID,
            call_update_after_finish=True,
        ),
        "stop_token_524": _run_case(
            _make_req(tokenizer, rid="stop_token_524", stop_token_ids=[SOURCE_END_ID]),
            SOURCE_END_ID,
            call_update_after_finish=True,
        ),
        "stop_string_prefix": _run_case(
            _make_req(tokenizer, rid="stop_string_prefix", stop=["</"]),
            SOURCE_END_ID,
            call_update_after_finish=True,
        ),
        "ignore_eos_token_stop": _run_case(
            _make_req(
                tokenizer,
                rid="ignore_eos_token_stop",
                stop_token_ids=[SOURCE_END_ID],
                ignore_eos=True,
            ),
            SOURCE_END_ID,
            call_update_after_finish=True,
        ),
        "model_eos": _run_case(
            _make_req(tokenizer, rid="model_eos"),
            int(tokenizer.eos_token_id),
            call_update_after_finish=True,
        ),
    }

    normal = cases["nonstopping_524"]
    if normal["finished_after_check"]:
        failures.append("a default request treats 524 as terminal")
    if normal["latent_mode_after_update"]:
        failures.append("nonstopping 524 did not exit latent mode")
    if normal["topk_ids_after_update"] != [SOURCE_END_ID, -100, -100]:
        failures.append("nonstopping 524 did not force the hard end-token mixture")
    if normal["topk_probs_after_update"] != [1.0, 0.0, 0.0]:
        failures.append("nonstopping 524 did not force one-hot end-token probability")

    token_stop = cases["stop_token_524"]
    if not token_stop["finished_after_check"] or token_stop["finish_kind_after_check"] != "matched_token":
        failures.append("explicit stop-token 524 did not finish in actual Req.check_finished")

    string_stop = cases["stop_string_prefix"]
    if not string_stop["finished_after_check"] or string_stop["finish_kind_after_check"] != "matched_string":
        failures.append("decoded </ stop string did not finish in actual Req.check_finished")

    ignored = cases["ignore_eos_token_stop"]
    if ignored["finished_after_check"]:
        failures.append("ignore_eos did not disable explicit token-stop handling")

    model_eos = cases["model_eos"]
    if not model_eos["finished_after_check"] or model_eos["finish_kind_after_check"] != "matched_token":
        failures.append("model EOS did not finish in actual Req.check_finished")

    return RequestSemanticsResult(
        status="PASS" if not failures else "FAIL",
        runtime=runtime,
        tokenizer=tokenizer_info,
        scheduler_order=scheduler_order,
        cases=cases,
        failures=failures,
    )


if __name__ == "__main__":
    print(json.dumps(asdict(run_preflight()), ensure_ascii=False, indent=2, sort_keys=True))

