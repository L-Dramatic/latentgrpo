from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from research.coordinate_invariance.real_models.switch import (
    SwitchAuditRunner,
    SwitchDifferentiableReplay,
    build_first_block_replay_plan,
    build_switch_prompt,
    load_pinned_switch_types,
    verify_pinned_switch_source,
)
from research.coordinate_invariance.switch_c2_eligibility_scan import (
    classify_switch_run,
)


class DeterministicSwitchLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(12, 8)
        generator = torch.Generator().manual_seed(71521)
        with torch.no_grad():
            self.embedding.weight.copy_(
                torch.randn(self.embedding.weight.shape, generator=generator)
            )
        self.config = SimpleNamespace(model_type="deterministic-test")
        projection_generator = torch.Generator().manual_seed(71522)
        self.register_buffer(
            "projection",
            torch.randn(8, 12, generator=projection_generator) * 0.01,
        )
        self.calls: list[dict[str, torch.Tensor]] = []

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values=None,
        use_cache: bool = True,
        output_hidden_states: bool = False,
    ) -> SimpleNamespace:
        del past_key_values, use_cache
        self.calls.append(
            {
                "inputs_embeds": inputs_embeds.detach().clone(),
                "attention_mask": attention_mask.detach().clone(),
                "position_ids": position_ids.detach().clone(),
            }
        )
        positions = position_ids.to(dtype=inputs_embeds.dtype).unsqueeze(-1)
        hidden = inputs_embeds + 0.01 * (positions + 1.0)
        logits = torch.full(
            (*inputs_embeds.shape[:2], 12),
            -100.0,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        logits = logits + hidden @ self.projection
        decisions = {1: 3, 2: 8, 3: 4, 4: 4, 5: 7, 6: 5}
        for column, position in enumerate(position_ids[0].tolist()):
            logits[0, column, decisions.get(position, 5)] = 100.0
        return SimpleNamespace(
            logits=logits,
            hidden_states=(hidden,) if output_hidden_states else None,
            past_key_values=("deterministic-cache",),
        )


def _make_models():
    switch_type, config_type = load_pinned_switch_types("_external/switch")
    token_config = config_type(
        swi_start_id=3,
        swi_end_id=4,
        latent_id=6,
        eos_token_id=5,
    )
    official_base = DeterministicSwitchLM()
    audit_base = DeterministicSwitchLM()
    audit_base.load_state_dict(official_base.state_dict())
    return (
        switch_type(official_base, token_config),
        switch_type(audit_base, token_config),
        official_base,
        audit_base,
    )


def test_pinned_source_integrity() -> None:
    record = verify_pinned_switch_source("_external/switch")
    assert record["commit"] == "d8d97cdc6276fcfa6e48f6a6b19ce472c7b87fcd"
    assert record["source_sha256"] == (
        "3bdd5e66076bbcab1c3e2ee600c16fad749839cda21616b3d094211ba4fa1b27"
    )


def test_audit_loop_matches_official_generation_call_by_call() -> None:
    official, audited, official_base, audit_base = _make_models()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    mask = torch.ones_like(prompt)

    official_ids, official_info = official.generate(
        prompt,
        attention_mask=mask,
        max_new_tokens=6,
        min_latent_steps=2,
    )
    audit = SwitchAuditRunner(audited).run(
        prompt,
        mask,
        max_new_tokens=6,
        min_latent_steps=2,
        capture_trace=True,
    )

    assert torch.equal(official_ids, audit.output_ids)
    assert official_ids.tolist() == [[1, 2, 3, 4, 7, 5]]
    assert official_info == list(audit.latent_info)
    assert audit.model_forward_calls == len(official_base.calls) == len(audit_base.calls)
    assert len(audit.latent_steps) == 2
    for expected, measured in zip(official_base.calls, audit_base.calls):
        assert expected.keys() == measured.keys()
        for name in expected:
            assert torch.equal(expected[name], measured[name]), name


def test_identity_latent_hook_is_exact_no_op() -> None:
    official, audited, official_base, audit_base = _make_models()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    mask = torch.ones_like(prompt)
    official_ids, _ = official.generate(
        prompt,
        attention_mask=mask,
        max_new_tokens=6,
        min_latent_steps=2,
    )
    audit = SwitchAuditRunner(audited).run(
        prompt,
        mask,
        max_new_tokens=6,
        min_latent_steps=2,
        operation=lambda latent, block, step: latent,
        capture_trace=True,
    )
    assert torch.equal(official_ids, audit.output_ids)
    for record in audit.latent_steps:
        assert torch.equal(record.proposed_native, record.consumed_native)
    for expected, measured in zip(official_base.calls, audit_base.calls):
        assert torch.equal(expected["inputs_embeds"], measured["inputs_embeds"])


def test_differentiable_replay_matches_factual_block_and_has_jvp() -> None:
    _, audited, _, _ = _make_models()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    run = SwitchAuditRunner(audited).run(
        prompt,
        torch.ones_like(prompt),
        max_new_tokens=6,
        min_latent_steps=2,
        capture_trace=True,
    )
    plan = build_first_block_replay_plan(
        run,
        prompt_length=2,
        swi_start_id=3,
        swi_end_id=4,
        eos_id=5,
        visible_horizon=1,
    )
    replay = SwitchDifferentiableReplay(audited, plan)
    basis = torch.eye(8, 2)
    origin = torch.zeros(2, requires_grad=True)
    latent_logits, visible_logits = replay.logits(origin, basis)
    assert torch.allclose(latent_logits, torch.stack([x.exit_logits for x in run.latent_steps]))
    assert torch.allclose(
        visible_logits[0],
        run.visible_decision_logits[plan.visible_decision_start_index],
    )

    def flattened(coefficients: torch.Tensor) -> torch.Tensor:
        latent, visible = replay.logits(coefficients, basis)
        return torch.cat([latent.flatten(), visible.flatten()])

    _, derivative = torch.autograd.functional.jvp(
        flattened,
        origin,
        torch.tensor([1.0, 0.0]),
        strict=True,
    )
    assert torch.isfinite(derivative).all()
    assert torch.linalg.vector_norm(derivative) > 0
    assert replay.prefix_evaluations == 2
    assert replay.logit_evaluations == 2
    assert replay.model_forward_calls == 8
    assert replay.visible_logit_tokens == 2
    assert replay.latent_logit_steps == 4
    assert replay.prefix_cache_builds == 1
    assert replay.prefix_cache_hits == 1


def test_differentiable_replay_can_limit_visible_gradient_horizon() -> None:
    _, audited, _, _ = _make_models()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    run = SwitchAuditRunner(audited).run(
        prompt,
        torch.ones_like(prompt),
        max_new_tokens=6,
        min_latent_steps=2,
        capture_trace=True,
    )
    plan = build_first_block_replay_plan(
        run,
        prompt_length=2,
        swi_start_id=3,
        swi_end_id=4,
        eos_id=5,
        visible_horizon=1,
    )
    replay = SwitchDifferentiableReplay(audited, plan)
    latent, visible = replay.logits(
        torch.zeros(2), torch.eye(8, 2), visible_horizon=1
    )
    assert latent.shape == (2, 12)
    assert visible.shape == (1, 12)
    try:
        replay.logits(torch.zeros(2), torch.eye(8, 2), visible_horizon=2)
    except ValueError as error:
        assert "outside the replay target" in str(error)
    else:
        raise AssertionError("invalid visible horizon was accepted")


def test_switch_prompt_matches_pinned_evaluator_with_compatibility_fallback() -> None:
    class Tokenizer:
        def __init__(self, *, reject_thinking: bool) -> None:
            self.reject_thinking = reject_thinking
            self.calls = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            if self.reject_thinking and "enable_thinking" in kwargs:
                raise TypeError("legacy tokenizer")
            return "rendered"

    suffix = "\nPlease reason step by step."
    modern = Tokenizer(reject_thinking=False)
    legacy = Tokenizer(reject_thinking=True)
    assert build_switch_prompt(modern, "question", suffix=suffix) == "rendered"
    assert modern.calls == [
        (
            [{"role": "user", "content": "question" + suffix}],
            {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": True,
            },
        )
    ]
    assert build_switch_prompt(legacy, "question", suffix=suffix) == "rendered"
    assert len(legacy.calls) == 2
    assert "enable_thinking" in legacy.calls[0][1]
    assert "enable_thinking" not in legacy.calls[1][1]


def test_eligibility_classification_builds_replay_plan_without_trace_logits() -> None:
    _, audited, _, _ = _make_models()
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    run = SwitchAuditRunner(audited).run(
        prompt,
        torch.ones_like(prompt),
        max_new_tokens=6,
        min_latent_steps=2,
        capture_trace=False,
    )
    eligibility, plan = classify_switch_run(
        run,
        prompt_length=2,
        max_new_tokens=6,
        swi_start_id=3,
        swi_end_id=4,
        eos_id=5,
        visible_horizon=1,
        reject_max_token_truncation=True,
    )
    assert eligibility == "eligible"
    assert plan is not None
    assert plan.visible_target_ids.tolist() == [7]
    assert plan.visible_decision_start_index == 1
    assert run.visible_decision_logits == ()
