"""Adapters for released latent-reasoning checkpoints used in real-model gates."""

from .coconut import (
    CoconutChartRunner,
    CoconutContinuationAdapter,
    CoconutModelBundle,
    CoconutRunResult,
    load_public_coconut,
    prepare_coconut_input,
)
from .switch import (
    PINNED_SWITCH_COMMIT,
    SwitchAuditRun,
    SwitchAuditRunner,
    SwitchDifferentiableReplay,
    SwitchLatentStep,
    SwitchModelBundle,
    SwitchReplayPlan,
    build_first_block_replay_plan,
    build_switch_prompt,
    load_public_switch,
    load_pinned_switch_types,
    verify_pinned_switch_source,
)

__all__ = [
    "CoconutChartRunner",
    "CoconutContinuationAdapter",
    "CoconutModelBundle",
    "CoconutRunResult",
    "load_public_coconut",
    "prepare_coconut_input",
    "PINNED_SWITCH_COMMIT",
    "SwitchAuditRun",
    "SwitchAuditRunner",
    "SwitchDifferentiableReplay",
    "SwitchLatentStep",
    "SwitchModelBundle",
    "SwitchReplayPlan",
    "build_first_block_replay_plan",
    "build_switch_prompt",
    "load_public_switch",
    "load_pinned_switch_types",
    "verify_pinned_switch_source",
]
