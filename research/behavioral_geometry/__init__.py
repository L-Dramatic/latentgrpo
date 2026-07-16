"""Behavioral continuation geometry for recurrent latent reasoning."""

from .joint_kl import (
    DirectionalJointKL,
    JointContinuationEvaluator,
    ReferenceContinuationBatch,
    SymmetricJointKL,
)
from .p1_fake_preflight import (
    FinishEvent,
    FinishKind,
    LatentActionKind,
    LatentExecution,
    LatentRequestState,
    StopConfig,
    ToyRecursiveLatentPolicy,
    directional_jvp,
    execute_latent_action,
)
from .p1_source_sampler_contract import (
    SourceLatentSamplerConfig,
    SourceLatentSamplerResult,
    source_style_latent_action,
)
from .p1_official_sampler_replay import (
    OFFICIAL_SAMPLER_PATH,
    PINNED_OFFICIAL_SOURCE_COMMIT,
    OfficialSamplerReplay,
    replay_pinned_sampler_on_fake_request,
)
from .p1_recursive_source_adapter import (
    SOURCE_LITERAL_LATENT_END_ID,
    IsolatedSamplerRNG,
    SourceAdapterState,
    SourceClosureEndpoint,
    SourceRecursiveAdapter,
    ToySourceCacheModel,
    execute_source_scheduled_action,
    propose_source_action,
    source_weighted_embedding,
)
from .p1_official_objective_replay import (
    OFFICIAL_TORCH_FUNCTIONAL_PATH,
    replay_pinned_gumbel_likelihood,
)
from .p1_source_objective_contract import source_gumbel_likelihood_contract
from .p1_official_ppo_replay import OFFICIAL_PPO_CORE_PATH, replay_pinned_policy_loss
from .p1_source_ppo_contract import source_policy_loss_contract
from .p1_exact_law_contract import (
    ExactAutoregressiveLaw,
    enumerate_autoregressive_law,
    exact_chain_rule_kl,
    exact_joint_kl,
    full_support_probs_from_logits,
    source_visible_top_p_probs,
)
from .p1_source_advantage_contract import (
    source_actor_zero_max_length_advantages,
    source_include_advantage_contract,
)
from .p1_forward_kl_contract import (
    ContinuationEndpoint,
    EndpointPairAudit,
    PairedPathKL,
    audit_endpoint_pair,
    paired_path_kl,
)

__all__ = [
    "DirectionalJointKL",
    "JointContinuationEvaluator",
    "ReferenceContinuationBatch",
    "SymmetricJointKL",
    "FinishEvent",
    "FinishKind",
    "LatentActionKind",
    "LatentExecution",
    "LatentRequestState",
    "StopConfig",
    "ToyRecursiveLatentPolicy",
    "directional_jvp",
    "execute_latent_action",
    "SourceLatentSamplerConfig",
    "SourceLatentSamplerResult",
    "source_style_latent_action",
    "OFFICIAL_SAMPLER_PATH",
    "PINNED_OFFICIAL_SOURCE_COMMIT",
    "OfficialSamplerReplay",
    "replay_pinned_sampler_on_fake_request",
    "SOURCE_LITERAL_LATENT_END_ID",
    "IsolatedSamplerRNG",
    "SourceAdapterState",
    "SourceClosureEndpoint",
    "SourceRecursiveAdapter",
    "ToySourceCacheModel",
    "execute_source_scheduled_action",
    "propose_source_action",
    "source_weighted_embedding",
    "OFFICIAL_TORCH_FUNCTIONAL_PATH",
    "replay_pinned_gumbel_likelihood",
    "source_gumbel_likelihood_contract",
    "OFFICIAL_PPO_CORE_PATH",
    "replay_pinned_policy_loss",
    "source_policy_loss_contract",
    "ExactAutoregressiveLaw",
    "enumerate_autoregressive_law",
    "exact_chain_rule_kl",
    "exact_joint_kl",
    "full_support_probs_from_logits",
    "source_visible_top_p_probs",
    "source_actor_zero_max_length_advantages",
    "source_include_advantage_contract",
    "ContinuationEndpoint",
    "EndpointPairAudit",
    "PairedPathKL",
    "audit_endpoint_pair",
    "paired_path_kl",
]
