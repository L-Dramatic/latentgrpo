"""Tools for auditing coordinate dependence in continuous latent policies."""

from .charts import AffineChart, ChartDiagnostics, LatentChart, SinhChart
from .accounting import ComputeLedger, ComputeSnapshot
from .adapters import (
    ChartedThoughtTrajectory,
    LatentGRPOChartAdapter,
    LatentGRPOContinuationAdapter,
)
from .evaluator import (
    ContinuationComparison,
    ContinuationEvaluation,
    ContinuationEvaluator,
    ContinuationOutput,
)
from .metrics import (
    categorical_divergence_from_logits,
    categorical_kl_from_logits,
    multi_horizon_divergence,
    stable_categorical_kl_from_logits,
)
from .operations import (
    NeighborResult,
    add_isotropic_noise_in_chart,
    coordinate_distances,
    interpolate_in_chart,
    nearest_neighbor_in_chart,
)
from .trace import LatentStepRecord, LatentTrace

__all__ = [
    "AffineChart",
    "ChartDiagnostics",
    "ChartedThoughtTrajectory",
    "ComputeLedger",
    "ComputeSnapshot",
    "ContinuationComparison",
    "ContinuationEvaluation",
    "ContinuationEvaluator",
    "ContinuationOutput",
    "LatentStepRecord",
    "LatentGRPOChartAdapter",
    "LatentGRPOContinuationAdapter",
    "LatentChart",
    "LatentTrace",
    "NeighborResult",
    "SinhChart",
    "add_isotropic_noise_in_chart",
    "categorical_divergence_from_logits",
    "categorical_kl_from_logits",
    "coordinate_distances",
    "interpolate_in_chart",
    "multi_horizon_divergence",
    "stable_categorical_kl_from_logits",
    "nearest_neighbor_in_chart",
]
