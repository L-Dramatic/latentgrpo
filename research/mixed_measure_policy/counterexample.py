"""Small exact counterexamples for clipped-Gumbel latent actions.

The official sampler clips Gumbel noise before adding policy-dependent token
scores.  Even with two tokens and no Top-K selection, the resulting mixture
weight has policy-dependent point-mass locations.  This module keeps that
minimal case separate from any proposed optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ClippedGumbelMasses:
    lower: float
    upper: float
    lower_mass: float
    upper_mass: float

    @property
    def boundary_mass(self) -> float:
        return self.lower_mass + self.upper_mass


@dataclass(frozen=True)
class WeightAtom:
    location: float
    mass: float
    events: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class AtomDominanceAudit:
    old_only_mass: float
    new_only_mass: float
    old_atoms: tuple[WeightAtom, ...]
    new_atoms: tuple[WeightAtom, ...]

    @property
    def has_bidirectional_singularity(self) -> bool:
        return self.old_only_mass > 0.0 and self.new_only_mass > 0.0


@dataclass(frozen=True)
class LogMeasureValue:
    component: str
    log_value: float


def _validate_bounds(lower: float, upper: float) -> None:
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError("expected finite clipping bounds with lower < upper")


def _gumbel_cdf(value: float) -> float:
    return math.exp(-math.exp(-value))


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        decay = math.exp(-value)
        return 1.0 / (1.0 + decay)
    growth = math.exp(value)
    return growth / (1.0 + growth)


def clipped_gumbel_masses(
    lower: float = -1.5,
    upper: float = 3.0,
) -> ClippedGumbelMasses:
    """Return the exact lower and upper masses after hard clipping."""

    _validate_bounds(lower, upper)
    lower_mass = _gumbel_cdf(lower)
    upper_mass = -math.expm1(-math.exp(-upper))
    return ClippedGumbelMasses(lower, upper, lower_mass, upper_mass)


def standard_gumbel_log_density(value: float) -> float:
    """Log density used by an interior, unclipped standard Gumbel law."""

    return -value - math.exp(-value)


def clipped_gumbel_log_measure(
    value: float,
    lower: float = -1.5,
    upper: float = 3.0,
    *,
    atol: float = 1e-12,
) -> LogMeasureValue:
    """Evaluate the correct mixed-measure component at ``value``.

    Boundary values are probabilities with respect to Dirac measures.  Values
    in the open interval are densities with respect to Lebesgue measure.
    These quantities must not be substituted for one another.
    """

    _validate_bounds(lower, upper)
    if value < lower - atol or value > upper + atol:
        return LogMeasureValue("outside_support", -math.inf)
    if math.isclose(value, lower, rel_tol=0.0, abs_tol=atol):
        return LogMeasureValue("lower_atom", -math.exp(-lower))
    if math.isclose(value, upper, rel_tol=0.0, abs_tol=atol):
        upper_mass = -math.expm1(-math.exp(-upper))
        return LogMeasureValue("upper_atom", math.log(upper_mass))
    return LogMeasureValue("interior_density", standard_gumbel_log_density(value))


def two_token_weight_atoms(
    logit_gap: float,
    *,
    noise_scale: float = 1.0,
    temperature: float = 1.0,
    lower: float = -1.5,
    upper: float = 3.0,
) -> tuple[WeightAtom, ...]:
    """Return all point masses of a two-token executed mixture weight.

    For clipped noises C1 and C2, the first-token weight is

        sigmoid((logit_gap + noise_scale * (C1 - C2)) / temperature).

    Both noises landing on clipping atoms creates three distinct weight atoms.
    Their locations move whenever the policy logit gap changes.
    """

    _validate_bounds(lower, upper)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not math.isfinite(noise_scale):
        raise ValueError("noise_scale must be finite")

    masses = clipped_gumbel_masses(lower, upper)
    shared_location = _sigmoid(logit_gap / temperature)
    lower_upper = _sigmoid(
        (logit_gap + noise_scale * (lower - upper)) / temperature
    )
    upper_lower = _sigmoid(
        (logit_gap + noise_scale * (upper - lower)) / temperature
    )
    atoms = (
        WeightAtom(
            lower_upper,
            masses.lower_mass * masses.upper_mass,
            (("lower", "upper"),),
        ),
        WeightAtom(
            shared_location,
            masses.lower_mass**2 + masses.upper_mass**2,
            (("lower", "lower"), ("upper", "upper")),
        ),
        WeightAtom(
            upper_lower,
            masses.upper_mass * masses.lower_mass,
            (("upper", "lower"),),
        ),
    )
    return tuple(sorted(atoms, key=lambda atom: atom.location))


def moving_atom_dominance(
    old_logit_gap: float,
    new_logit_gap: float,
    *,
    noise_scale: float = 1.0,
    temperature: float = 1.0,
    lower: float = -1.5,
    upper: float = 3.0,
    atol: float = 1e-12,
) -> AtomDominanceAudit:
    """Measure point-mass support absent from the opposing policy law."""

    old_atoms = two_token_weight_atoms(
        old_logit_gap,
        noise_scale=noise_scale,
        temperature=temperature,
        lower=lower,
        upper=upper,
    )
    new_atoms = two_token_weight_atoms(
        new_logit_gap,
        noise_scale=noise_scale,
        temperature=temperature,
        lower=lower,
        upper=upper,
    )

    def unmatched_mass(
        source: tuple[WeightAtom, ...], target: tuple[WeightAtom, ...]
    ) -> float:
        return sum(
            atom.mass
            for atom in source
            if not any(
                math.isclose(
                    atom.location,
                    candidate.location,
                    rel_tol=0.0,
                    abs_tol=atol,
                )
                for candidate in target
            )
        )

    return AtomDominanceAudit(
        old_only_mass=unmatched_mass(old_atoms, new_atoms),
        new_only_mass=unmatched_mass(new_atoms, old_atoms),
        old_atoms=old_atoms,
        new_atoms=new_atoms,
    )
