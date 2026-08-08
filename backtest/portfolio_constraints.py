"""Portfolio exposure checks run before every event-driven execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class PortfolioConstraints:
    max_position_size: float
    max_leverage: float
    max_gross_exposure: float
    max_net_exposure: float


@dataclass(frozen=True)
class ConstraintResult:
    max_position: float
    gross_exposure: float
    net_exposure: float
    leverage: float
    passed: bool
    violations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self) | {"violations": ";".join(self.violations)}


def validate_target_weights(weights: Mapping[str, float], constraints: PortfolioConstraints) -> ConstraintResult:
    values = [float(weight) for weight in weights.values()]
    max_position = max((abs(value) for value in values), default=0.0)
    gross = sum(abs(value) for value in values)
    net = abs(sum(values))
    violations: list[str] = []
    tolerance = 1e-10
    if max_position > constraints.max_position_size + tolerance:
        violations.append("max_position_size")
    if gross > constraints.max_gross_exposure + tolerance:
        violations.append("max_gross_exposure")
    if gross > constraints.max_leverage + tolerance:
        violations.append("max_leverage")
    if net > constraints.max_net_exposure + tolerance:
        violations.append("max_net_exposure")
    return ConstraintResult(
        max_position=max_position,
        gross_exposure=gross,
        net_exposure=net,
        leverage=gross,
        passed=not violations,
        violations=tuple(violations),
    )
