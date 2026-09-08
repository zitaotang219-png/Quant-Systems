"""Shared turnover semantics for all one-session portfolio simulations."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TurnoverConvention:
    name: str = "flat_intraday_entry_plus_exit"

    def report(self, starting_equity: float, entry_notional: float, exit_notional: float) -> dict[str, float | str]:
        if starting_equity == 0.0:
            return {
                "convention": self.name,
                "entry_turnover": 0.0,
                "exit_turnover": 0.0,
                "turnover": 0.0,
            }
        entry_turnover = abs(float(entry_notional)) / float(starting_equity)
        exit_turnover = abs(float(exit_notional)) / float(starting_equity)
        return {
            "convention": self.name,
            "entry_turnover": entry_turnover,
            "exit_turnover": exit_turnover,
            "turnover": entry_turnover + exit_turnover,
        }

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


DEFAULT_TURNOVER_CONVENTION = TurnoverConvention()


@dataclass(frozen=True)
class ContinuousRebalanceTurnoverConvention:
    """Turnover from actual delta trades in a continuously held portfolio."""

    name: str = "continuous_crypto_executed_delta"

    def report(self, starting_equity: float, executed_delta_notional: float) -> dict[str, float | str]:
        turnover = 0.0 if starting_equity == 0.0 else abs(float(executed_delta_notional)) / float(starting_equity)
        return {
            "convention": self.name,
            "executed_delta_turnover": turnover,
            "turnover": turnover,
        }

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


CONTINUOUS_CRYPTO_TURNOVER_CONVENTION = ContinuousRebalanceTurnoverConvention()
