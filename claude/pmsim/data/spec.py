"""What a run needs that the data does not say.

A leaf module: the repositories build a ``SimulationSpec`` and the orchestrator consumes
one, so it lives apart from both.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping

FX_QUOTES = ("base_per_usd", "usd_per_base")
LIQUID_KINDS = ("levels", "returns")


@dataclass(frozen=True)
class SimulationSpec:
    """Everything a run needs that the tables do not carry.

    ``liquid_series`` and ``fx_series`` name columns of market_data. ``liquid_kind`` says
    what the liquid column holds: ``levels`` (used as is; the first level is the starting
    balance) or ``returns`` (simple per-period returns). With returns, the simulation
    starts one period before the first return — ``inception_date`` if given, otherwise
    inferred from the series' frequency and rolled back to a business day — where the
    balance is ``initial_value``; every return is then applied. If the FX series starts
    later than that inception date, its first rate is taken to apply there.
    ``fx_quote`` says how the rate is quoted: ``base_per_usd`` (GBP per 1 USD, used as is)
    or ``usd_per_base`` (USD per 1 GBP, inverted). ``commitment_rates`` overrides the
    repository's rate table when given. ``calls_are_negative`` is the sign convention of
    ``Flow`` rows: negative values are calls and positive values distributions (the LP's
    view); set False for the opposite. ``Call`` and ``Distribution`` rows are read as
    magnitudes regardless. ``weights`` and ``carry_forward`` go to ``AnnualRatePolicy``: with
    carry-forward, a year in which no fund of a type closes is still sized — that year's
    rate on that year's balance — and its dollars wait for the next fund of the type;
    without it such a year is not used.
    """

    base_currency: str
    liquid_series: str
    fx_series: str | None = None
    fx_quote: str = "base_per_usd"
    liquid_kind: str = "levels"
    initial_value: float | None = None
    inception_date: Any = None
    commitment_rates: Any = None
    weights: Mapping[str, float] | None = None
    carry_forward: bool = False
    calls_are_negative: bool = True
    stop_on_shortfall: bool = True
    cash_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        if self.fx_quote not in FX_QUOTES:
            raise ValueError(f"fx_quote must be one of {FX_QUOTES}, got {self.fx_quote!r}")
        if self.liquid_kind not in LIQUID_KINDS:
            raise ValueError(f"liquid_kind must be one of {LIQUID_KINDS}, got {self.liquid_kind!r}")
        if not isinstance(self.liquid_series, str) or not self.liquid_series.strip():
            raise ValueError("liquid_series must name a market_data column")
        if self.liquid_kind == "returns":
            value = self.initial_value  # numbers.Real: numpy scalars count, bool is excluded explicitly
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError("initial_value (the starting liquid balance) must be a positive number when liquid_kind is 'returns'")
        elif self.inception_date is not None:
            raise ValueError("inception_date only applies when liquid_kind is 'returns'")
