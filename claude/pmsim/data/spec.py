"""What a run needs that the data does not say.

A leaf module: the repositories build a ``SimulationSpec`` and the orchestrator consumes
one, so it lives apart from both.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..policy import validate_expected_return, validate_rounding_unit

FX_QUOTES = ("base_per_usd", "usd_per_base")
LIQUID_KINDS = ("levels", "returns")


@dataclass(frozen=True)
class SimulationSpec:
    """Everything a run needs that the tables do not carry.

    **The liquid portfolio.** ``liquid_series`` names its column of market_data, and
    ``liquid_kind`` says what that column holds. With ``levels`` the series is read as an
    index — only its changes matter — and rescaled to start at the starting value.
    ``returns`` are simple per-period returns: the simulation then starts one
    period before the first return — on ``inception_date`` if given, otherwise on a date
    inferred from the series' frequency and rolled back to a business day — and every return
    is applied from there.

    **The starting value is not a setting.** Every run starts with
    ``pmsim.STARTING_VALUE``, 100,000,000, in the portfolio's own base currency. There is no
    field for another amount, and no way to start from an amount of another currency.

    **The exchange rate.** ``fx_series`` names its column of market_data, and ``fx_quote`` says
    how it is quoted: ``base_per_usd`` (GBP per 1 USD, used as it is) or ``usd_per_base``
    (USD per 1 GBP, inverted). If the series starts later than the inception date, its first
    rate is taken to apply there.

    **The fund flows.** ``calls_are_negative`` is the sign convention of ``Flow`` rows:
    negative values are calls and positive values distributions (the LP's view); set False
    for the opposite. ``Call`` and ``Distribution`` rows are read as magnitudes regardless.

    **The commitment schedule.** ``commitment_rates`` overrides the repository's rate table
    when given. ``expected_return`` is the yearly return X the pacing schedule was built on,
    as a decimal: the schedule is divided by the pacing model's expected liquid value — 1 on
    the first commitment date, growing at X — to become a share of the liquid value. Given
    here it overrides the repository's expected_returns table; with neither, the rates are
    taken to be shares of the liquid value already.

    **Which years each fund collects.** ``weights`` and ``carry_forward`` go to
    ``AnnualRatePolicy``. With carry-forward, a year in which no fund of a type closes is
    still sized — that year's rate on that year's balance — and its dollars wait for the
    next fund of the type; without it such a year is not used. ``draws`` states the schedule
    years a fund collects outright, as ``{fund name: {calendar year: multiplier}}``,
    overriding the repository's own draw plans; naming any fund of a type switches
    carry-forward off for that type.

    **Rounding.** ``commitment_rounding_unit_usd`` rounds each year's dollar commitment to the
    nearest multiple of that many dollars, halves away from zero, as Excel's ROUND does;
    10_000 is ``ROUND(value, -4)``, which is ``pmsim.COMMITMENT_ROUNDING_UNIT_USD`` and what
    ``WorkbookRepository.simulation_spec`` applies by default. None rounds nothing.

    **The run.** ``stop_on_shortfall`` ends the run at the first observation whose calls the
    liquid account cannot meet; ``cash_tolerance`` is how far short it may fall before that
    counts.
    """

    base_currency: str
    liquid_series: str
    fx_series: str | None = None
    fx_quote: str = "base_per_usd"
    liquid_kind: str = "levels"
    inception_date: Any = None
    commitment_rates: Any = None
    expected_return: float | None = None
    draws: Mapping[str, Mapping[int, float]] | None = None
    commitment_rounding_unit_usd: float | None = None
    weights: Mapping[str, float] | None = None
    carry_forward: bool = False
    calls_are_negative: bool = True
    stop_on_shortfall: bool = True
    cash_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        # The two settings that must be one of a fixed list.
        if self.fx_quote not in FX_QUOTES:
            raise ValueError(f"fx_quote must be one of {FX_QUOTES}, got {self.fx_quote!r}")

        if self.liquid_kind not in LIQUID_KINDS:
            raise ValueError(f"liquid_kind must be one of {LIQUID_KINDS}, got {self.liquid_kind!r}")

        if not isinstance(self.liquid_series, str) or not self.liquid_series.strip():
            raise ValueError("liquid_series must name a market_data column")

        # Levels carry their own first date, so there is no inception date to set.
        if self.liquid_kind == "levels" and self.inception_date is not None:
            raise ValueError("inception_date only applies when liquid_kind is 'returns'")

        if self.expected_return is not None:
            validate_expected_return(self.expected_return)

        if self.commitment_rounding_unit_usd is not None:
            validate_rounding_unit(self.commitment_rounding_unit_usd, label="commitment_rounding_unit_usd")
