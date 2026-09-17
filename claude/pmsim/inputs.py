"""Input objects, split by who holds the data.

``Fund`` is what the fund knows: its type, its closing date, and its realized history per
$1 committed — calls, distributions and NAV marks — always in US dollars.

``Portfolio`` is what the portfolio knows: its base currency, the liquid total-return
index in that currency, the annual commitment rates by fund type, and the price of a
dollar when the base currency is not USD.

Both are frozen and copy their inputs at construction. A simulation never mutates them;
the dollar commitment to each fund is the engine's decision and lives in the result.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

from .dates import as_date, coerce_dated_series
from .timeline import AlignedFundHistory, Timeline

PRIVATE_CURRENCY = "USD"  # fund flows and NAV marks are always in dollars
UNIT_NAV_TOLERANCE = 1e-12  # floating-point slack before a negative unit NAV is a data error
YEAR_RANGE = (1900, 9999)


def _set(obj: Any, name: str, value: Any) -> None:
    object.__setattr__(obj, name, value)


@dataclass(frozen=True, eq=False)
class Fund:
    """Fund-held data: identity, closing date, and unit history in USD per $1 committed.

    ``unit_calls`` and ``unit_distributions`` are gross, non-negative, dated magnitudes;
    keep them separate even when they fall on the same day so gross figures survive.
    ``unit_nav`` holds dated NAV marks (at most one per day). Each may be given as a Series
    indexed by dates, a ``{date: value}`` mapping, an iterable of ``(date, value)`` pairs,
    or omitted. Nothing may be dated before the closing.
    """

    name: str
    fund_type: str
    closing_date: Any
    unit_calls: Any = None
    unit_distributions: Any = None
    unit_nav: Any = None

    def __post_init__(self) -> None:
        for label in ("name", "fund_type"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Fund {label} must be a non-empty string")
            _set(self, label, value.strip())
        try:
            _set(self, "closing_date", as_date(self.closing_date))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Fund {self.name!r}: closing_date: {exc}") from None
        closing = pd.Timestamp(self.closing_date)
        for label, sum_same_day in (("unit_calls", True), ("unit_distributions", True), ("unit_nav", False)):
            series = coerce_dated_series(getattr(self, label), name=f"{self.name} {label}", sum_same_day=sum_same_day)
            if (series < 0).any():
                first = series.index[series < 0][0].date()
                raise ValueError(f"Fund {self.name!r}: {label} on {first} is negative; supply gross magnitudes")
            early = series.index[series.index < closing]
            if len(early):
                raise ValueError(
                    f"Fund {self.name!r}: {label} dated {early[0].date()} precedes the closing on {self.closing_date}"
                )
            _set(self, label, series)

    @property
    def closing_year(self) -> int:
        return self.closing_date.year

    def events_by_day(self) -> list[tuple[pd.Timestamp, float, float, float | None]]:
        """``(day, call, distribution, mark or None)`` for every day with activity, in date order."""
        days = self.unit_calls.index.union(self.unit_distributions.index).union(self.unit_nav.index)
        return [
            (
                day,
                float(self.unit_calls.get(day, 0.0)),
                float(self.unit_distributions.get(day, 0.0)),
                float(self.unit_nav[day]) if day in self.unit_nav.index else None,
            )
            for day in days
        ]

    def align_history(self, timeline: Timeline) -> AlignedFundHistory:
        """Bucket this fund's dated history onto ``timeline`` and rebuild its unit NAV.

        Flows dated in ``(dates[t-1], dates[t]]`` are summed into period ``t``; the first
        period takes everything on or before ``dates[0]``. Unit NAV is rebuilt by walking
        events in date order from zero: a call adds, a distribution subtracts, and a mark
        replaces the running value (marks are taken to include that day's flows). ``nav[t]``
        is the running value after the last event on or before ``dates[t]``, so a mark
        between observations counts and a missing mark leaves the cash-adjusted estimate.
        Events after the last observation are ignored. A negative running value is a data
        error naming the fund and the day.
        """
        events = self.events_by_day()
        n = timeline.n_observations
        calls, distributions, nav = np.zeros(n), np.zeros(n), np.zeros(n)
        j, running = 0, 0.0
        for t, observation in enumerate(timeline.dates):
            while j < len(events) and events[j][0] <= observation:
                day, call, distribution, mark = events[j]
                running = mark if mark is not None else running + call - distribution
                if running < -UNIT_NAV_TOLERANCE:
                    raise ValueError(
                        f"Fund {self.name!r}: unit NAV would be {running:.6g} on {day.date()}; "
                        "supply a NAV mark on or before that date"
                    )
                calls[t] += call
                distributions[t] += distribution
                j += 1
            nav[t] = max(running, 0.0)
        return AlignedFundHistory(calls, distributions, nav, timeline.first_observation_on_or_after(self.closing_date))

    def __repr__(self) -> str:
        return (
            f"Fund(name={self.name!r}, fund_type={self.fund_type!r}, closing_date={self.closing_date}, "
            f"calls={len(self.unit_calls)}, distributions={len(self.unit_distributions)}, marks={len(self.unit_nav)})"
        )


def coerce_rate_table(value: Any) -> pd.DataFrame:
    """Coerce commitment rates to a DataFrame indexed by contiguous calendar years, fund types as columns.

    Accepts a DataFrame or anything ``pd.DataFrame`` accepts, e.g. ``{"BUYOUT": {2027: 0.10}}``.
    Rates are unit commitments per 1 of sizing base (0.10 means commit 10%), finite and
    non-negative. Every year between the first and the last must be present — use 0 for a
    year with no target rather than leaving it out.
    """
    table = value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    if table.index.has_duplicates or table.columns.has_duplicates:
        raise ValueError("commitment_rates must have unique years and unique fund types")
    years = []
    for year in table.index:
        if isinstance(year, bool) or not isinstance(year, Integral):
            raise ValueError(f"commitment_rates must be indexed by integer calendar years, got {year!r}")
        if not YEAR_RANGE[0] <= int(year) <= YEAR_RANGE[1]:
            raise ValueError(f"commitment_rates year {year} is outside {YEAR_RANGE[0]}..{YEAR_RANGE[1]}")
        years.append(int(year))
    columns = []
    for column in table.columns:
        if not isinstance(column, str) or not column.strip():
            raise ValueError(f"commitment_rates columns must be non-empty fund-type strings, got {column!r}")
        columns.append(column.strip())
    table.index = pd.Index(years, name="year")
    table.columns = pd.Index(columns, name="fund_type")
    table = table.sort_index()
    years = [int(year) for year in table.index]
    try:
        table = table.astype(float)
    except (TypeError, ValueError):
        raise ValueError("commitment_rates must be numeric") from None
    numbers = table.to_numpy()
    if not np.isfinite(numbers).all() or (numbers < 0).any():
        raise ValueError("commitment_rates must be finite and non-negative")
    if len(years) and years != list(range(years[0], years[-1] + 1)):
        raise ValueError(
            f"commitment_rates must list every calendar year from {years[0]} to {years[-1]}; "
            "use 0 for years with no target"
        )
    return table


@dataclass(frozen=True, eq=False)
class Portfolio:
    """Portfolio-held data, in the portfolio's base currency.

    ``base_currency`` is the currency of the liquid index and of every report, and it alone
    decides whether exchange rates apply: ``"USD"`` means none and ``usd_rate`` must be
    omitted; anything else requires ``usd_rate``, the dated price of 1 USD in base currency.
    ``liquid_levels`` are dated total-return levels; their dates are the simulation grid and
    the first level is the starting balance. ``commitment_rates`` is year × fund type.
    """

    base_currency: str
    liquid_levels: Any
    commitment_rates: Any
    usd_rate: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.base_currency, str) or not self.base_currency.strip():
            raise ValueError("base_currency must be a currency code such as 'USD' or 'GBP'")
        _set(self, "base_currency", self.base_currency.strip().upper())
        levels = coerce_dated_series(self.liquid_levels, name="liquid_levels", sum_same_day=False)
        if levels.empty:
            raise ValueError("liquid_levels needs at least one observation")
        if (levels <= 0).any():
            raise ValueError("liquid_levels must be strictly positive")
        _set(self, "liquid_levels", levels)
        _set(self, "commitment_rates", coerce_rate_table(self.commitment_rates))
        if self.base_currency == PRIVATE_CURRENCY:
            if self.usd_rate is not None:
                raise ValueError("usd_rate must be omitted when base_currency is USD")
        else:
            if self.usd_rate is None:
                raise ValueError(
                    f"usd_rate is required: base_currency {self.base_currency!r} is not {PRIVATE_CURRENCY}"
                )
            rate = coerce_dated_series(self.usd_rate, name="usd_rate", sum_same_day=False)
            if rate.empty or (rate <= 0).any():
                raise ValueError("usd_rate must contain strictly positive rates")
            _set(self, "usd_rate", rate)

    @property
    def requires_fx_conversion(self) -> bool:
        return self.base_currency != PRIVATE_CURRENCY

    @property
    def first_date(self) -> date:
        return self.liquid_levels.index[0].date()

    @property
    def last_date(self) -> date:
        return self.liquid_levels.index[-1].date()

    @property
    def calendar_years(self) -> range:
        return range(self.first_date.year, self.last_date.year + 1)

    def __repr__(self) -> str:
        return (
            f"Portfolio(base_currency={self.base_currency!r}, observations={len(self.liquid_levels)}, "
            f"{self.first_date}..{self.last_date}, fund_types={list(self.commitment_rates.columns)})"
        )
