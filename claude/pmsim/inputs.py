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

# Every run assembled from data starts with this much, in the portfolio's own base currency.
# It is not a setting: there is no option to start from another amount, or from an amount of
# another currency converted in.
STARTING_VALUE = 100_000_000.0
UNIT_NAV_TOLERANCE = 1e-12  # floating-point slack before a negative unit NAV is a data error
YEAR_RANGE = (1900, 9999)

# A fund's three dated series, and whether two entries on one day are added together.
# Two calls on one day are that day's call; two NAV marks on one day are a data error.
FUND_SERIES_AND_WHETHER_SAME_DAY_ENTRIES_ARE_SUMMED = (
    ("unit_calls", True),
    ("unit_distributions", True),
    ("unit_nav", False),
)


def _set_frozen_field(obj: Any, name: str, value: Any) -> None:
    """Set a field on a frozen dataclass: only used by ``__post_init__`` to store cleaned inputs."""
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
        # The name and the type: non-empty text, stored without surrounding spaces.
        for label in ("name", "fund_type"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Fund {label} must be a non-empty string")

            _set_frozen_field(self, label, value.strip())

        # The closing date: a calendar date.
        try:
            closing_date = as_date(self.closing_date)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Fund {self.name!r}: closing_date: {exc}") from None

        _set_frozen_field(self, "closing_date", closing_date)

        # The three dated series: gross magnitudes, none dated before the closing.
        closing = pd.Timestamp(self.closing_date)

        for label, sum_same_day in FUND_SERIES_AND_WHETHER_SAME_DAY_ENTRIES_ARE_SUMMED:
            series = coerce_dated_series(
                getattr(self, label),
                name=f"{self.name} {label}",
                sum_same_day=sum_same_day,
            )

            is_negative = series < 0
            if is_negative.any():
                first_negative_day = series.index[is_negative][0].date()
                raise ValueError(
                    f"Fund {self.name!r}: {label} on {first_negative_day} is negative; "
                    "supply gross magnitudes"
                )

            days_before_closing = series.index[series.index < closing]
            if len(days_before_closing):
                first_early_day = days_before_closing[0].date()
                raise ValueError(
                    f"Fund {self.name!r}: {label} dated {first_early_day} "
                    f"precedes the closing on {self.closing_date}"
                )

            _set_frozen_field(self, label, series)

    @property
    def closing_year(self) -> int:
        return self.closing_date.year

    def events_by_day(self) -> list[tuple[pd.Timestamp, float, float, float | None]]:
        """``(day, call, distribution, mark or None)`` for every day with activity, in date order."""
        days_with_activity = (
            self.unit_calls.index
            .union(self.unit_distributions.index)
            .union(self.unit_nav.index)
        )

        events = []
        for day in days_with_activity:
            call = float(self.unit_calls.get(day, 0.0))
            distribution = float(self.unit_distributions.get(day, 0.0))

            mark = None
            if day in self.unit_nav.index:
                mark = float(self.unit_nav[day])

            events.append((day, call, distribution, mark))

        return events

    def align_history(self, timeline: Timeline) -> AlignedFundHistory:
        """Bucket this fund's dated history onto ``timeline`` and rebuild its unit NAV.

        Flows dated in ``(dates[t-1], dates[t]]`` are summed into period ``t``; the first
        period takes everything on or before ``dates[0]``. Unit NAV is rebuilt by walking
        events in date order from zero: a call adds, a distribution subtracts, and a mark
        replaces the running value (marks are taken to include that day's flows). ``unit_nav[t]``
        is the running value after the last event on or before ``dates[t]``, so a mark
        between observations counts and a missing mark leaves the cash-adjusted estimate.
        Events after the last observation are ignored. A negative running value is a data
        error naming the fund and the day.
        """
        events = self.events_by_day()
        n_observations = timeline.n_observations

        calls = np.zeros(n_observations)
        distributions = np.zeros(n_observations)
        nav = np.zeros(n_observations)

        next_event = 0  # position in ``events`` of the first one not yet pooled
        running_nav = 0.0  # unit NAV after the last event pooled so far

        for t, observation in enumerate(timeline.dates):

            # Pool every event dated on or before this observation that has not been pooled yet.
            while next_event < len(events) and events[next_event][0] <= observation:
                day, call, distribution, mark = events[next_event]

                # A mark replaces the running NAV; without one the flows move it.
                if mark is not None:
                    running_nav = mark
                else:
                    running_nav = running_nav + call - distribution

                if running_nav < -UNIT_NAV_TOLERANCE:
                    raise ValueError(
                        f"Fund {self.name!r}: unit NAV would be {running_nav:.6g} on {day.date()}; "
                        "supply a NAV mark on or before that date"
                    )

                calls[t] += call
                distributions[t] += distribution
                next_event += 1

            # The NAV carried at this observation: the running value, never below zero.
            nav[t] = max(running_nav, 0.0)

        closing_period = timeline.first_observation_on_or_after(self.closing_date)
        return AlignedFundHistory(calls, distributions, nav, closing_period)

    def __repr__(self) -> str:
        return (
            f"Fund(name={self.name!r}, fund_type={self.fund_type!r}, closing_date={self.closing_date}, "
            f"calls={len(self.unit_calls)}, distributions={len(self.unit_distributions)}, "
            f"marks={len(self.unit_nav)})"
        )


def _validated_rate_table_years(index: pd.Index) -> list[int]:
    """The rate table's index as integer calendar years, in the order given."""
    first_allowed, last_allowed = YEAR_RANGE

    years = []
    for year in index:
        if isinstance(year, bool) or not isinstance(year, Integral):
            raise ValueError(f"commitment_rates must be indexed by integer calendar years, got {year!r}")

        if not first_allowed <= int(year) <= last_allowed:
            raise ValueError(f"commitment_rates year {year} is outside {first_allowed}..{last_allowed}")

        years.append(int(year))

    return years


def _validated_rate_table_fund_types(columns: pd.Index) -> list[str]:
    """The rate table's columns as fund-type names, without surrounding spaces."""
    fund_types = []
    for column in columns:
        if not isinstance(column, str) or not column.strip():
            raise ValueError(f"commitment_rates columns must be non-empty fund-type strings, got {column!r}")

        fund_types.append(column.strip())

    return fund_types


def coerce_rate_table(value: Any) -> pd.DataFrame:
    """Coerce commitment rates to a DataFrame: contiguous calendar years down, fund types across.

    Accepts a DataFrame or anything ``pd.DataFrame`` accepts, e.g. ``{"BUYOUT": {2027: 0.10}}``.
    Rates are unit commitments per 1 of sizing base (0.10 means commit 10%), finite and
    non-negative. Every year between the first and the last must be present — use 0 for a
    year with no target rather than leaving it out.
    """
    if isinstance(value, pd.DataFrame):
        table = value.copy()
    else:
        table = pd.DataFrame(value)

    if table.index.has_duplicates or table.columns.has_duplicates:
        raise ValueError("commitment_rates must have unique years and unique fund types")

    # Label the rows with integer years and the columns with fund types, in year order.
    years = _validated_rate_table_years(table.index)
    fund_types = _validated_rate_table_fund_types(table.columns)

    table.index = pd.Index(years, name="year")
    table.columns = pd.Index(fund_types, name="fund_type")
    table = table.sort_index()
    years = sorted(years)

    # Every rate is a finite, non-negative number.
    try:
        table = table.astype(float)
    except (TypeError, ValueError):
        raise ValueError("commitment_rates must be numeric") from None

    numbers = table.to_numpy()
    if not np.isfinite(numbers).all() or (numbers < 0).any():
        raise ValueError("commitment_rates must be finite and non-negative")

    # No year may be missing between the first and the last.
    if len(years):
        first_year = years[0]
        last_year = years[-1]
        every_year = list(range(first_year, last_year + 1))

        if years != every_year:
            raise ValueError(
                f"commitment_rates must list every calendar year from {first_year} to {last_year}; "
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
        # The base currency: a code, stored in upper case.
        if not isinstance(self.base_currency, str) or not self.base_currency.strip():
            raise ValueError("base_currency must be a currency code such as 'USD' or 'GBP'")

        _set_frozen_field(self, "base_currency", self.base_currency.strip().upper())

        # The liquid index: at least one level, all strictly positive.
        levels = coerce_dated_series(self.liquid_levels, name="liquid_levels", sum_same_day=False)

        if levels.empty:
            raise ValueError("liquid_levels needs at least one observation")

        if (levels <= 0).any():
            raise ValueError("liquid_levels must be strictly positive")

        _set_frozen_field(self, "liquid_levels", levels)

        # The commitment rates: year × fund type.
        _set_frozen_field(self, "commitment_rates", coerce_rate_table(self.commitment_rates))

        # A dollar portfolio has no exchange rate, and must not be given one.
        if self.base_currency == PRIVATE_CURRENCY:
            if self.usd_rate is not None:
                raise ValueError("usd_rate must be omitted when base_currency is USD")
            return

        # Any other base currency needs the price of a dollar.
        if self.usd_rate is None:
            raise ValueError(
                f"usd_rate is required: base_currency {self.base_currency!r} is not {PRIVATE_CURRENCY}"
            )

        rate = coerce_dated_series(self.usd_rate, name="usd_rate", sum_same_day=False)

        if rate.empty or (rate <= 0).any():
            raise ValueError("usd_rate must contain strictly positive rates")

        _set_frozen_field(self, "usd_rate", rate)

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
