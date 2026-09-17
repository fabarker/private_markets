"""Calendar-date alignment and per-unit fund histories used by the simulator."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from typing import Sequence

import numpy as np
import pandas as pd

from vintage import DateLike, EntryType, FundVintage, _coerce_date


def validate_model_dates(values: Sequence[DateLike], *, allow_empty=False) -> pd.DatetimeIndex:
    """Normalize dates without dropping times, timezones, or duplicate observations."""
    try:
        dates = pd.DatetimeIndex([_coerce_date(d) for d in values], name="date")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid model dates: {exc}") from exc
    if dates.hasnans:
        raise ValueError("Model dates cannot contain NaT")
    if not allow_empty and len(dates) == 0:
        raise ValueError("At least one model date is required")
    if not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("Model dates must be unique and strictly increasing")
    return dates


@dataclass(frozen=True)
class FundEventPath:
    calls: np.ndarray
    distributions: np.ndarray
    nav: np.ndarray
    latest_mark_date: tuple[date | None, ...]
    minimum_nav: np.ndarray
    minimum_nav_date: tuple[date | None, ...]


def prepare_fund_events(fund: FundVintage, dates: pd.DatetimeIndex) -> FundEventPath:
    """Sweep actual event days; marks replace the day's cash-adjusted unit NAV.

    The first bucket includes history through the first observation, supporting
    FundVintage.nav_series. Simulation separately disallows pre-inception funds
    and nonzero events, so its first bucket includes inception-day events only.
    Negative inferred values are recorded, not raised here: a later valuation
    error must not hide an earlier liquidity failure in Simulation.run().
    """
    n = len(dates)
    calls, distributions, navs, minima = (np.zeros(n) for _ in range(4))
    marks_at, minima_at = [None] * n, [None] * n
    flows = FundVintage._clean(fund.normalized_realized_net_cash_flow, EntryType.FLOW)
    marks = FundVintage._clean(fund.normalized_realized_nav, EntryType.NAV)
    daily_calls: dict[date, list[float]] = {}
    daily_distributions: dict[date, list[float]] = {}
    daily_marks: dict[date, float] = {}
    for entry in flows:
        if entry.value < 0:
            daily_calls.setdefault(entry.date, []).append(-entry.value)
        elif entry.value > 0:
            daily_distributions.setdefault(entry.date, []).append(entry.value)
    for entry in marks:
        if entry.date in daily_marks:
            raise ValueError(f"Duplicate NAV marks for {fund.name!r} on {entry.date}")
        daily_marks[entry.date] = entry.value
    event_days = sorted(daily_calls.keys() | daily_distributions.keys() | daily_marks.keys())
    j, nav, latest = 0, 0.0, None
    for i, observation in enumerate(dates.date):
        period_calls, period_distributions = [], []
        while j < len(event_days) and event_days[j] <= observation:
            day = event_days[j]
            try:
                call = math.fsum(daily_calls.get(day, ()))
                distribution = math.fsum(daily_distributions.get(day, ()))
                nav = math.fsum((nav, call, -distribution))
            except OverflowError as exc:
                raise ValueError(f"Non-finite event totals for {fund.name!r} on {day}") from exc
            if day in daily_marks:
                nav, latest = daily_marks[day], day
            if not math.isfinite(nav):
                raise ValueError(f"Non-finite NAV for {fund.name!r} on {day}")
            if nav < minima[i]:
                minima[i], minima_at[i] = nav, day
            period_calls.append(call)
            period_distributions.append(distribution)
            j += 1
        calls[i] = math.fsum(period_calls)
        distributions[i] = math.fsum(period_distributions)
        navs[i], marks_at[i] = nav, latest
    return FundEventPath(calls, distributions, navs, tuple(marks_at), minima, tuple(minima_at))
