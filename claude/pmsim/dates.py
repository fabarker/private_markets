"""Date coercion shared by the inputs and the timeline.

Everything dated in this package is a timezone-naive calendar date. Midnight datetimes
are accepted as dates; anything carrying a time of day or a timezone is rejected rather
than silently truncated.
"""
from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any, Mapping

import pandas as pd


def as_date(value: Any) -> date:
    """Coerce ``value`` to a calendar date.

    Accepts ``date``, midnight ``datetime`` / ``pd.Timestamp``, ISO strings and anything
    ``pd.Timestamp`` understands. Raises ``ValueError`` for intraday or tz-aware values
    and ``TypeError`` for values that are not dates at all.
    """
    if value is None or value is pd.NaT or (isinstance(value, float) and math.isnan(value)):
        raise TypeError("date is missing")
    if isinstance(value, str):
        try:
            value = pd.Timestamp(value)
        except ValueError:
            raise TypeError(f"cannot interpret {value!r} as a date") from None
    if isinstance(value, datetime):  # pd.Timestamp is a datetime subclass
        stamp = pd.Timestamp(value)
        if stamp.tz is not None or stamp != stamp.normalize():
            raise ValueError(f"{value!r} is not a timezone-naive calendar date")
        return stamp.date()
    if isinstance(value, date):
        return value
    try:
        return as_date(pd.Timestamp(value))  # numpy datetime64 and similar
    except (TypeError, ValueError):
        raise TypeError(f"cannot interpret {value!r} as a date") from None


def _anniversary(start: date, years: int) -> date:
    try:
        return start.replace(year=start.year + years)
    except ValueError:  # 29 February has no anniversary in a year without one
        return start.replace(year=start.year + years, day=28)


def years_between(start: Any, end: Any) -> float:
    """Years from ``start`` to ``end`` counted in anniversaries: the whole ones, plus the elapsed share of the next.

    31 Dec 2010 to 31 Dec 2012 is exactly 2, and to 30 Jun 2012 it is 1 + 182/366, so a
    path that compounds once a year lands exactly on its annual values. Negative when
    ``end`` is the earlier date.
    """
    start, end = as_date(start), as_date(end)
    if end < start:
        return -years_between(end, start)
    whole = end.year - start.year
    if _anniversary(start, whole) > end:
        whole -= 1
    last, following = _anniversary(start, whole), _anniversary(start, whole + 1)
    return whole + (end - last).days / (following - last).days


def coerce_dated_series(values: Any, *, name: str, sum_same_day: bool) -> pd.Series:
    """Coerce dated values to a float Series on a unique, sorted, naive ``DatetimeIndex``.

    ``values`` may be a Series indexed by dates, a mapping ``{date: value}``, an iterable of
    ``(date, value)`` pairs, or ``None`` for an empty series. Values must be finite numbers.
    Two entries on one day are summed when ``sum_same_day`` is set (two calls on one day are
    that day's call) and rejected otherwise (two NAV marks on one day are a data error).
    The result is a fresh object; the caller's data is never referenced.
    """
    if values is None:
        pairs: list[Any] = []
    elif isinstance(values, pd.Series):
        pairs = list(zip(values.index, values.to_numpy()))
    elif isinstance(values, Mapping):
        pairs = list(values.items())
    else:
        pairs = list(values)

    stamps: list[pd.Timestamp] = []
    numbers: list[float] = []
    for pair in pairs:
        try:
            day, value = pair
        except (TypeError, ValueError):
            raise ValueError(f"{name}: expected (date, value) pairs, got {pair!r}") from None
        try:
            stamps.append(pd.Timestamp(as_date(day)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}: {exc}") from None
        if isinstance(value, bool):
            raise ValueError(f"{name}: value on {day} is a boolean, not a number")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name}: value {value!r} on {day} is not a number") from None
        if not math.isfinite(number):
            raise ValueError(f"{name}: value on {day} must be finite")
        numbers.append(number)

    series = pd.Series(numbers, index=pd.DatetimeIndex(stamps, name="date"), dtype=float, name=name)
    series = series.sort_index()
    if not series.index.is_unique:
        if not sum_same_day:
            day = series.index[series.index.duplicated()][0].date()
            raise ValueError(f"{name}: more than one entry on {day}")
        series = series.groupby(level=0).sum()
        series.name = name
        series.index.name = "date"
    return series
