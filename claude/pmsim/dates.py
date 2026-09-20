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
    # Nothing there at all: None, NaT or a float NaN.
    is_nan = isinstance(value, float) and math.isnan(value)
    if value is None or value is pd.NaT or is_nan:
        raise TypeError("date is missing")

    # Text is parsed first, then handled as the timestamp it turned into.
    if isinstance(value, str):
        try:
            value = pd.Timestamp(value)
        except ValueError:
            raise TypeError(f"cannot interpret {value!r} as a date") from None

    # A datetime (pd.Timestamp is one) counts as a date only at midnight, without a timezone.
    if isinstance(value, datetime):
        stamp = pd.Timestamp(value)

        has_timezone = stamp.tz is not None
        has_time_of_day = stamp != stamp.normalize()
        if has_timezone or has_time_of_day:
            raise ValueError(f"{value!r} is not a timezone-naive calendar date")

        return stamp.date()

    # Already a plain date.
    if isinstance(value, date):
        return value

    # Anything else pandas can read: numpy datetime64 and similar.
    try:
        return as_date(pd.Timestamp(value))
    except (TypeError, ValueError):
        raise TypeError(f"cannot interpret {value!r} as a date") from None


def _anniversary(start: date, years: int) -> date:
    """The same day of the year, ``years`` later. 29 February falls back to the 28th."""
    try:
        return start.replace(year=start.year + years)
    except ValueError:
        # 29 February has no anniversary in a year without one
        return start.replace(year=start.year + years, day=28)


def years_between(start: Any, end: Any) -> float:
    """Years from ``start`` to ``end``, counted in anniversaries.

    The whole anniversaries passed, plus the elapsed share of the next one. 31 Dec 2010 to
    31 Dec 2012 is exactly 2, and to 30 Jun 2012 it is 1 + 182/366, so a path that compounds
    once a year lands exactly on its annual values. Negative when ``end`` is the earlier date.
    """
    start = as_date(start)
    end = as_date(end)

    if end < start:
        return -years_between(end, start)

    # How many whole anniversaries of ``start`` have passed by ``end``.
    whole_years = end.year - start.year
    if _anniversary(start, whole_years) > end:
        whole_years -= 1

    # Where ``end`` sits between the last anniversary and the next one.
    last_anniversary = _anniversary(start, whole_years)
    next_anniversary = _anniversary(start, whole_years + 1)

    days_since_last = (end - last_anniversary).days
    days_in_this_year = (next_anniversary - last_anniversary).days

    return whole_years + days_since_last / days_in_this_year


def _as_date_value_pairs(values: Any) -> list[Any]:
    """``values`` in any accepted shape, as a list of ``(date, value)`` pairs."""
    if values is None:
        return []

    if isinstance(values, pd.Series):
        return list(zip(values.index, values.to_numpy()))

    if isinstance(values, Mapping):
        return list(values.items())

    return list(values)


def _as_finite_number(value: Any, *, day: Any, name: str) -> float:
    """``value`` as a float, or a ``ValueError`` that names the series and the day."""
    if isinstance(value, bool):
        raise ValueError(f"{name}: value on {day} is a boolean, not a number")

    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name}: value {value!r} on {day} is not a number") from None

    if not math.isfinite(number):
        raise ValueError(f"{name}: value on {day} must be finite")

    return number


def coerce_dated_series(values: Any, *, name: str, sum_same_day: bool) -> pd.Series:
    """Coerce dated values to a float Series on a unique, sorted, naive ``DatetimeIndex``.

    ``values`` may be a Series indexed by dates, a mapping ``{date: value}``, an iterable of
    ``(date, value)`` pairs, or ``None`` for an empty series. Values must be finite numbers.
    Two entries on one day are summed when ``sum_same_day`` is set (two calls on one day are
    that day's call) and rejected otherwise (two NAV marks on one day are a data error).
    The result is a fresh object; the caller's data is never referenced.
    """
    pairs = _as_date_value_pairs(values)

    # Check every pair: a real date, then a finite number.
    stamps: list[pd.Timestamp] = []
    numbers: list[float] = []

    for pair in pairs:
        try:
            day, value = pair
        except (TypeError, ValueError):
            raise ValueError(f"{name}: expected (date, value) pairs, got {pair!r}") from None

        try:
            stamp = pd.Timestamp(as_date(day))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}: {exc}") from None

        stamps.append(stamp)
        numbers.append(_as_finite_number(value, day=day, name=name))

    # Build the series, in date order.
    index = pd.DatetimeIndex(stamps, name="date")
    series = pd.Series(numbers, index=index, dtype=float, name=name)
    series = series.sort_index()

    if series.index.is_unique:
        return series

    # Two entries on one day: a data error, or that day's total.
    if not sum_same_day:
        first_repeated_day = series.index[series.index.duplicated()][0].date()
        raise ValueError(f"{name}: more than one entry on {first_repeated_day}")

    series = series.groupby(level=0).sum()
    series.name = name
    series.index.name = "date"
    return series
