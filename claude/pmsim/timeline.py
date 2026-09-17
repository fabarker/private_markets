"""The observation grid, and a fund's history laid onto it.

The engine never touches a date: a ``Timeline`` built from the liquid index converts
every dated input into per-period arrays once, and the loop then works on integer
period indices ``t``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .dates import as_date


@dataclass(frozen=True)
class Timeline:
    """Observation dates: unique, increasing calendar dates. Period ``t`` is ``dates[t]``."""

    dates: pd.DatetimeIndex

    def __post_init__(self) -> None:
        try:
            dates = pd.DatetimeIndex([pd.Timestamp(as_date(d)) for d in self.dates], name="date")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"timeline: {exc}") from None
        if len(dates) == 0:
            raise ValueError("timeline needs at least one date")
        if not dates.is_unique or not dates.is_monotonic_increasing:
            raise ValueError("timeline dates must be unique and increasing")
        object.__setattr__(self, "dates", dates)

    @property
    def n(self) -> int:
        return len(self.dates)

    def date_at(self, t: int) -> date:
        return self.dates[t].date()

    @property
    def years(self) -> range:
        """Every calendar year the timeline touches, first to last inclusive."""
        return range(self.dates[0].year, self.dates[-1].year + 1)

    def index_of(self, day: Any) -> int:
        """Index of the first observation on or after ``day``; ``n`` when ``day`` is past the last one."""
        return int(self.dates.searchsorted(pd.Timestamp(as_date(day)), side="left"))

    def asof(self, series: pd.Series, *, name: str = "series") -> np.ndarray:
        """The last value on or before each observation, as an array aligned to the timeline."""
        aligned = series.sort_index().reindex(self.dates, method="ffill")
        if aligned.isna().any():
            raise ValueError(f"{name} has no value on or before the first observation {self.date_at(0)}")
        return aligned.to_numpy(dtype=float)


@dataclass(frozen=True)
class FundPath:
    """A fund's unit history on a timeline.

    ``calls[t]`` and ``distributions[t]`` are the gross unit flows dated in
    ``(dates[t-1], dates[t]]`` (the first period takes everything on or before ``dates[0]``).
    ``nav[t]`` is the unit NAV after the last event on or before ``dates[t]``.
    ``closing_index`` is the period the fund is committed in, or ``n`` when its closing
    lies beyond the last observation. Arrays are read-only.
    """

    calls: np.ndarray
    distributions: np.ndarray
    nav: np.ndarray
    closing_index: int

    def __post_init__(self) -> None:
        arrays = {}
        for label in ("calls", "distributions", "nav"):
            array = np.array(getattr(self, label), dtype=float)  # a copy
            array.flags.writeable = False
            arrays[label] = array
        if any(a.ndim != 1 for a in arrays.values()) or len({a.shape for a in arrays.values()}) != 1:
            raise ValueError("calls, distributions and nav must be 1-D arrays of one length")
        for label, array in arrays.items():
            object.__setattr__(self, label, array)
        closing_index = int(self.closing_index)
        if not 0 <= closing_index <= self.n:
            raise ValueError(f"closing_index {closing_index} is outside 0..{self.n}")
        object.__setattr__(self, "closing_index", closing_index)

    @property
    def n(self) -> int:
        return len(self.nav)

    @property
    def beyond_horizon(self) -> bool:
        return self.closing_index >= self.n
