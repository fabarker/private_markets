"""The observation grid, and a fund's history laid onto it.

The engine never touches a date: a ``Timeline`` built from the liquid index converts
every dated input into per-period arrays once, and the loop then works on integer
period indices ``t``.

The liquid index sets the observation frequency — month ends, quarter ends, business
days, or any irregular dates — and fund events are dated on whatever day they happened.
One rule covers every mismatch: an event on any day pools onto the first observation on
or after that day (``first_observation_on_or_after``). Exchange rates go the other way,
because a rate is a state rather than an event: each observation uses the last rate on
or before it (``last_value_on_or_before``).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .dates import as_date

UNIT_HISTORY_FIELDS = ("unit_calls", "unit_distributions", "unit_nav")


@dataclass(frozen=True)
class Timeline:
    """Observation dates: unique, increasing calendar dates. Period ``t`` is ``dates[t]``."""

    dates: pd.DatetimeIndex

    def __post_init__(self) -> None:
        # Every entry must be a calendar date.
        try:
            stamps = [pd.Timestamp(as_date(d)) for d in self.dates]
            dates = pd.DatetimeIndex(stamps, name="date")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"timeline: {exc}") from None

        if len(dates) == 0:
            raise ValueError("timeline needs at least one date")

        if not dates.is_unique or not dates.is_monotonic_increasing:
            raise ValueError("timeline dates must be unique and increasing")

        # The dataclass is frozen, so the cleaned index is set this way.
        object.__setattr__(self, "dates", dates)

    @property
    def n_observations(self) -> int:
        return len(self.dates)

    def observation_date(self, t: int) -> date:
        return self.dates[t].date()

    @property
    def calendar_years(self) -> range:
        """Every calendar year the timeline touches, first to last inclusive."""
        first_year = self.dates[0].year
        last_year = self.dates[-1].year
        return range(first_year, last_year + 1)

    def first_observation_on_or_after(self, day: Any) -> int:
        """Index of the first observation on or after ``day``.

        ``n_observations`` when ``day`` is past the last observation.
        """
        stamp = pd.Timestamp(as_date(day))
        return int(self.dates.searchsorted(stamp, side="left"))

    def first_observations_on_or_after(self, days: Any) -> np.ndarray:
        """``first_observation_on_or_after`` for many days at once, as an array."""
        stamps = pd.DatetimeIndex([pd.Timestamp(as_date(d)) for d in days])
        positions = self.dates.searchsorted(stamps, side="left")
        return np.asarray(positions, dtype=int)

    def last_value_on_or_before(self, series: pd.Series, *, name: str = "series") -> np.ndarray:
        """The last value on or before each observation, as an array aligned to the timeline."""
        in_date_order = series.sort_index()
        aligned = in_date_order.reindex(self.dates, method="ffill")

        # A gap can only be at the start: nothing was known yet at the first observation.
        if aligned.isna().any():
            first_observation = self.observation_date(0)
            raise ValueError(f"{name} has no value on or before the first observation {first_observation}")

        return aligned.to_numpy(dtype=float)


@dataclass(frozen=True)
class AlignedFundHistory:
    """A fund's unit history aligned to a timeline, one entry per observation.

    ``unit_calls[t]`` and ``unit_distributions[t]`` are the gross unit flows dated in
    ``(dates[t-1], dates[t]]`` (the first period takes everything on or before ``dates[0]``).
    ``unit_nav[t]`` is the unit NAV after the last event on or before ``dates[t]``.
    ``closing_period`` is the period the fund is committed in, or ``n_observations`` when
    its closing lies beyond the last observation. Arrays are read-only.
    """

    unit_calls: np.ndarray
    unit_distributions: np.ndarray
    unit_nav: np.ndarray
    closing_period: int

    def __post_init__(self) -> None:
        # Take a private, read-only copy of each array.
        arrays = {}
        for label in UNIT_HISTORY_FIELDS:
            array = np.array(getattr(self, label), dtype=float)
            array.flags.writeable = False
            arrays[label] = array

        # They must all be flat and of one length: one entry per observation.
        all_one_dimensional = all(array.ndim == 1 for array in arrays.values())
        all_the_same_shape = len({array.shape for array in arrays.values()}) == 1
        if not all_one_dimensional or not all_the_same_shape:
            raise ValueError("unit_calls, unit_distributions and unit_nav must be 1-D arrays of one length")

        for label, array in arrays.items():
            object.__setattr__(self, label, array)

        # The closing period is an observation index, or n_observations for "beyond the horizon".
        closing_period = int(self.closing_period)
        if not 0 <= closing_period <= self.n_observations:
            raise ValueError(f"closing_period {closing_period} is outside 0..{self.n_observations}")

        object.__setattr__(self, "closing_period", closing_period)

    @property
    def n_observations(self) -> int:
        return len(self.unit_nav)

    @property
    def closes_beyond_horizon(self) -> bool:
        return self.closing_period >= self.n_observations
