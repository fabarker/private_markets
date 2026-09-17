from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, List, Optional, Sequence, Union


class EntryType(str, Enum):
    """Type of a realized history entry."""
    FLOW = "FLOW"   # a cash flow event (call negative, distribution positive)
    NAV = "NAV"     # a point-in-time net asset value mark


DateLike = Union[date, datetime, str]


def _coerce_date(d: DateLike) -> date:
    """Accept an unambiguous calendar date, never silently truncate a time."""
    if isinstance(d, str):
        return _coerce_date(datetime.fromisoformat(d))
    if isinstance(d, datetime):
        if d.tzinfo is not None or any((d.hour, d.minute, d.second, d.microsecond,
                                      getattr(d, "nanosecond", 0))):
            raise ValueError("Dates must be timezone-naive calendar dates (midnight)")
        return d.date()
    if isinstance(d, date):
        return d
    raise TypeError(f"Unsupported date type: {type(d).__name__}")


@dataclass(frozen=True)
class HistoryEntry:
    """A single dated observation for a fund vintage."""
    date: date
    value: float
    type: EntryType

    def __post_init__(self) -> None:
        object.__setattr__(self, "date", _coerce_date(self.date))
        object.__setattr__(self, "type", EntryType(self.type))
        value = float(self.value)
        if not math.isfinite(value):
            raise ValueError("History values must be finite numbers")
        if self.type is EntryType.NAV and value < 0:
            raise ValueError("NAV marks must be non-negative")
        object.__setattr__(self, "value", value)

    @classmethod
    def create(cls, date: DateLike, value: float,
               type: Union[EntryType, str]) -> "HistoryEntry":
        return cls(
            date=_coerce_date(date),
            value=float(value),
            type=type if isinstance(type, EntryType) else EntryType(type),
        )

    def to_dict(self) -> dict:
        return {"date": self.date.isoformat(),
                "value": self.value,
                "type": self.type.value}


@dataclass
class FundVintage:
    """Represents a private-markets fund vintage.

    ``normalized_realized_nav`` and ``normalized_realized_net_cash_flow``
    carry the fund's history on a **per-unit-committed** basis (i.e. the
    cash flows / NAVs of a hypothetical $1 commitment). The dollar-scaled
    views used for reporting are derived by
    :meth:`get_realized_nav` / :meth:`get_realized_net_cash_flow`, which
    multiply every entry by :attr:`commitment_size`.
    """

    name: str
    strategy: Any
    commitment_size: float = 0.0
    vintage_year: Optional[int] = None
    commitment_date: Optional[date] = None
    normalized_realized_nav: List[HistoryEntry] = field(default_factory=list)
    normalized_realized_net_cash_flow: List[HistoryEntry] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Validation / normalisation
    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")

        self.commitment_size = float(self.commitment_size)
        if not math.isfinite(self.commitment_size) or self.commitment_size < 0:
            raise ValueError(
                "commitment_size must be a finite, non-negative number; "
                f"got {self.commitment_size!r}"
            )

        # Coerce commitment_date if supplied as str/datetime.
        if self.commitment_date is not None:
            self.commitment_date = _coerce_date(self.commitment_date)

        # Invalid entries must not disappear from financial histories.
        self.normalized_realized_nav = self._clean(
            self.normalized_realized_nav, EntryType.NAV,
        )
        self.normalized_realized_net_cash_flow = self._clean(
            self.normalized_realized_net_cash_flow, EntryType.FLOW,
        )
        self.normalized_realized_nav.sort(key=lambda e: e.date)
        self.normalized_realized_net_cash_flow.sort(key=lambda e: e.date)
        mark_dates = [e.date for e in self.normalized_realized_nav]
        if len(mark_dates) != len(set(mark_dates)):
            raise ValueError(f"Duplicate NAV marks for {self.name!r}")

        # Vintage year: must be explicit or derivable from ``commitment_date``
        # — inference from the first cash flow is disallowed because the
        # commitment often precedes the first call by a year (e.g. PEM2011
        # commits in 2010).
        if self.vintage_year is None:
            if self.commitment_date is None:
                raise ValueError(
                    "vintage_year or commitment_date is required; "
                    "inference from the first cash flow is disallowed."
                )
            self.vintage_year = self.commitment_date.year

        if isinstance(self.vintage_year, bool) or not isinstance(self.vintage_year, int):
            raise TypeError("vintage_year must be an int")
        if not (1900 <= self.vintage_year <= 9999):
            raise ValueError(
                f"vintage_year {self.vintage_year} out of reasonable range "
                "[1900, 9999]"
            )

    @classmethod
    def _clean(cls, entries, expected: EntryType) -> List[HistoryEntry]:
        cleaned: List[HistoryEntry] = []
        for i, e in enumerate(entries):
            try:
                cleaned.append(cls._normalise(e, expected))
            except (TypeError, ValueError, KeyError, OverflowError) as exc:
                raise ValueError(f"Invalid {expected.value} entry at index {i}: {exc}") from exc
        return cleaned

    @staticmethod
    def _normalise(
            entry: Union[HistoryEntry, dict, tuple],
            expected: EntryType
    ) -> HistoryEntry:

        if isinstance(entry, HistoryEntry):
            he = HistoryEntry.create(entry.date, entry.value, entry.type)
        elif isinstance(entry, dict):
            he = HistoryEntry.create(
                date=entry["date"],
                value=entry["value"],
                type=entry.get("type", expected),
            )
        elif isinstance(entry, tuple) and len(entry) in (2, 3):
            d, v = entry[0], entry[1]
            t = entry[2] if len(entry) == 3 else expected
            he = HistoryEntry.create(date=d, value=v, type=t)
        else:
            raise TypeError(
                f"Cannot build HistoryEntry from {type(entry).__name__}: {entry!r}"
            )
        if he.type is not expected:
            raise ValueError(
                f"Entry type {he.type} does not match expected bucket {expected}"
            )
        return he

    # ------------------------------------------------------------------ #
    # Mutation helpers (all inputs are per-unit-committed)
    # ------------------------------------------------------------------ #
    def add_flow(self, d: DateLike, value: float) -> None:
        """Append a normalized (per-unit) cash flow.

        ``value`` is expressed on a $1-committed basis: positive =
        distribution, negative = capital call. If a same-date call and
        distribution occur, prefer :meth:`add_call` + :meth:`add_distribution`
        so gross figures are preserved (see class docstring).
        """
        self.normalized_realized_net_cash_flow.append(
            HistoryEntry.create(d, value, EntryType.FLOW)
        )
        self.normalized_realized_net_cash_flow.sort(key=lambda e: e.date)

    def add_call(self, d: DateLike, value: float) -> None:
        """Append a normalized (per-unit) **gross** capital call.

        ``value`` is the positive magnitude of the call on a $1-committed
        basis; it is stored as a negative FLOW entry. Use this instead of
        :meth:`add_flow` when a call and a distribution occur on the same
        date, so ``total_called`` / ``total_distributed`` /
        ``unfunded_commitment`` remain accurate on a gross basis.
        """
        value = float(value)
        if value < 0:
            raise ValueError(
                "add_call expects a positive magnitude; got "
                f"{value!r}. Pass the gross call amount as a positive number."
            )
        self.normalized_realized_net_cash_flow.append(
            HistoryEntry.create(d, -float(value), EntryType.FLOW)
        )
        self.normalized_realized_net_cash_flow.sort(key=lambda e: e.date)

    def add_distribution(self, d: DateLike, value: float) -> None:
        """Append a normalized (per-unit) **gross** distribution.

        ``value`` is the positive magnitude of the distribution on a
        $1-committed basis; it is stored as a positive FLOW entry. Use
        this instead of :meth:`add_flow` when a call and a distribution
        occur on the same date.
        """
        value = float(value)
        if value < 0:
            raise ValueError(
                "add_distribution expects a positive magnitude; got "
                f"{value!r}. Pass the gross distribution amount as positive."
            )
        self.normalized_realized_net_cash_flow.append(
            HistoryEntry.create(d, float(value), EntryType.FLOW)
        )
        self.normalized_realized_net_cash_flow.sort(key=lambda e: e.date)

    def add_nav(self, d: DateLike, value: float) -> None:
        """Append a normalized (per-unit) NAV mark."""
        entry = HistoryEntry.create(d, value, EntryType.NAV)
        if any(e.date == entry.date for e in self.normalized_realized_nav):
            raise ValueError(f"Duplicate NAV mark for {self.name!r} on {entry.date}")
        self.normalized_realized_nav.append(entry)
        self.normalized_realized_nav.sort(key=lambda e: e.date)

    # ------------------------------------------------------------------ #
    # Dollar-scaled views (normalized × commitment_size)
    # ------------------------------------------------------------------ #
    def _scaled(self, entries: List[HistoryEntry]) -> List[HistoryEntry]:
        """Return ``entries`` with every value multiplied by the commitment."""
        c = self.commitment_size
        return [
            HistoryEntry(date=e.date, value=e.value * c, type=e.type)
            for e in entries
        ]

    def get_realized_nav(self) -> List[HistoryEntry]:
        """Dollar-scaled NAV history (``normalized_realized_nav`` × commitment)."""
        return self._scaled(self.normalized_realized_nav)

    def get_realized_net_cash_flow(self) -> List[HistoryEntry]:
        """Dollar-scaled cash-flow history
        (``normalized_realized_net_cash_flow`` × commitment)."""
        return self._scaled(self.normalized_realized_net_cash_flow)

    @property
    def realized_nav(self) -> List[HistoryEntry]:
        """Convenience alias for :meth:`get_realized_nav`."""
        return self.get_realized_nav()

    @property
    def realized_net_cash_flow(self) -> List[HistoryEntry]:
        """Convenience alias for :meth:`get_realized_net_cash_flow`."""
        return self.get_realized_net_cash_flow()

    # ------------------------------------------------------------------ #
    # Derived / convenience (all on the dollar-scaled series)
    # ------------------------------------------------------------------ #
    @property
    def age(self) -> int:
        return max(0, datetime.utcnow().year - self.vintage_year)

    @property
    def cumulative_net_cash_flow(self) -> float:
        return sum(e.value for e in self.realized_net_cash_flow)

    @property
    def total_called(self) -> float:
        """Sum of capital calls (absolute value of negative flows)."""
        return sum(-e.value for e in self.realized_net_cash_flow if e.value < 0)

    @property
    def total_distributed(self) -> float:
        """Sum of distributions (positive flows)."""
        return sum(e.value for e in self.realized_net_cash_flow if e.value > 0)

    @property
    def unfunded_commitment(self) -> float:
        """Commitment less all supplied calls (floored at 0; no recycling model)."""
        return max(0.0, self.commitment_size - self.total_called)

    @property
    def latest_nav(self) -> Optional[HistoryEntry]:
        nav = self.realized_nav
        return nav[-1] if nav else None

    @property
    def total_value(self) -> float:
        nav = self.latest_nav.value if self.latest_nav else 0.0
        return nav + self.total_distributed

    @property
    def net_profit(self) -> float:
        nav = self.latest_nav.value if self.latest_nav else 0.0
        return nav + self.total_distributed - self.total_called

    # ------------------------------------------------------------------ #
    # Performance metrics
    # ------------------------------------------------------------------ #
    def irr(
        self,
        *,
        as_of: Optional[DateLike] = None,
        tol: float = 1e-7,
        max_iter: int = 200,
    ) -> Optional[float]:
        """Annualised money-weighted IRR (XIRR) on dollar-scaled flows.

        Treats the vintage as if liquidated on ``as_of`` (defaults to
        the latest available NAV date) by appending the cash-flow-adjusted
        NAV at that date as a terminal positive cash flow. Cash-flow signs
        follow the class convention — calls negative, distributions
        positive — so the IRR is expressed from the LP's perspective.

        Uses an ACT/365 day-count and a bisection root-finder on
        ``NPV(r) = Σ cf_i / (1 + r) ** (days_i / 365)``. Returns
        ``None`` when there is no two-sided dated cash-flow sequence or no
        bracketed root. The initial bracket is [-0.999, 10.0]; the upper
        bound can expand six times. Non-conventional cash flows may have
        multiple IRRs; this method does not enumerate all of them.

        Parameters
        ----------
        as_of:
            Optional valuation date. Uses the last NAV date when
            omitted; returns None if neither is available. An explicit
            as_of adjusts the latest earlier mark for subsequent flows.
        tol, max_iter:
            Convergence tolerance on ``|NPV|`` and maximum number of
            bisection iterations.
        """
        if not math.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be finite and positive")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        cf = list(self.realized_net_cash_flow)  # dollar-scaled
        nav_hist = list(self.realized_nav)

        # Determine terminal date + NAV.
        if as_of is None:
            if not nav_hist:
                # No NAV to close out the position — IRR undefined.
                return None
            terminal_date = nav_hist[-1].date
            terminal_nav = nav_hist[-1].value
        else:
            terminal_date = _coerce_date(as_of)
            # Include flows after the last mark instead of double-counting
            # distributions against a stale terminal value.
            terminal_nav = self.nav_on(terminal_date)

        # Only keep flows up to the terminal date; append terminal NAV.
        pairs: List[tuple[date, float]] = [
            (e.date, e.value) for e in cf if e.date <= terminal_date
        ]
        if terminal_nav:
            pairs.append((terminal_date, terminal_nav))
        if len(pairs) < 2:
            return None

        # Need at least one negative and one positive flow for a sign change.
        has_pos = any(v > 0 for _, v in pairs)
        has_neg = any(v < 0 for _, v in pairs)
        if not (has_pos and has_neg):
            return None

        t0 = min(d for d, _ in pairs)
        days = [(d - t0).days for d, _ in pairs]
        values = [v for _, v in pairs]

        def npv(rate: float) -> float:
            base = 1.0 + rate
            if base <= 0.0:
                return float("inf")
            return sum(v / (base ** (days[i] / 365.0))
                       for i, v in enumerate(values))

        lo, hi = -0.999, 10.0
        f_lo, f_hi = npv(lo), npv(hi)
        if abs(f_lo) <= tol:
            return lo
        if abs(f_hi) <= tol:
            return hi
        # Expand the upper bound to allow unusually large positive IRRs.
        expansions = 0
        while f_lo * f_hi > 0.0 and expansions < 6:
            hi *= 2.0
            f_hi = npv(hi)
            expansions += 1
        if f_lo * f_hi > 0.0:
            return None

        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            f_mid = npv(mid)
            if abs(f_mid) < tol or (hi - lo) < tol:
                return mid
            if f_lo * f_mid < 0.0:
                hi, f_hi = mid, f_mid
            else:
                lo, f_lo = mid, f_mid
        return 0.5 * (lo + hi)

    # ------------------------------------------------------------------ #
    # Interval / monthly helpers used by the portfolio engine
    # ------------------------------------------------------------------ #
    def total_called_as_of(self, d: DateLike) -> float:
        """Gross calls through a date, excluding later supplied history."""
        target = _coerce_date(d)
        return self.commitment_size * math.fsum(
            -e.value for e in self.normalized_realized_net_cash_flow
            if e.date <= target and e.value < 0
        )

    def total_distributed_as_of(self, d: DateLike) -> float:
        """Gross distributions through a date, excluding later history."""
        target = _coerce_date(d)
        return self.commitment_size * math.fsum(
            e.value for e in self.normalized_realized_net_cash_flow
            if e.date <= target and e.value > 0
        )

    def nav_series(self, model_dates: Sequence[DateLike]):
        """Cash-flow-adjusted dollar NAV on an arbitrary calendar-date grid.

        Process all dated history through each observation, including marks
        between observations. Same-day marks include same-day cash flows.
        No investment return is assumed between marks. Unlike monthly_nav,
        this also reconstructs opening NAV from any preceding history.
        """
        import pandas as pd
        from simulation_events import prepare_fund_events, validate_model_dates

        dates = validate_model_dates(model_dates, allow_empty=True)
        path = prepare_fund_events(self, dates)
        invalid = path.minimum_nav * self.commitment_size < -1e-8
        if invalid.any():
            i = int(invalid.nonzero()[0][0])
            raise ValueError(
                f"Negative inferred NAV for {self.name!r} on {path.minimum_nav_date[i]}"
            )
        return pd.Series((path.nav * self.commitment_size).clip(min=0),
                         index=dates, name=self.name, dtype=float)

    def nav_on(self, d: DateLike) -> float:
        """Cash-flow-adjusted dollar NAV as of one date, without future marks."""
        return float(self.nav_series([d]).iloc[0])

    def net_flow_between(self, start: DateLike, end: DateLike) -> float:
        """Sum of dollar-scaled flows in ``(start, end]`` (start exclusive)."""
        s = _coerce_date(start)
        e = _coerce_date(end)
        if e < s:
            raise ValueError(f"end ({e}) precedes start ({s})")
        return sum(
            entry.value for entry in self.realized_net_cash_flow
            if s < entry.date <= e
        )

    def monthly_flows(self, model_dates: Sequence[DateLike]):
        """Per-period dollar-scaled net cash flow at each model date.

        Returns a :class:`pandas.Series` indexed by ``model_dates``.
        The first entry is 0.0 (no prior interval); each subsequent
        entry sums flows in ``(model_dates[i-1], model_dates[i]]``.
        """
        import pandas as pd
        dates = [_coerce_date(d) for d in model_dates]
        idx = pd.DatetimeIndex([datetime(d.year, d.month, d.day) for d in dates])
        out = pd.Series(0.0, index=idx)
        for i in range(1, len(dates)):
            out.iloc[i] = self.net_flow_between(dates[i - 1], dates[i])
        return out

    def nav_mark_on(self, d: DateLike) -> Optional[float]:
        """Return the dollar-scaled NAV mark on exactly ``d`` (or ``None``).

        Raises ``ValueError`` if more than one NAV entry lies on ``d``
        — duplicate marks are a data-integrity error, not something to
        silently reduce.
        """
        target = _coerce_date(d)
        matches = [e.value for e in self.realized_nav if e.date == target]
        if len(matches) > 1:
            raise ValueError(
                f"Duplicate NAV marks for {self.name!r} on {target}"
            )
        return matches[0] if matches else None

    def monthly_nav(self, model_dates: Sequence[DateLike]):
        """Roll NAV forward across ``model_dates``.

        Rules (matching the workbook):

        * First model date → NAV is 0 unless an explicit mark exists.
        * Quarter-end months {3, 6, 9, 12}: refresh from an exact
          quarter-end NAV mark; if the mark is missing, carry the
          prior NAV forward.
        * Other months: ``nav[t] = nav[t-1] − flow[t]`` (a negative
          call therefore increases NAV, a positive distribution
          decreases it — the workbook sign convention).
        """
        import pandas as pd
        dates = [_coerce_date(d) for d in model_dates]
        idx = pd.DatetimeIndex([datetime(d.year, d.month, d.day) for d in dates])
        flows = self.monthly_flows(dates)
        nav = pd.Series(0.0, index=idx)

        # Seed period 0 with an exact mark if one is available on t0.
        first_mark = self.nav_mark_on(dates[0]) if dates else None
        if first_mark is not None:
            nav.iloc[0] = first_mark

        for i in range(1, len(dates)):
            current = dates[i]
            prev_nav = float(nav.iloc[i - 1])
            if current.month in (3, 6, 9, 12):
                mark = self.nav_mark_on(current)
                nav.iloc[i] = prev_nav if mark is None else mark
            else:
                nav.iloc[i] = prev_nav - float(flows.iloc[i])
        return nav

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "vintage_year": self.vintage_year,
            "commitment_date": (
                self.commitment_date.isoformat()
                if self.commitment_date is not None else None
            ),
            "name": self.name,
            "strategy": self.strategy,
            "commitment_size": self.commitment_size,
            "normalized_realized_nav": [
                e.to_dict() for e in self.normalized_realized_nav
            ],
            "normalized_realized_net_cash_flow": [
                e.to_dict() for e in self.normalized_realized_net_cash_flow
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FundVintage":
        vy = data.get("vintage_year")
        cd = data.get("commitment_date")
        return cls(
            name=data["name"],
            strategy=data["strategy"],
            commitment_size=float(data.get("commitment_size", 0.0)),
            vintage_year=vy,
            commitment_date=_coerce_date(cd) if cd else None,
            normalized_realized_nav=list(
                data.get("normalized_realized_nav", [])
            ),
            normalized_realized_net_cash_flow=list(
                data.get("normalized_realized_net_cash_flow", [])
            ),
        )

    def __repr__(self) -> str:  # pragma: no cover
        nav = self.latest_nav.value if self.latest_nav else 0.0
        return (
            f"FundVintage(name={self.name!r}, vintage_year={self.vintage_year}, "
            f"strategy={self.strategy!r}, commitment={self.commitment_size:,.2f}, "
            f"unfunded={self.unfunded_commitment:,.2f}, "
            f"latest_nav={nav:,.2f}, cum_net_cf={self.cumulative_net_cash_flow:,.2f}, "
            f"n_flows={len(self.normalized_realized_net_cash_flow)}, "
            f"n_navs={len(self.normalized_realized_nav)})"
        )
