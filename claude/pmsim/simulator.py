"""The one loop.

``Simulator(portfolio, funds).run()`` walks the liquid index's observations. Each period:

1. snapshot the opening balances;
2. apply the liquid return (``level[t] / level[t-1]``; 1 at ``t = 0``);
3. bank distributions from existing commitments;
4. size and fix commitments for the funds closing at this observation, all from the same
   base, converting each to a dollar amount at today's rate;
5. bank the new cohort's own distributions, then pay every commitment's calls;
6. value the book, record the period, and stop if the pot could not cover the calls.

Private figures stay in USD until they touch the liquid pot or a report; the exchange
rate of the observation date is applied there and nowhere else.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .inputs import Fund, Portfolio
from .policy import AnnualRatePolicy, CommitmentPolicy, SizingBase
from .state import Commitment, LiquidAccount, PrivateBook
from .timeline import FundPath, Timeline

PERIOD_COLUMNS = [
    "liquid_open", "private_open", "total_open", "return_factor", "usd_rate",
    "liquid_pnl", "distributions", "sizing_base", "commitments", "commitments_usd", "calls",
    "liquid_close", "private_close", "total_close", "private_valuation_pnl", "fx_translation",
]
FUND_COLUMNS = [
    "fund_type", "commitment_usd", "calls_usd", "distributions_usd", "nav_usd",
    "calls_base", "distributions_base", "nav_base",
]
COMMITMENT_COLUMNS = [
    "fund_type", "closing_date", "policy_year", "sizing_base", "rate", "commitment_base",
    "usd_rate", "commitment_usd", "current_year_rate", "carried_rate", "pooled_rate", "weight",
]
EVENT_COLUMNS = ["observation_date", "period", "unit_call", "unit_distribution", "unit_nav_mark"]
_TEXT_COLUMNS = {"fund", "fund_type"}
_DATE_COLUMNS = {"date", "closing_date", "event_date", "observation_date"}
_INT_COLUMNS = {"policy_year", "period"}


def _frame(rows: list[dict[str, Any]], columns: list[str], index: list[str]) -> pd.DataFrame:
    """Build a table with fixed columns and dtypes, correct even when ``rows`` is empty."""
    frame = pd.DataFrame(rows, columns=index + columns)
    for column in frame.columns:
        if column in _DATE_COLUMNS:
            frame[column] = pd.to_datetime(frame[column])
        elif column in _INT_COLUMNS:
            frame[column] = frame[column].astype("int64")
        elif column not in _TEXT_COLUMNS:
            frame[column] = frame[column].astype(float)
    return frame.set_index(index)


@dataclass(frozen=True)
class Shortfall:
    """The first observation at which calls exceeded the cash available, in base currency."""

    t: int
    date: date
    available: float
    calls: float
    calls_by_fund: pd.Series

    @property
    def deficit(self) -> float:
        return self.calls - self.available

    def __str__(self) -> str:
        return (
            f"shortfall of {self.deficit:,.2f} on {self.date}: calls of {self.calls:,.2f} "
            f"against {self.available:,.2f} available"
        )


@dataclass(frozen=True, eq=False)
class SimulationResult:
    """Two tidy tables, a commitment log and a verdict.

    ``periods`` (index: date) is in base currency except ``commitments_usd`` and ``usd_rate``.
    ``funds`` (index: date, fund) has one row per live commitment per period, in both
    currencies. ``commitments`` (index: date, fund) has one row per closing with the rate,
    sizing base and exchange rate used. ``shortfall`` names the first failed observation,
    or is ``None``. ``beyond_horizon`` lists funds whose closing falls after the last
    observation; they are never committed.
    """

    base_currency: str
    periods: pd.DataFrame
    funds: pd.DataFrame
    commitments: pd.DataFrame
    shortfall: Shortfall | None
    beyond_horizon: tuple[str, ...]

    @property
    def status(self) -> str:
        return "shortfall" if self.shortfall is not None else "completed"

    def by_type(self) -> pd.DataFrame:
        """Fund-level figures summed by date and fund type."""
        columns = [c for c in FUND_COLUMNS if c != "fund_type"]
        if self.funds.empty:
            index = pd.MultiIndex.from_arrays([pd.DatetimeIndex([]), []], names=["date", "fund_type"])
            return pd.DataFrame(columns=columns, index=index, dtype=float)
        return self.funds.reset_index().groupby(["date", "fund_type"])[columns].sum()

    def exposures(self) -> pd.DataFrame:
        """Private NAV in base currency by date (rows) and fund (columns); zero before a fund closes."""
        if self.funds.empty:
            return pd.DataFrame(index=self.periods.index, dtype=float)
        return self.funds["nav_base"].unstack("fund").reindex(self.periods.index).fillna(0.0)


class Simulator:
    """Prepare everything dated once, then run the loop from a fresh state on every ``run()``."""

    def __init__(
        self,
        portfolio: Portfolio,
        funds: Sequence[Fund] = (),
        policy: CommitmentPolicy | None = None,
        *,
        stop_on_shortfall: bool = True,
        cash_tolerance: float = 1e-9,
    ) -> None:
        if not isinstance(portfolio, Portfolio):
            raise TypeError("portfolio must be a Portfolio")
        funds = list(funds)
        for fund in funds:
            if not isinstance(fund, Fund):
                raise TypeError(f"funds must contain Fund objects, got {type(fund).__name__}")
        names = [f.name for f in funds]
        if len(set(names)) != len(names):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"fund names must be unique; duplicated: {duplicates}")
        if not (isinstance(cash_tolerance, (int, float)) and math.isfinite(cash_tolerance) and cash_tolerance >= 0):
            raise ValueError("cash_tolerance must be a finite non-negative number")

        self.portfolio = portfolio
        self.funds: tuple[Fund, ...] = tuple(funds)
        self.timeline = Timeline(portfolio.liquid_levels.index)
        first = self.timeline.date_at(0)
        for fund in funds:
            if fund.closing_date < first:
                raise ValueError(
                    f"Fund {fund.name!r} closes on {fund.closing_date}, before the first observation "
                    f"{first}; there are no pre-existing commitments"
                )
        levels = portfolio.liquid_levels.to_numpy(dtype=float)
        self.factors: np.ndarray = np.concatenate([[1.0], levels[1:] / levels[:-1]])
        self.fx: np.ndarray = (
            self.timeline.asof(portfolio.usd_rate, name="usd_rate")
            if portfolio.converts_currency
            else np.ones(self.timeline.n)
        )
        self.paths: dict[str, FundPath] = {f.name: f.on(self.timeline) for f in funds}
        self.policy: CommitmentPolicy = (
            policy if policy is not None
            else AnnualRatePolicy(portfolio.commitment_rates, funds, years=self.timeline.years)
        )
        self.stop_on_shortfall = bool(stop_on_shortfall)
        self.cash_tolerance = float(cash_tolerance)
        self.beyond_horizon: tuple[str, ...] = tuple(f.name for f in funds if self.paths[f.name].beyond_horizon)
        self._cohorts: dict[int, list[Fund]] = {}
        for fund in funds:
            path = self.paths[fund.name]
            if not path.beyond_horizon:
                self._cohorts.setdefault(path.closing_index, []).append(fund)

    # ------------------------------------------------------------------ run
    def run(self) -> SimulationResult:
        timeline, fx, factors = self.timeline, self.fx, self.factors
        liquid = LiquidAccount(float(self.portfolio.liquid_levels.iloc[0]), factors)
        book = PrivateBook()
        period_rows: list[dict[str, Any]] = []
        fund_rows: list[dict[str, Any]] = []
        commitment_rows: list[dict[str, Any]] = []
        shortfall: Shortfall | None = None
        private_open = 0.0

        for t in range(timeline.n):
            day = timeline.date_at(t)
            stamp = pd.Timestamp(day)

            liquid_open = liquid.balance                                          # 1
            nav_usd_open = book.nav(t - 1)

            pnl = liquid.grow(t)                                                  # 2

            distributions_existing = book.distributions(t) * fx[t]                # 3
            liquid.deposit(distributions_existing)

            cohort = self._cohorts.get(t, [])                                     # 4
            base = SizingBase(t, day, liquid.balance, nav_usd_open * fx[t])
            amounts = self._size(cohort, base)
            for fund in cohort:
                commitment = Commitment(fund, self.paths[fund.name], amounts[fund.name] / fx[t])
                book.add(commitment)
                commitment_rows.append(self._commitment_row(stamp, fund, base, amounts[fund.name], fx[t], commitment.usd))

            new = book.closed_at(t)                                               # 5
            distributions_new = math.fsum(c.distributions(t) for c in new) * fx[t]
            liquid.deposit(distributions_new)
            calls = book.calls(t) * fx[t]
            missing = liquid.withdraw(calls)

            private_close = book.nav(t) * fx[t]                                   # 6
            distributions = distributions_existing + distributions_new
            fx_translation = nav_usd_open * (fx[t] - fx[t - 1]) if t > 0 else 0.0
            period_rows.append({
                "date": stamp,
                "liquid_open": liquid_open, "private_open": private_open, "total_open": liquid_open + private_open,
                "return_factor": float(factors[t]), "usd_rate": float(fx[t]),
                "liquid_pnl": pnl, "distributions": distributions, "sizing_base": base.liquid,
                "commitments": math.fsum(amounts.values()), "commitments_usd": math.fsum(c.usd for c in new),
                "calls": calls, "liquid_close": liquid.balance, "private_close": private_close,
                "total_close": liquid.balance + private_close,
                "private_valuation_pnl": private_close - private_open - calls + distributions,
                "fx_translation": fx_translation,
            })
            for c in book.commitments:
                fund_rows.append({
                    "date": stamp, "fund": c.fund.name, "fund_type": c.fund.fund_type,
                    "commitment_usd": c.usd, "calls_usd": c.calls(t), "distributions_usd": c.distributions(t),
                    "nav_usd": c.nav(t), "calls_base": c.calls(t) * fx[t],
                    "distributions_base": c.distributions(t) * fx[t], "nav_base": c.nav(t) * fx[t],
                })
            private_open = private_close

            if missing > self.cash_tolerance and shortfall is None:
                by_fund = pd.Series(
                    {c.fund.name: c.calls(t) * fx[t] for c in book.commitments},
                    dtype=float, name="calls_base",
                )
                by_fund.index.name = "fund"
                shortfall = Shortfall(t, day, calls - missing, calls, by_fund)
                if self.stop_on_shortfall:
                    break

        return SimulationResult(
            self.portfolio.base_currency,
            _frame(period_rows, PERIOD_COLUMNS, ["date"]),
            _frame(fund_rows, FUND_COLUMNS, ["date", "fund"]),
            _frame(commitment_rows, COMMITMENT_COLUMNS, ["date", "fund"]),
            shortfall,
            self.beyond_horizon,
        )

    # ---------------------------------------------------------------- audit
    def event_map(self) -> pd.DataFrame:
        """Where every fund event lands: one row per fund and event day.

        The liquid index sets the observation frequency; a flow or mark dated on any day
        pools onto the first observation on or after it — a call on the 15th lands on that
        month's end on a month-end grid, on the next business day on a daily grid.
        ``observation_date`` is NaT and ``period`` is ``n`` for events after the last
        observation, which the simulation ignores.
        """
        rows: list[dict[str, Any]] = []
        for fund in self.funds:
            events = fund.events()
            periods = self.timeline.assign([day for day, *_ in events])
            for (day, call, distribution, mark), t in zip(events, periods):
                rows.append({
                    "fund": fund.name, "event_date": day,
                    "observation_date": self.timeline.dates[t] if t < self.timeline.n else pd.NaT,
                    "period": int(t), "unit_call": call, "unit_distribution": distribution,
                    "unit_nav_mark": float("nan") if mark is None else mark,
                })
        return _frame(rows, EVENT_COLUMNS, ["fund", "event_date"])

    # -------------------------------------------------------------- helpers
    def _size(self, cohort: Sequence[Fund], base: SizingBase) -> dict[str, float]:
        if not cohort:
            return {}
        try:
            sized: Mapping[str, float] = self.policy.size(cohort, base)
        except KeyError as exc:
            raise ValueError(f"policy has no sizing for fund {exc}") from None
        unknown = set(sized) - {f.name for f in cohort}
        if unknown:
            raise ValueError(f"policy sized funds that are not closing now: {sorted(unknown)}")
        amounts = {}
        for fund in cohort:
            amount = float(sized.get(fund.name, 0.0))
            if not math.isfinite(amount) or amount < 0:
                raise ValueError(f"policy returned an invalid commitment {amount!r} for {fund.name!r}")
            amounts[fund.name] = amount
        return amounts

    def _commitment_row(
        self, stamp: pd.Timestamp, fund: Fund, base: SizingBase, amount: float, rate: float, usd: float
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "date": stamp, "fund": fund.name, "fund_type": fund.fund_type,
            "closing_date": pd.Timestamp(fund.closing_date), "policy_year": fund.closing_year,
            "sizing_base": base.liquid, "rate": amount / base.liquid if base.liquid else float("nan"),
            "commitment_base": amount, "usd_rate": float(rate), "commitment_usd": usd,
            "current_year_rate": float("nan"), "carried_rate": float("nan"),
            "pooled_rate": float("nan"), "weight": float("nan"),
        }
        explain = getattr(self.policy, "explain", None)
        if callable(explain):
            info = explain(fund.name)
            for key in ("current_year_rate", "carried_rate", "pooled_rate", "weight"):
                if key in info:
                    row[key] = float(info[key])
            if "effective_rate" in info:
                row["rate"] = float(info["effective_rate"])
        return row
