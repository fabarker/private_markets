"""The one loop.

``Simulator(portfolio, funds).run()`` walks the liquid index's observations. Each period:

1. snapshot the opening balances;
2. apply the liquid return, the period's percent change (``level[t] / level[t-1] − 1``; 0 at ``t = 0``);
3. bank distributions from existing commitments;
4. size and fix commitments for the funds closing at this observation — in US dollars, on
   the liquid-only value: the initial value compounded by the liquid returns, converted at
   today's rate. No call or distribution is in it, so every commitment follows from the
   liquid returns, the initial value, the exchange rates and the schedule alone;
5. bank the new cohort's own distributions, then pay every commitment's calls;
6. value the book, record the period, and stop if the pot could not cover the calls.

Private figures stay in USD until they touch the liquid pot or a report, and commitments are
sized in USD, never in base currency; the exchange rate of the observation date is applied
at that boundary and nowhere else.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .benchmark import compare_with_liquid_only, public_market_equivalent
from .inputs import Fund, Portfolio
from .policy import AnnualRatePolicy, CommitmentPolicy, SizingBalances, YearEndBalance
from .state import Commitment, LiquidAccount, CommitmentBook
from .timeline import AlignedFundHistory, Timeline

PERIOD_COLUMNS = [
    "liquid_open", "private_open", "total_open", "period_return", "usd_rate",
    "liquid_pnl", "distributions", "sizing_base", "sizing_base_usd", "commitments", "commitments_usd", "calls",
    "liquid_close", "private_close", "total_close", "private_valuation_pnl", "fx_translation",
]
FUND_COLUMNS = [
    "fund_type", "commitment_usd", "calls_usd", "distributions_usd", "nav_usd",
    "calls_base", "distributions_base", "nav_base",
]
COMMITMENT_COLUMNS = [
    "fund_type", "closing_date", "policy_year", "sizing_base_usd", "rate", "commitment_usd",
    "usd_rate", "sizing_base", "commitment_base",
    "current_year_rate", "expected_value", "weight", "current_year_usd", "carried_usd", "carried_years",
]
EVENT_COLUMNS = ["observation_date", "period", "unit_call", "unit_distribution", "unit_nav_mark"]
_TEXT_COLUMNS = {"fund", "fund_type", "carried_years"}
_DATE_COLUMNS = {"date", "closing_date", "event_date", "observation_date"}
_INT_COLUMNS = {"policy_year", "period"}


def _build_table(rows: list[dict[str, Any]], columns: list[str], index: list[str]) -> pd.DataFrame:
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
    cash_available: float
    calls_due: float
    calls_by_fund: pd.Series

    @property
    def deficit(self) -> float:
        return self.calls_due - self.cash_available

    def __str__(self) -> str:
        return (
            f"shortfall of {self.deficit:,.2f} on {self.date}: calls of {self.calls_due:,.2f} "
            f"against {self.cash_available:,.2f} available"
        )


@dataclass(frozen=True, eq=False)
class SimulationResult:
    """Two tidy tables, a commitment log and a verdict.

    ``periods`` (index: date) is in base currency except the ``_usd`` columns and ``usd_rate``.
    ``funds`` (index: date, fund) has one row per live commitment per period, in both
    currencies. ``sizing_base`` is the liquid-only value — the initial value compounded by the
    liquid returns, untouched by any call or distribution — and ``sizing_base_usd`` the same in
    dollars, which is what commitments are sized on. ``commitments`` (index: date, fund) has one
    row per closing: that USD value, the dollars committed and their share of it (``rate``), the
    exchange rate and the base-currency equivalents, then how ``AnnualRatePolicy`` got there:
    ``commitment_usd = weight × (current_year_usd + carried_usd)``, with ``current_year_usd =
    current_year_rate / expected_value × sizing_base_usd``. ``shortfall`` names the first failed
    observation, or is ``None``. ``funds_beyond_horizon`` lists funds whose closing falls
    after the last observation; they are never committed.
    """

    base_currency: str
    periods: pd.DataFrame
    funds: pd.DataFrame
    commitments: pd.DataFrame
    shortfall: Shortfall | None
    funds_beyond_horizon: tuple[str, ...]

    @property
    def status(self) -> str:
        return "shortfall" if self.shortfall is not None else "completed"

    def totals_by_fund_type(self) -> pd.DataFrame:
        """Fund-level figures summed by date and fund type."""
        columns = [c for c in FUND_COLUMNS if c != "fund_type"]
        if self.funds.empty:
            index = pd.MultiIndex.from_arrays([pd.DatetimeIndex([]), []], names=["date", "fund_type"])
            return pd.DataFrame(columns=columns, index=index, dtype=float)
        return self.funds.reset_index().groupby(["date", "fund_type"])[columns].sum()

    def nav_by_fund(self) -> pd.DataFrame:
        """Private NAV in base currency by date (rows) and fund (columns); zero before a fund closes."""
        if self.funds.empty:
            return pd.DataFrame(index=self.periods.index, dtype=float)
        return self.funds["nav_base"].unstack("fund").reindex(self.periods.index).fillna(0.0)

    def compare_with_liquid_only(self) -> pd.DataFrame:
        """This run beside the same liquid portfolio with no private programme: both paths and the value added."""
        return compare_with_liquid_only(self.periods)

    def public_market_equivalent(self) -> pd.DataFrame:
        """KS-PME, IRR and direct alpha against the liquid portfolio, for the programme, each fund type and each fund."""
        return public_market_equivalent(self.periods, self.funds)


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
        if (isinstance(cash_tolerance, bool) or not isinstance(cash_tolerance, Real)
                or not math.isfinite(cash_tolerance) or cash_tolerance < 0):
            raise ValueError("cash_tolerance must be a finite non-negative number")

        self.portfolio = portfolio
        self.funds: tuple[Fund, ...] = tuple(funds)
        self.timeline = Timeline(portfolio.liquid_levels.index)
        first = self.timeline.observation_date(0)
        for fund in funds:
            if fund.closing_date < first:
                raise ValueError(
                    f"Fund {fund.name!r} closes on {fund.closing_date}, before the first observation "
                    f"{first}; there are no pre-existing commitments"
                )
        levels = portfolio.liquid_levels.to_numpy(dtype=float)
        # the period return as a percent change; the first period has no preceding observation, so 0
        self.returns: np.ndarray = np.concatenate([[0.0], levels[1:] / levels[:-1] - 1.0])
        self.fx: np.ndarray = (
            self.timeline.last_value_on_or_before(portfolio.usd_rate, name="usd_rate")
            if portfolio.requires_fx_conversion
            else np.ones(self.timeline.n_observations)
        )
        # What commitments are sized on: the initial value compounded by the liquid returns, which is the level
        # series itself. It is known before the run starts, and no call or distribution ever touches it.
        self.liquid_only: np.ndarray = levels
        self.liquid_only_usd: np.ndarray = levels / self.fx
        self.aligned_histories: dict[str, AlignedFundHistory] = {f.name: f.align_history(self.timeline) for f in funds}
        self.policy: CommitmentPolicy = (
            policy if policy is not None
            else AnnualRatePolicy(portfolio.commitment_rates, funds, years=self.timeline.calendar_years)
        )
        self.stop_on_shortfall = bool(stop_on_shortfall)
        self.cash_tolerance = float(cash_tolerance)
        beyond_horizon: list[str] = []
        self._closings_by_period: dict[int, list[Fund]] = {}
        for fund in funds:
            history = self.aligned_histories[fund.name]
            if history.closes_beyond_horizon:
                beyond_horizon.append(fund.name)
            else:
                self._closings_by_period.setdefault(history.closing_period, []).append(fund)
        self.funds_beyond_horizon: tuple[str, ...] = tuple(beyond_horizon)
        # The pacing model's value is 1 on the day of the first commitment.
        self.first_commitment_date: date | None = (
            self.timeline.observation_date(min(self._closings_by_period)) if self._closings_by_period else None
        )
        last_period_of_year = {self.timeline.observation_date(t).year: t for t in range(self.timeline.n_observations)}
        self._year_ends: dict[int, YearEndBalance] = {  # at each calendar year's last observation
            year: YearEndBalance(self.timeline.observation_date(t), float(self.liquid_only_usd[t]))
            for year, t in last_period_of_year.items()
        }

    # ------------------------------------------------------------------ run
    def run(self) -> SimulationResult:
        timeline, fx, returns = self.timeline, self.fx, self.returns
        liquid = LiquidAccount(float(self.portfolio.liquid_levels.iloc[0]), returns)
        book = CommitmentBook()
        period_rows: list[dict[str, Any]] = []
        fund_rows: list[dict[str, Any]] = []
        commitment_rows: list[dict[str, Any]] = []
        shortfall: Shortfall | None = None
        private_open = 0.0

        for t in range(timeline.n_observations):
            day = timeline.observation_date(t)
            stamp = pd.Timestamp(day)

            liquid_open = liquid.balance                                            # 1
            nav_usd_open = book.nav_at(t - 1)

            pnl = liquid.apply_return(t)                                            # 2

            distributions_existing = book.distributions_in_period(t) * fx[t]        # 3
            liquid.deposit(distributions_existing)

            cohort = self._closings_by_period.get(t, [])                            # 4
            completed_years = {year: year_end for year, year_end in self._year_ends.items() if year < day.year}
            balances = SizingBalances(  # USD in, USD out; sized on the liquid-only value, never on the account
                t, day, liquid_only_usd=float(self.liquid_only_usd[t]), liquid_account_usd=liquid.balance / fx[t],
                private_nav_usd=nav_usd_open, year_ends=completed_years, first_commitment_date=self.first_commitment_date,
            )
            commitments_usd = self._size_commitments(cohort, balances)
            committed_usd = math.fsum(commitments_usd.values())
            for fund in cohort:
                book.add(Commitment(fund, self.aligned_histories[fund.name], commitments_usd[fund.name]))
                commitment_rows.append(
                    self._commitment_row(stamp, fund, balances, commitments_usd[fund.name], float(self.liquid_only[t]), fx[t]))

            new = book.commitments_closing_in(t)                                    # 5
            distributions_new = math.fsum(c.distributions_in_period(t) for c in new) * fx[t]
            liquid.deposit(distributions_new)
            calls = book.calls_in_period(t) * fx[t]
            missing = liquid.withdraw(calls)

            private_close = book.nav_at(t) * fx[t]                                  # 6
            distributions = distributions_existing + distributions_new
            fx_translation = nav_usd_open * (fx[t] - fx[t - 1]) if t > 0 else 0.0
            period_rows.append({
                "date": stamp,
                "liquid_open": liquid_open, "private_open": private_open, "total_open": liquid_open + private_open,
                "period_return": float(returns[t]), "usd_rate": float(fx[t]),
                "liquid_pnl": pnl, "distributions": distributions,
                "sizing_base": float(self.liquid_only[t]), "sizing_base_usd": balances.liquid_only_usd,
                "commitments": committed_usd * fx[t], "commitments_usd": committed_usd,
                "calls": calls, "liquid_close": liquid.balance, "private_close": private_close,
                "total_close": liquid.balance + private_close,
                "private_valuation_pnl": private_close - private_open - calls + distributions,
                "fx_translation": fx_translation,
            })
            for c in book.commitments:
                calls_usd, distributions_usd, nav_usd = c.calls_in_period(t), c.distributions_in_period(t), c.nav_at(t)
                fund_rows.append({
                    "date": stamp, "fund": c.fund.name, "fund_type": c.fund.fund_type,
                    "commitment_usd": c.usd, "calls_usd": calls_usd, "distributions_usd": distributions_usd,
                    "nav_usd": nav_usd, "calls_base": calls_usd * fx[t],
                    "distributions_base": distributions_usd * fx[t], "nav_base": nav_usd * fx[t],
                })
            private_open = private_close

            if missing > self.cash_tolerance and shortfall is None:
                by_fund = pd.Series(
                    {c.fund.name: c.calls_in_period(t) * fx[t] for c in book.commitments},
                    dtype=float, name="calls_base",
                )
                by_fund.index.name = "fund"
                shortfall = Shortfall(t, day, calls - missing, calls, by_fund)
                if self.stop_on_shortfall:
                    break

        return SimulationResult(
            self.portfolio.base_currency,
            _build_table(period_rows, PERIOD_COLUMNS, ["date"]),
            _build_table(fund_rows, FUND_COLUMNS, ["date", "fund"]),
            _build_table(commitment_rows, COMMITMENT_COLUMNS, ["date", "fund"]),
            shortfall,
            self.funds_beyond_horizon,
        )

    # ---------------------------------------------------------------- audit
    def map_events_to_observations(self) -> pd.DataFrame:
        """Where every fund event lands: one row per fund and event day.

        The liquid index sets the observation frequency; a flow or mark dated on any day
        pools onto the first observation on or after it — a call on the 15th lands on that
        month's end on a month-end grid, on the next business day on a daily grid.
        ``observation_date`` is NaT and ``period`` is ``n_observations`` for events after
        the last observation, which the simulation ignores.
        """
        rows: list[dict[str, Any]] = []
        for fund in self.funds:
            events = fund.events_by_day()
            periods = self.timeline.first_observations_on_or_after([day for day, *_ in events])
            for (day, call, distribution, mark), t in zip(events, periods):
                rows.append({
                    "fund": fund.name, "event_date": day,
                    "observation_date": self.timeline.dates[t] if t < self.timeline.n_observations else pd.NaT,
                    "period": int(t), "unit_call": call, "unit_distribution": distribution,
                    "unit_nav_mark": float("nan") if mark is None else mark,
                })
        return _build_table(rows, EVENT_COLUMNS, ["fund", "event_date"])

    # -------------------------------------------------------------- helpers
    def _size_commitments(self, cohort: Sequence[Fund], balances: SizingBalances) -> dict[str, float]:
        """The policy's US-dollar commitment for every fund in the cohort, checked before it reaches the book."""
        if not cohort:
            return {}
        try:
            sized: Mapping[str, float] = self.policy.size_commitments(cohort, balances)
        except KeyError as exc:
            raise ValueError(f"policy has no sizing for fund {exc}") from None
        unknown = set(sized) - {f.name for f in cohort}
        if unknown:
            raise ValueError(f"policy sized funds that are not closing now: {sorted(unknown)}")
        commitments_usd = {}
        for fund in cohort:
            dollars = float(sized.get(fund.name, 0.0))
            if not math.isfinite(dollars) or dollars < 0:
                raise ValueError(f"policy returned an invalid commitment {dollars!r} for {fund.name!r}")
            commitments_usd[fund.name] = dollars
        return commitments_usd

    def _commitment_row(
        self, stamp: pd.Timestamp, fund: Fund, balances: SizingBalances, commitment_usd: float,
        liquid_only: float, usd_rate: float,
    ) -> dict[str, Any]:
        """One closing: what was decided in dollars, then the same figures in base currency at that day's rate."""
        row: dict[str, Any] = {
            "date": stamp, "fund": fund.name, "fund_type": fund.fund_type,
            "closing_date": pd.Timestamp(fund.closing_date), "policy_year": fund.closing_year,
            "sizing_base_usd": balances.liquid_only_usd,
            "rate": commitment_usd / balances.liquid_only_usd if balances.liquid_only_usd else float("nan"),
            "commitment_usd": commitment_usd, "usd_rate": float(usd_rate),
            "sizing_base": liquid_only, "commitment_base": commitment_usd * usd_rate,
            "current_year_rate": float("nan"), "expected_value": float("nan"), "weight": float("nan"),
            "current_year_usd": float("nan"), "carried_usd": float("nan"), "carried_years": "",
        }
        explain_commitment = getattr(self.policy, "explain_commitment", None)  # optional: a policy may say how it got there
        if callable(explain_commitment):
            info = explain_commitment(fund.name, balances)
            for key in ("current_year_rate", "expected_value", "weight", "current_year_usd", "carried_usd"):
                if key in info:
                    row[key] = float(info[key])
            row["carried_years"] = str(info.get("carried_years", ""))
        return row
