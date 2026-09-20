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
from .dates import years_between
from .inputs import Fund, Portfolio
from .policy import AnnualRatePolicy, CommitmentPolicy, SizingBalances, YearEndBalance
from .state import Commitment, LiquidAccount, CommitmentBook
from .timeline import AlignedFundHistory, Timeline

PERIOD_COLUMNS = [
    # the five running values, in base currency
    "liquid_only", "expected_liquid", "liquid_close", "private_close", "total_close",
    # the same three, as the period opened
    "liquid_open", "private_open", "total_open",
    # what moved them
    "period_return", "liquid_pnl", "distributions", "calls", "private_valuation_pnl", "usd_rate", "fx_translation",
    # what was committed, and the dollars it was sized on
    "liquid_only_usd", "commitments", "commitments_usd",
]
TRACKED_COLUMNS = PERIOD_COLUMNS[:5]  # the five running values, in the order SimulationResult.tracked_values reports them
FUND_COLUMNS = [
    "fund_type", "commitment_usd", "calls_usd", "distributions_usd", "nav_usd",
    "calls_base", "distributions_base", "nav_base",
]
COMMITMENT_COLUMNS = [
    "fund_type", "closing_date", "policy_year", "sizing_base_usd", "rate", "commitment_usd",
    "usd_rate", "sizing_base", "commitment_base",
    "own_year_rate", "expected_value", "weight", "own_year_usd", "other_years_usd", "drawn_years",
]
DRAW_COLUMNS = [  # one row per schedule year a fund draws: the audit trail behind its commitment
    "fund_type", "policy_year", "multiplier", "rate", "plan_date", "expected_value",
    "funding_date", "liquid_only_usd", "year_budget_unrounded_usd", "year_budget_usd",
    "commitment_usd", "usd_rate", "commitment_base",
]
EVENT_COLUMNS = ["observation_date", "period", "unit_call", "unit_distribution", "unit_nav_mark"]
_TEXT_COLUMNS = {"fund", "fund_type", "drawn_years"}
_DATE_COLUMNS = {"date", "closing_date", "event_date", "observation_date", "plan_date", "funding_date"}
_INT_COLUMNS = {"policy_year", "period", "year"}


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
    currencies, and its first five columns are the running values ``tracked_values()`` returns.
    ``liquid_only`` is the liquid portfolio alone — the initial value compounded by the liquid
    returns, untouched by any call or distribution — and ``liquid_only_usd`` the same in dollars,
    which is what commitments are sized on. ``commitments`` (index: date, fund) has one
    row per closing: that USD value, the dollars committed and their share of it (``rate``), the
    exchange rate and the base-currency equivalents, then how ``AnnualRatePolicy`` got there:
    ``commitment_usd = weight × (own_year_usd + other_years_usd)``, and ``drawn_years`` naming
    every schedule year that went into it. ``draws`` (index: date, fund, year) breaks each
    commitment down one drawn year at a time — its rate, the two dates behind it (``plan_date``
    normalises the rate, ``funding_date`` supplies the liquid value), the year's dollar
    commitment as computed and as rounded (``year_budget_unrounded_usd``, ``year_budget_usd``)
    and the dollars it contributed, which sum to the fund's ``commitment_usd``. ``shortfall`` names the first
    failed observation, or is ``None``. ``funds_beyond_horizon`` lists funds whose closing falls
    after the last observation; they are never committed.
    """

    base_currency: str
    periods: pd.DataFrame
    funds: pd.DataFrame
    commitments: pd.DataFrame
    draws: pd.DataFrame
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

    def tracked_values(self) -> pd.DataFrame:
        """The five running values, by observation, in base currency.

        ``liquid_only``      the liquid portfolio alone: the initial value compounded by the liquid
                             returns, with no capital call or distribution in it
        ``expected_liquid``  that same starting value growing at the policy's expected return
                             instead; NaN when the policy has no expected return
        ``liquid_close``     the liquid account as it really stands, calls paid and distributions banked
        ``private_close``    the private book, at its marks
        ``total_close``      ``liquid_close + private_close``: everything the investor holds
        """
        return self.periods[TRACKED_COLUMNS]

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
        # The second of the five tracked values: the same starting value growing at the policy's
        # expected return instead of at the market's. NaN when the policy has no expected return.
        self.expected_liquid: np.ndarray = self._expected_liquid_path(float(levels[0]))

    def _expected_liquid_path(self, initial_value: float) -> np.ndarray:
        """The liquid portfolio had it grown at the expected return from the first observation."""
        expected_return = getattr(self.policy, "expected_return", None)
        if expected_return is None:
            return np.full(self.timeline.n_observations, np.nan)
        inception = self.timeline.observation_date(0)
        elapsed = np.array([years_between(inception, self.timeline.observation_date(t))
                            for t in range(self.timeline.n_observations)])
        return initial_value * (1.0 + expected_return) ** elapsed

    # ------------------------------------------------------------------ run
    def run(self) -> SimulationResult:
        """Walk the observations once, from a fresh state, recording what happened at each.

        The six numbered steps below are the whole engine. Their order is the cash
        convention and it is deliberate: the liquid portfolio earns its return before any
        private cash moves, commitments are sized before the cohort's own distributions
        arrive, and the calls are paid last of all.
        """
        liquid = LiquidAccount(float(self.portfolio.liquid_levels.iloc[0]), self.returns)
        book = CommitmentBook()

        period_rows: list[dict[str, Any]] = []
        fund_rows: list[dict[str, Any]] = []
        commitment_rows: list[dict[str, Any]] = []
        draw_rows: list[dict[str, Any]] = []
        shortfall: Shortfall | None = None
        private_open = 0.0  # private NAV in base currency, as it stood at the previous close

        for t in range(self.timeline.n_observations):
            day = self.timeline.observation_date(t)
            rate = float(self.fx[t])  # base currency per 1 USD, on this observation

            # 1 ── What is held as the period opens.
            liquid_open = liquid.balance
            private_nav_usd_open = book.nav_at(t - 1)

            # 2 ── The liquid portfolio earns this period's return.
            liquid_pnl = liquid.apply_return(t)

            # 3 ── Funds committed before today distribute; the cash lands in the account.
            distributions_existing = book.distributions_in_period(t) * rate
            liquid.deposit(distributions_existing)

            # 4 ── Funds closing today are committed — in US dollars, on the liquid-only
            #      value, all from the same balances. A commitment promises cash; it moves none.
            cohort = self._closings_by_period.get(t, [])
            balances = self._sizing_balances(t, day, liquid.balance / rate, private_nav_usd_open)
            dollars_by_fund = self._size_commitments(cohort, balances)

            for fund in cohort:
                book.add(Commitment(fund, self.aligned_histories[fund.name], dollars_by_fund[fund.name]))
                commitment_rows.append(self._commitment_row(t, day, fund, balances, dollars_by_fund[fund.name], rate))
                draw_rows.extend(self._draw_rows(day, fund, balances, rate))

            # 5 ── The new cohort's own distributions are banked — they could not be in step 3,
            #      since those funds did not exist yet — and then every call is paid.
            distributions_new = math.fsum(c.distributions_in_period(t) for c in book.commitments_closing_in(t)) * rate
            liquid.deposit(distributions_new)

            calls = book.calls_in_period(t) * rate
            unpaid = liquid.withdraw(calls)  # what the account could not cover, 0 when it could

            # 6 ── Value the book and record the observation.
            private_close = book.nav_at(t) * rate
            distributions = distributions_existing + distributions_new
            committed_usd = math.fsum(dollars_by_fund.values())

            period_rows.append({
                "date": pd.Timestamp(day),

                # the five running values: the liquid portfolio three ways, the private book, the total
                "liquid_only": float(self.liquid_only[t]),           # 1 · liquid alone, no private flow in it
                "expected_liquid": float(self.expected_liquid[t]),   # 2 · liquid at the expected return
                "liquid_close": liquid.balance,                      # 3 · liquid as it really stands
                "private_close": private_close,                      # 4 · the private book at its marks
                "total_close": liquid.balance + private_close,       # 5 · 3 + 4

                # the same three, as the period opened, so each period reconciles
                "liquid_open": liquid_open,
                "private_open": private_open,
                "total_open": liquid_open + private_open,

                # what moved it: the market, the private flows, and the exchange rate
                "period_return": float(self.returns[t]),
                "liquid_pnl": liquid_pnl,
                "distributions": distributions,
                "calls": calls,
                "private_valuation_pnl": private_close - private_open - calls + distributions,
                "usd_rate": rate,
                "fx_translation": private_nav_usd_open * (rate - float(self.fx[t - 1])) if t > 0 else 0.0,

                # what was committed today, and the dollars it was sized on
                "liquid_only_usd": balances.liquid_only_usd,
                "commitments": committed_usd * rate,
                "commitments_usd": committed_usd,
            })
            fund_rows.extend(self._fund_rows(t, day, book, rate))
            private_open = private_close

            # ... and stop here if the account could not meet the calls.
            if unpaid > self.cash_tolerance and shortfall is None:
                shortfall = self._shortfall(t, day, book, calls, unpaid, rate)
                if self.stop_on_shortfall:
                    break

        return SimulationResult(
            self.portfolio.base_currency,
            _build_table(period_rows, PERIOD_COLUMNS, ["date"]),
            _build_table(fund_rows, FUND_COLUMNS, ["date", "fund"]),
            _build_table(commitment_rows, COMMITMENT_COLUMNS, ["date", "fund"]),
            _build_table(draw_rows, DRAW_COLUMNS, ["date", "fund", "year"]),
            shortfall,
            self.funds_beyond_horizon,
        )

    # ------------------------------------------------------- what the loop records
    def _sizing_balances(self, t: int, day: date, liquid_account_usd: float,
                         private_nav_usd: float) -> SizingBalances:
        """What the policy is shown: US dollars throughout, and the liquid-only value to size on."""
        return SizingBalances(
            t, day,
            liquid_only_usd=float(self.liquid_only_usd[t]),
            liquid_account_usd=liquid_account_usd,
            private_nav_usd=private_nav_usd,
            year_ends={year: end for year, end in self._year_ends.items() if year < day.year},
            first_commitment_date=self.first_commitment_date,
        )

    def _fund_rows(self, t: int, day: date, book: CommitmentBook, rate: float) -> list[dict[str, Any]]:
        """One row per live commitment: this period's figures, in dollars and in base currency."""
        rows = []
        for commitment in book.commitments:
            calls_usd = commitment.calls_in_period(t)
            distributions_usd = commitment.distributions_in_period(t)
            nav_usd = commitment.nav_at(t)
            rows.append({
                "date": pd.Timestamp(day),
                "fund": commitment.fund.name,
                "fund_type": commitment.fund.fund_type,
                "commitment_usd": commitment.usd,
                "calls_usd": calls_usd,
                "distributions_usd": distributions_usd,
                "nav_usd": nav_usd,
                "calls_base": calls_usd * rate,
                "distributions_base": distributions_usd * rate,
                "nav_base": nav_usd * rate,
            })
        return rows

    def _shortfall(self, t: int, day: date, book: CommitmentBook,
                   calls: float, unpaid: float, rate: float) -> Shortfall:
        """The first observation whose calls the account could not meet, with the calls that caused it."""
        calls_by_fund = pd.Series(
            {c.fund.name: c.calls_in_period(t) * rate for c in book.commitments},
            dtype=float, name="calls_base",
        )
        calls_by_fund.index.name = "fund"
        return Shortfall(t, day, cash_available=calls - unpaid, calls_due=calls, calls_by_fund=calls_by_fund)

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

    def _commitment_row(self, t: int, day: date, fund: Fund, balances: SizingBalances,
                        commitment_usd: float, rate: float) -> dict[str, Any]:
        """One closing: what was decided in dollars, then the same figures in base currency at that day's rate."""
        row: dict[str, Any] = {
            "date": pd.Timestamp(day),
            "fund": fund.name,
            "fund_type": fund.fund_type,
            "closing_date": pd.Timestamp(fund.closing_date),
            "policy_year": fund.closing_year,

            # the decision, in dollars
            "sizing_base_usd": balances.liquid_only_usd,
            "rate": commitment_usd / balances.liquid_only_usd if balances.liquid_only_usd else float("nan"),
            "commitment_usd": commitment_usd,

            # and the same, translated at today's rate for reporting
            "usd_rate": rate,
            "sizing_base": float(self.liquid_only[t]),
            "commitment_base": commitment_usd * rate,

            # filled in below by policies that can say how they got there
            "own_year_rate": float("nan"), "expected_value": float("nan"), "weight": float("nan"),
            "own_year_usd": float("nan"), "other_years_usd": float("nan"), "drawn_years": "",
        }
        explain_commitment = getattr(self.policy, "explain_commitment", None)  # optional: a policy may say how it got there
        if callable(explain_commitment):
            info = explain_commitment(fund.name, balances)
            for key in ("own_year_rate", "expected_value", "weight", "own_year_usd", "other_years_usd"):
                if key in info:
                    row[key] = float(info[key])
            row["drawn_years"] = str(info.get("drawn_years", ""))
        return row

    def _draw_rows(self, day: date, fund: Fund, balances: SizingBalances, rate: float) -> list[dict[str, Any]]:
        """One row per schedule year the fund draws — empty for a policy that cannot break a commitment down."""
        drawn_years = getattr(self.policy, "drawn_years", None)  # optional, like explain_commitment
        if not callable(drawn_years):
            return []
        weight = self.policy.entitlements[fund.name].weight
        return [{
            "date": pd.Timestamp(day), "fund": fund.name, "year": drawn.year,
            "fund_type": fund.fund_type, "policy_year": fund.closing_year,
            "multiplier": drawn.multiplier, "rate": drawn.rate,
            "plan_date": pd.Timestamp(drawn.plan_date), "expected_value": drawn.expected_value,
            "funding_date": pd.Timestamp(drawn.funding_date), "liquid_only_usd": drawn.liquid_only_usd,
            # the year's own dollar commitment, as computed and then as rounded
            "year_budget_unrounded_usd": drawn.year_budget_unrounded_usd, "year_budget_usd": drawn.year_budget_usd,
            # the fund's weight is applied here, so these dollars add up to its commitment
            "commitment_usd": weight * drawn.commitment_usd,
            "usd_rate": rate, "commitment_base": weight * drawn.commitment_usd * rate,
        } for drawn in drawn_years(fund.name, balances)]
