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
from .state import Commitment, CommitmentBook, LiquidAccount
from .timeline import AlignedFundHistory, Timeline

# ------------------------------------------------------------------ the result tables
PERIOD_COLUMNS = [
    # the five running values, in base currency
    "liquid_only", "expected_liquid", "liquid_close", "private_close", "total_close",
    # the same three, as the period opened
    "liquid_open", "private_open", "total_open",
    # what moved them
    "period_return", "liquid_pnl", "distributions", "calls", "private_valuation_pnl",
    "usd_rate", "fx_translation",
    # what was committed, and the dollars it was sized on
    "liquid_only_usd", "commitments", "commitments_usd",
]

# The five running values, in the order SimulationResult.tracked_values reports them.
TRACKED_COLUMNS = PERIOD_COLUMNS[:5]


# After those fixed columns, every period row also carries the market value (NAV) of each fund
# type and of each fund, in US dollars and in base currency. Which columns those are depends on
# the funds in the run, so their names are built by these two functions and nowhere else.
def fund_type_market_value_columns(fund_type: str) -> tuple[str, str]:
    """The two period columns holding one fund type's total NAV: in USD, then in base currency."""
    return f"{fund_type}_total_nav_usd", f"{fund_type}_total_nav_base"


def fund_market_value_columns(fund_name: str) -> tuple[str, str]:
    """The two period columns holding one fund's NAV: in USD, then in base currency."""
    return f"{fund_name}_nav_usd", f"{fund_name}_nav_base"

FUND_COLUMNS = [
    "fund_type", "commitment_usd", "calls_usd", "distributions_usd", "nav_usd",
    "calls_base", "distributions_base", "nav_base",
]

COMMITMENT_COLUMNS = [
    "fund_type", "closing_date", "policy_year", "sizing_base_usd", "rate", "commitment_usd",
    "usd_rate", "sizing_base", "commitment_base",
    "own_year_rate", "expected_value", "weight", "own_year_usd", "other_years_usd", "drawn_years",
]

# One row per schedule year a fund draws: the audit trail behind its commitment.
DRAW_COLUMNS = [
    "fund_type", "policy_year", "multiplier", "rate", "sizing_date", "looks_ahead", "expected_value",
    "liquid_only_usd", "year_budget_unrounded_usd", "year_budget_usd",
    "commitment_usd", "usd_rate", "commitment_base",
]

EVENT_COLUMNS = ["observation_date", "period", "unit_call", "unit_distribution", "unit_nav_mark"]

# The figures a policy may report through explain_commitment, as numbers.
EXPLAINED_NUMBER_COLUMNS = ("own_year_rate", "expected_value", "weight", "own_year_usd", "other_years_usd")

_TEXT_COLUMNS = {"fund", "fund_type", "drawn_years"}
_DATE_COLUMNS = {"date", "closing_date", "event_date", "observation_date", "sizing_date"}
_INT_COLUMNS = {"policy_year", "period", "year"}
_TRUE_OR_FALSE_COLUMNS = {"looks_ahead"}


def _build_table(rows: list[dict[str, Any]], columns: list[str], index: list[str]) -> pd.DataFrame:
    """Build a table with fixed columns and dtypes, correct even when ``rows`` is empty."""
    frame = pd.DataFrame(rows, columns=index + columns)

    for column in frame.columns:
        if column in _DATE_COLUMNS:
            frame[column] = pd.to_datetime(frame[column])

        elif column in _INT_COLUMNS:
            frame[column] = frame[column].astype("int64")

        elif column in _TRUE_OR_FALSE_COLUMNS:
            frame[column] = frame[column].astype(bool)

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
    """Four tidy tables and a verdict.

    ``periods`` (index: date) is in base currency except the ``_usd`` columns and
    ``usd_rate``. Its first five columns are the running values ``tracked_values()`` returns.
    ``liquid_only`` is the liquid portfolio alone — the initial value compounded by the liquid
    returns, untouched by any call or distribution — and ``liquid_only_usd`` the same in
    dollars, which is what commitments are sized on. After the fixed ``PERIOD_COLUMNS`` come
    the market values ``market_values()`` returns: for each fund type
    ``<TYPE>_total_nav_usd`` and ``<TYPE>_total_nav_base``, then for each fund
    ``<fund>_nav_usd`` and ``<fund>_nav_base``; zero until a fund is committed.

    ``funds`` (index: date, fund) has one row per live commitment per period, in both
    currencies.

    ``commitments`` (index: date, fund) has one row per closing: the USD value it was sized
    on, the dollars committed and their share of it (``rate``), the exchange rate and the
    base-currency equivalents, then how ``AnnualRatePolicy`` got there:
    ``commitment_usd = weight × (own_year_usd + other_years_usd)``, and ``drawn_years`` naming
    every schedule year that went into it.

    ``draws`` (index: date, fund, year) breaks each commitment down one drawn year at a time —
    its rate, the ``sizing_date`` it was priced on and the expected value and liquid-only value
    on that date, the year's dollar commitment as computed and as rounded
    (``year_budget_unrounded_usd``, ``year_budget_usd``) and the dollars it contributed, which
    sum to the fund's ``commitment_usd``. ``looks_ahead`` is True for a year that lies after
    the fund's closing: it was priced on its own year end, with hindsight.

    ``shortfall`` names the first failed observation, or is ``None``.
    ``funds_beyond_horizon`` lists funds whose closing falls after the last observation;
    they are never committed.
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
        if self.shortfall is not None:
            return "shortfall"
        return "completed"

    def totals_by_fund_type(self) -> pd.DataFrame:
        """Fund-level figures summed by date and fund type."""
        columns = [column for column in FUND_COLUMNS if column != "fund_type"]

        # No fund was ever committed: an empty table of the right shape.
        if self.funds.empty:
            empty_index = pd.MultiIndex.from_arrays([pd.DatetimeIndex([]), []], names=["date", "fund_type"])
            return pd.DataFrame(columns=columns, index=empty_index, dtype=float)

        by_date_and_type = self.funds.reset_index().groupby(["date", "fund_type"])
        return by_date_and_type[columns].sum()

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

    def market_values(self) -> pd.DataFrame:
        """The market value (NAV) of each fund type and of each fund, by observation.

        Every value twice: ``_usd`` in US dollars, as the funds report it, and ``_base`` in the
        portfolio's base currency at that observation's exchange rate. These are the columns of
        ``periods`` that come after the fixed ``PERIOD_COLUMNS``. The fund-type totals in base
        currency add up to ``private_close``.
        """
        market_value_columns = [column for column in self.periods.columns if column not in PERIOD_COLUMNS]
        return self.periods[market_value_columns]

    def nav_by_fund(self) -> pd.DataFrame:
        """Private NAV in base currency by date (rows) and fund (columns); zero before a fund closes."""
        if self.funds.empty:
            return pd.DataFrame(index=self.periods.index, dtype=float)

        nav_with_a_column_per_fund = self.funds["nav_base"].unstack("fund")
        on_every_observation = nav_with_a_column_per_fund.reindex(self.periods.index)
        return on_every_observation.fillna(0.0)

    def compare_with_liquid_only(self) -> pd.DataFrame:
        """This run beside the same liquid portfolio with no private programme.

        Both paths, and the value the programme added.
        """
        return compare_with_liquid_only(self.periods)

    def public_market_equivalent(self) -> pd.DataFrame:
        """KS-PME, IRR and direct alpha against the liquid portfolio.

        One row for the programme, then one per fund type and one per fund.
        """
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
        funds = list(funds)
        self._check_inputs(portfolio, funds, cash_tolerance)

        self.portfolio = portfolio
        self.funds: tuple[Fund, ...] = tuple(funds)
        self.stop_on_shortfall = bool(stop_on_shortfall)
        self.cash_tolerance = float(cash_tolerance)

        # The liquid index's dates are the observations. No fund may close before the first.
        self.timeline = Timeline(portfolio.liquid_levels.index)
        self._check_no_fund_closes_before_the_first_observation(funds)

        # The liquid portfolio: its levels, its period returns, and the price of a dollar.
        levels = portfolio.liquid_levels.to_numpy(dtype=float)
        self.returns: np.ndarray = self._period_returns(levels)
        self.fx: np.ndarray = self._base_currency_per_usd()

        # What commitments are sized on: the initial value compounded by the liquid returns,
        # which is the level series itself. It is known before the run starts, and no call
        # or distribution ever touches it.
        self.liquid_only: np.ndarray = levels
        self.liquid_only_usd: np.ndarray = levels / self.fx

        # Every fund's dated history, laid onto the observations once.
        self.aligned_histories: dict[str, AlignedFundHistory] = {}
        for fund in funds:
            self.aligned_histories[fund.name] = fund.align_history(self.timeline)

        # The policy that sizes commitments: the one given, or annual rates with no extras.
        if policy is not None:
            self.policy: CommitmentPolicy = policy
        else:
            self.policy = AnnualRatePolicy(
                portfolio.commitment_rates,
                funds,
                years=self.timeline.calendar_years,
            )

        # Which funds close at which observation, and which never do within the horizon.
        self._closings_by_period: dict[int, list[Fund]] = {}
        beyond_horizon: list[str] = []

        for fund in funds:
            history = self.aligned_histories[fund.name]

            if history.closes_beyond_horizon:
                beyond_horizon.append(fund.name)
            else:
                self._closings_by_period.setdefault(history.closing_period, []).append(fund)

        self.funds_beyond_horizon: tuple[str, ...] = tuple(beyond_horizon)

        # The pacing model's value is 1 on the day of the first commitment.
        self.first_commitment_date: date | None = None
        if self._closings_by_period:
            first_closing_period = min(self._closings_by_period)
            self.first_commitment_date = self.timeline.observation_date(first_closing_period)

        # The fund types among the funds, in the order they first appear, and the names of the
        # market-value columns every period row will carry: the fund types' totals, then the funds.
        self.fund_types: tuple[str, ...] = tuple(dict.fromkeys(fund.fund_type for fund in funds))
        self.market_value_columns: list[str] = self._market_value_column_names()

        # The liquid-only value at each calendar year's last observation.
        self._year_ends: dict[int, YearEndBalance] = self._year_end_balances()

        # The second of the five tracked values: the same starting value growing at the policy's
        # expected return instead of at the market's. NaN when the policy has no expected return.
        self.expected_liquid: np.ndarray = self._expected_liquid_path(float(levels[0]))

    # ---------------------------------------------------------- preparation
    @staticmethod
    def _check_inputs(portfolio: Any, funds: list[Any], cash_tolerance: Any) -> None:
        """The portfolio is a Portfolio, the funds are uniquely named Funds, the tolerance a usable number."""
        if not isinstance(portfolio, Portfolio):
            raise TypeError("portfolio must be a Portfolio")

        for fund in funds:
            if not isinstance(fund, Fund):
                raise TypeError(f"funds must contain Fund objects, got {type(fund).__name__}")

        names = [fund.name for fund in funds]
        if len(set(names)) != len(names):
            duplicates = sorted({name for name in names if names.count(name) > 1})
            raise ValueError(f"fund names must be unique; duplicated: {duplicates}")

        # numbers.Real: numpy scalars count as numbers; a bool does not, though Python says it is one
        is_a_number = isinstance(cash_tolerance, Real) and not isinstance(cash_tolerance, bool)
        if not is_a_number or not math.isfinite(cash_tolerance) or cash_tolerance < 0:
            raise ValueError("cash_tolerance must be a finite non-negative number")

    def _check_no_fund_closes_before_the_first_observation(self, funds: list[Fund]) -> None:
        first_observation = self.timeline.observation_date(0)

        for fund in funds:
            if fund.closing_date < first_observation:
                raise ValueError(
                    f"Fund {fund.name!r} closes on {fund.closing_date}, before the first observation "
                    f"{first_observation}; there are no pre-existing commitments"
                )

    @staticmethod
    def _period_returns(levels: np.ndarray) -> np.ndarray:
        """Each period's return as a percent change: ``level[t] / level[t-1] − 1``.

        The first period has no preceding observation, so its return is 0.
        """
        percent_changes = levels[1:] / levels[:-1] - 1.0
        return np.concatenate([[0.0], percent_changes])

    def _base_currency_per_usd(self) -> np.ndarray:
        """The price of 1 US dollar in base currency at each observation: all ones for a dollar portfolio."""
        if not self.portfolio.requires_fx_conversion:
            return np.ones(self.timeline.n_observations)

        # A rate is a state, not an event: each observation uses the last rate on or before it.
        return self.timeline.last_value_on_or_before(self.portfolio.usd_rate, name="usd_rate")

    def _market_value_column_names(self) -> list[str]:
        """The names of the market-value columns: two per fund type, then two per fund."""
        names: list[str] = []

        for fund_type in self.fund_types:
            names.extend(fund_type_market_value_columns(fund_type))

        for fund in self.funds:
            names.extend(fund_market_value_columns(fund.name))

        # A fund called "BUYOUT_total", say, would collide with the BUYOUT total's columns.
        all_names = PERIOD_COLUMNS + names
        repeated = sorted({name for name in all_names if all_names.count(name) > 1})
        if repeated:
            raise ValueError(
                f"fund and fund-type names give the periods table the same column twice: {repeated}; "
                f"rename the fund"
            )

        return names

    def _year_end_balances(self) -> dict[int, YearEndBalance]:
        """The liquid-only value in USD at each calendar year's last observation."""
        # Walking forward, a later observation of the same year overwrites an earlier one.
        last_period_of_year: dict[int, int] = {}
        for t in range(self.timeline.n_observations):
            year = self.timeline.observation_date(t).year
            last_period_of_year[year] = t

        year_ends: dict[int, YearEndBalance] = {}
        for year, t in last_period_of_year.items():
            year_ends[year] = YearEndBalance(
                date=self.timeline.observation_date(t),
                liquid_only_usd=float(self.liquid_only_usd[t]),
            )

        return year_ends

    def _expected_liquid_path(self, initial_value: float) -> np.ndarray:
        """The liquid portfolio had it grown at the expected return from the first observation."""
        n_observations = self.timeline.n_observations

        # Only a policy built on an expected return has one; any other leaves this column empty.
        expected_return = getattr(self.policy, "expected_return", None)
        if expected_return is None:
            return np.full(n_observations, np.nan)

        inception = self.timeline.observation_date(0)

        years_since_inception = np.array([
            years_between(inception, self.timeline.observation_date(t))
            for t in range(n_observations)
        ])

        return initial_value * (1.0 + expected_return) ** years_since_inception

    # ------------------------------------------------------------------ run
    def run(self) -> SimulationResult:
        """Walk the observations once, from a fresh state, recording what happened at each.

        The six numbered steps below are the whole engine. Their order is the cash
        convention and it is deliberate: the liquid portfolio earns its return before any
        private cash moves, commitments are sized before the cohort's own distributions
        arrive, and the calls are paid last of all.
        """
        starting_balance = float(self.portfolio.liquid_levels.iloc[0])
        liquid = LiquidAccount(starting_balance, self.returns)
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
            liquid_account_usd = liquid.balance / rate
            balances = self._sizing_balances(t, day, liquid_account_usd, private_nav_usd_open)
            dollars_by_fund = self._size_commitments(cohort, balances)

            for fund in cohort:
                commitment_usd = dollars_by_fund[fund.name]
                history = self.aligned_histories[fund.name]

                book.add(Commitment(fund, history, commitment_usd))
                commitment_rows.append(self._commitment_row(t, day, fund, balances, commitment_usd, rate))
                draw_rows.extend(self._draw_rows(day, fund, balances, rate))

            # 5 ── The new cohort's own distributions are banked — they could not be in step 3,
            #      since those funds did not exist yet — and then every call is paid.
            new_commitments = book.commitments_closing_in(t)
            distributions_new_usd = math.fsum(c.distributions_in_period(t) for c in new_commitments)
            distributions_new = distributions_new_usd * rate
            liquid.deposit(distributions_new)

            calls = book.calls_in_period(t) * rate
            unpaid = liquid.withdraw(calls)  # what the account could not cover, 0 when it could

            # 6 ── Value the book and record the observation.
            private_close = book.nav_at(t) * rate
            distributions = distributions_existing + distributions_new
            committed_usd = math.fsum(dollars_by_fund.values())

            # The exchange rate moved the base-currency value of the dollars already held.
            if t > 0:
                rate_change = rate - float(self.fx[t - 1])
                fx_translation = private_nav_usd_open * rate_change
            else:
                fx_translation = 0.0

            period_row = {
                "date": pd.Timestamp(day),

                # the five running values: the liquid portfolio three ways, the private book, the total
                "liquid_only": float(self.liquid_only[t]),           # 1 · liquid alone, no private flows
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
                "fx_translation": fx_translation,

                # what was committed today, and the dollars it was sized on
                "liquid_only_usd": balances.liquid_only_usd,
                "commitments": committed_usd * rate,
                "commitments_usd": committed_usd,
            }

            # The market value of each fund type and of each fund, in dollars and in base currency.
            market_values = self._market_values(t, book, rate)
            period_row.update(market_values)

            period_rows.append(period_row)
            fund_rows.extend(self._fund_rows(t, day, book, rate))
            private_open = private_close

            # ... and stop here if the account could not meet the calls.
            if unpaid > self.cash_tolerance and shortfall is None:
                shortfall = self._shortfall(t, day, book, calls, unpaid, rate)

                if self.stop_on_shortfall:
                    break

        return SimulationResult(
            base_currency=self.portfolio.base_currency,
            periods=_build_table(period_rows, PERIOD_COLUMNS + self.market_value_columns, ["date"]),
            funds=_build_table(fund_rows, FUND_COLUMNS, ["date", "fund"]),
            commitments=_build_table(commitment_rows, COMMITMENT_COLUMNS, ["date", "fund"]),
            draws=_build_table(draw_rows, DRAW_COLUMNS, ["date", "fund", "year"]),
            shortfall=shortfall,
            funds_beyond_horizon=self.funds_beyond_horizon,
        )

    # ------------------------------------------------- what the loop hands the policy
    def _sizing_balances(self, t: int, day: date, liquid_account_usd: float,
                         private_nav_usd: float) -> SizingBalances:
        """What the policy is shown: US dollars throughout, and the liquid-only value to size on."""
        # The year ends fall into two groups, and they are kept apart on purpose.
        #
        # Completed years are what the investor could have known today. Carry-forward reads
        # only these.
        #
        # The current year's end and every later one's could NOT have been known today. They
        # are handed over because a draw plan may name a year after the fund's closing, and
        # such a year is priced on its own year-end value — a deliberate look ahead.
        completed_year_ends = {}
        future_year_ends = {}

        for year, year_end in self._year_ends.items():
            if year < day.year:
                completed_year_ends[year] = year_end
            else:
                future_year_ends[year] = year_end

        return SizingBalances(
            t,
            day,
            liquid_only_usd=float(self.liquid_only_usd[t]),
            liquid_account_usd=liquid_account_usd,
            private_nav_usd=private_nav_usd,
            year_ends=completed_year_ends,
            first_commitment_date=self.first_commitment_date,
            future_year_ends=future_year_ends,
        )

    def _size_commitments(self, cohort: Sequence[Fund], balances: SizingBalances) -> dict[str, float]:
        """The policy's US-dollar commitment for every fund in the cohort.

        Checked here, before any of it reaches the book.
        """
        if not cohort:
            return {}

        try:
            sized: Mapping[str, float] = self.policy.size_commitments(cohort, balances)
        except KeyError as exc:
            raise ValueError(f"policy has no sizing for fund {exc}") from None

        # The policy may only size the funds it was asked about.
        names_in_cohort = {fund.name for fund in cohort}
        unknown_names = set(sized) - names_in_cohort
        if unknown_names:
            raise ValueError(f"policy sized funds that are not closing now: {sorted(unknown_names)}")

        # A fund the policy left out gets nothing; every amount must be a usable number.
        commitments_usd = {}
        for fund in cohort:
            dollars = float(sized.get(fund.name, 0.0))

            if not math.isfinite(dollars) or dollars < 0:
                raise ValueError(f"policy returned an invalid commitment {dollars!r} for {fund.name!r}")

            commitments_usd[fund.name] = dollars

        return commitments_usd

    # ------------------------------------------------------- rows of the result tables
    def _market_values(self, t: int, book: CommitmentBook, rate: float) -> dict[str, float]:
        """The market value (NAV) of every fund type and of every fund at observation ``t``.

        Each value twice: in US dollars, as the fund reports it, and in base currency at this
        observation's exchange rate. A fund that has not been committed yet is worth zero.
        """
        # Every fund's NAV in dollars: zero until it is in the book.
        nav_usd_by_fund = {fund.name: 0.0 for fund in self.funds}

        for commitment in book.commitments:
            nav_usd_by_fund[commitment.fund.name] = commitment.nav_at(t)

        values: dict[str, float] = {}

        # Each fund type: the sum of its funds.
        for fund_type in self.fund_types:
            navs_of_this_type = [
                nav_usd_by_fund[fund.name]
                for fund in self.funds
                if fund.fund_type == fund_type
            ]
            total_nav_usd = math.fsum(navs_of_this_type)

            usd_column, base_column = fund_type_market_value_columns(fund_type)
            values[usd_column] = total_nav_usd
            values[base_column] = total_nav_usd * rate

        # Each fund on its own.
        for fund in self.funds:
            nav_usd = nav_usd_by_fund[fund.name]

            usd_column, base_column = fund_market_value_columns(fund.name)
            values[usd_column] = nav_usd
            values[base_column] = nav_usd * rate

        return values

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

                # in dollars, as the fund reports them
                "calls_usd": calls_usd,
                "distributions_usd": distributions_usd,
                "nav_usd": nav_usd,

                # and in base currency, at this observation's rate
                "calls_base": calls_usd * rate,
                "distributions_base": distributions_usd * rate,
                "nav_base": nav_usd * rate,
            })

        return rows

    def _commitment_row(self, t: int, day: date, fund: Fund, balances: SizingBalances,
                        commitment_usd: float, rate: float) -> dict[str, Any]:
        """One closing: what was decided in dollars, then the same in base currency at that day's rate."""
        # The commitment as a share of what it was sized on.
        if balances.liquid_only_usd:
            share_of_sizing_base = commitment_usd / balances.liquid_only_usd
        else:
            share_of_sizing_base = float("nan")

        row: dict[str, Any] = {
            "date": pd.Timestamp(day),
            "fund": fund.name,
            "fund_type": fund.fund_type,
            "closing_date": pd.Timestamp(fund.closing_date),
            "policy_year": fund.closing_year,

            # the decision, in dollars
            "sizing_base_usd": balances.liquid_only_usd,
            "rate": share_of_sizing_base,
            "commitment_usd": commitment_usd,

            # and the same, translated at today's rate for reporting
            "usd_rate": rate,
            "sizing_base": float(self.liquid_only[t]),
            "commitment_base": commitment_usd * rate,

            # filled in below by policies that can say how they got there
            "own_year_rate": float("nan"),
            "expected_value": float("nan"),
            "weight": float("nan"),
            "own_year_usd": float("nan"),
            "other_years_usd": float("nan"),
            "drawn_years": "",
        }

        # Optional: a policy may say how it got there. AnnualRatePolicy does.
        explain_commitment = getattr(self.policy, "explain_commitment", None)
        if not callable(explain_commitment):
            return row

        explanation = explain_commitment(fund.name, balances)

        for column in EXPLAINED_NUMBER_COLUMNS:
            if column in explanation:
                row[column] = float(explanation[column])

        row["drawn_years"] = str(explanation.get("drawn_years", ""))
        return row

    def _draw_rows(self, day: date, fund: Fund, balances: SizingBalances,
                   rate: float) -> list[dict[str, Any]]:
        """One row per schedule year the fund draws.

        Empty for a policy that cannot break a commitment down.
        """
        # Optional, like explain_commitment.
        drawn_years = getattr(self.policy, "drawn_years", None)
        if not callable(drawn_years):
            return []

        weight = self.policy.entitlements[fund.name].weight

        rows = []
        for drawn in drawn_years(fund.name, balances):

            # The fund's weight is applied here, so these dollars add up to its commitment.
            commitment_usd = weight * drawn.commitment_usd

            rows.append({
                "date": pd.Timestamp(day),
                "fund": fund.name,
                "year": drawn.year,
                "fund_type": fund.fund_type,
                "policy_year": fund.closing_year,

                # the schedule's rate for the year, and how many times it is drawn
                "multiplier": drawn.multiplier,
                "rate": drawn.rate,

                # the date the year is priced on, whether that date lies after the closing,
                # and the pacing model's value and the liquid-only value on it
                "sizing_date": pd.Timestamp(drawn.sizing_date),
                "looks_ahead": drawn.looks_ahead,
                "expected_value": drawn.expected_value,
                "liquid_only_usd": drawn.liquid_only_usd,

                # the year's own dollar commitment, as computed and then as rounded
                "year_budget_unrounded_usd": drawn.year_budget_unrounded_usd,
                "year_budget_usd": drawn.year_budget_usd,

                # what the fund collects from this year, in dollars and in base currency
                "commitment_usd": commitment_usd,
                "usd_rate": rate,
                "commitment_base": commitment_usd * rate,
            })

        return rows

    def _shortfall(self, t: int, day: date, book: CommitmentBook,
                   calls: float, unpaid: float, rate: float) -> Shortfall:
        """The first observation whose calls the account could not meet, with the calls that caused it."""
        calls_in_base_currency = {}
        for commitment in book.commitments:
            calls_in_base_currency[commitment.fund.name] = commitment.calls_in_period(t) * rate

        calls_by_fund = pd.Series(calls_in_base_currency, dtype=float, name="calls_base")
        calls_by_fund.index.name = "fund"

        return Shortfall(
            t,
            day,
            cash_available=calls - unpaid,
            calls_due=calls,
            calls_by_fund=calls_by_fund,
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
        n_observations = self.timeline.n_observations
        rows: list[dict[str, Any]] = []

        for fund in self.funds:
            events = fund.events_by_day()
            event_days = [day for day, *_ in events]
            periods = self.timeline.first_observations_on_or_after(event_days)

            for event, period in zip(events, periods):
                day, call, distribution, mark = event

                # An event after the last observation lands nowhere.
                if period < n_observations:
                    observation_date = self.timeline.dates[period]
                else:
                    observation_date = pd.NaT

                if mark is None:
                    unit_nav_mark = float("nan")
                else:
                    unit_nav_mark = mark

                rows.append({
                    "fund": fund.name,
                    "event_date": day,
                    "observation_date": observation_date,
                    "period": int(period),
                    "unit_call": call,
                    "unit_distribution": distribution,
                    "unit_nav_mark": unit_nav_mark,
                })

        return _build_table(rows, EVENT_COLUMNS, ["fund", "event_date"])
