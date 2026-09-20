"""From tables to a run.

``Orchestrator`` joins a repository and a ``SimulationSpec`` into ``Fund`` and ``Portfolio``
objects, builds the policy and the simulator, and runs them. ``load_profile_workbook`` is
the front door for the Excel portfolio workbook: one call from a path and a profile to
an orchestrator ready to run.
"""
from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd

from ..dates import as_date
from ..inputs import PRIVATE_CURRENCY, Fund, Portfolio
from ..policy import AnnualRatePolicy
from ..simulator import SimulationResult, Simulator
from .repository import DataRepository, SheetLayout, WorkbookRepository
from .spec import SimulationSpec
from .tables import canonical_name

FUND_SUMMARY_COLUMNS = [
    "fund_name", "fund_type", "closing_date", "calls", "distributions", "marks", "unit_called",
    "unit_distributed", "latest_unit_nav", "first_event", "last_event", "beyond_horizon",
]

DatedValues = list[tuple[Any, float]]  # (date, value) pairs, as Fund accepts them


# ------------------------------------------------------------ returns → levels
def infer_inception_date(dates: Any) -> pd.Timestamp:
    """One period before the first date, rolled back to a business day.

    The period is the frequency inferred from the dates themselves. Month-end returns
    starting 30 April give 31 March; quarter-ends give the previous quarter end; daily
    returns give the previous day. A period end that falls on a weekend rolls back to the
    Friday before it. The frequency comes from ``pd.infer_freq``, which needs at least three
    regularly spaced dates.
    """
    dates = pd.DatetimeIndex(dates)

    if len(dates) < 3:
        raise ValueError(
            "at least three dated returns are needed to infer their frequency; supply inception_date"
        )

    frequency = pd.infer_freq(dates)
    if frequency is None:
        raise ValueError("cannot infer the frequency of the returns from their dates; supply inception_date")

    # Step back one period from the first date, then onto a business day.
    one_period = pd.tseries.frequencies.to_offset(frequency)
    one_period_earlier = dates[0] - one_period

    return pd.offsets.BDay().rollback(one_period_earlier)


def returns_to_levels(returns: pd.Series, initial_value: float, *, inception_date: Any = None) -> pd.Series:
    """Total-return levels from simple per-period returns, starting one period before the first.

    The inception date — ``inception_date`` if given, otherwise ``infer_inception_date`` of
    the return dates — carries ``initial_value``; every return is then applied:
    ``level[t] = level[t-1] × (1 + returns[t])``.
    """
    if returns.empty:
        raise ValueError("returns series is empty")

    return_factors = 1.0 + returns.to_numpy(dtype=float)
    if not np.isfinite(return_factors).all() or (return_factors <= 0).any():
        raise ValueError("returns must be finite and greater than -100%")

    dates = pd.DatetimeIndex(returns.index)

    # The date the starting balance sits on: given, or inferred from the returns' frequency.
    if inception_date is None:
        inception = infer_inception_date(dates)
    else:
        inception = pd.Timestamp(as_date(inception_date))

        if inception >= dates[0]:
            raise ValueError(
                f"inception_date {inception.date()} must be before the first return on {dates[0].date()}"
            )

    # Compound: the starting balance, then every return applied in turn.
    growth_since_inception = np.cumprod(np.concatenate([[1.0], return_factors]))
    levels = float(initial_value) * growth_since_inception

    dates_with_inception = pd.DatetimeIndex([inception]).append(dates)
    return pd.Series(levels, index=dates_with_inception, name=returns.name)


def _rate_at_inception(usd_rate: pd.Series, inception: pd.Timestamp) -> pd.Series:
    """When the FX series starts after the inception date, its first rate is taken to apply there.

    Nothing is normally converted on the inception date — no commitment exists yet — so the
    rate only labels that row; without it the timeline would reject a series that starts
    on the first return date, as the workbook's FX sheet does.
    """
    already_covers_inception = usd_rate.empty or usd_rate.index.min() <= inception
    if already_covers_inception:
        return usd_rate

    first_rate = usd_rate.sort_index().iloc[0]
    rate_on_inception = pd.Series([first_rate], index=pd.DatetimeIndex([inception]))

    return pd.concat([rate_on_inception, usd_rate])


# ------------------------------------------------------------- tables → inputs
def _unit_history_from_market_rows(rows: pd.DataFrame, *,
                                   calls_are_negative: bool) -> tuple[DatedValues, DatedValues, DatedValues]:
    """One fund's market rows sorted into its calls, its distributions and its NAV marks."""
    calls: DatedValues = []
    distributions: DatedValues = []
    marks: DatedValues = []

    for row in rows.itertuples(index=False):
        if row.kind == "nav":
            marks.append((row.date, row.unit))

        elif row.kind == "call":
            calls.append((row.date, abs(row.unit)))

        elif row.kind == "distribution":
            distributions.append((row.date, abs(row.unit)))

        else:
            # A flow: its sign says which way the money went.
            if calls_are_negative:
                amount_called = -row.unit
            else:
                amount_called = row.unit

            if amount_called > 0:
                calls.append((row.date, amount_called))
            elif amount_called < 0:
                distributions.append((row.date, -amount_called))

    return calls, distributions, marks


def build_funds(specs: pd.DataFrame, market: pd.DataFrame, *, calls_are_negative: bool = True) -> list[Fund]:
    """One ``Fund`` per fund_spec row, with its unit history from fund_market_data.

    A fund in the spec with no market rows gets an empty history (a future closing).
    Market rows for a fund missing from the spec are an error, not silently dropped.
    """
    funds_with_rows_but_no_spec = sorted(set(market["fund_name"]) - set(specs["fund_name"]))
    if funds_with_rows_but_no_spec:
        raise ValueError(
            f"fund_market_data has rows for funds missing from fund_spec: {funds_with_rows_but_no_spec}"
        )

    # Each fund's market rows, in the order the table has them.
    rows_by_fund = {}
    for name, rows in market.groupby("fund_name", sort=False):
        rows_by_fund[name] = rows

    no_rows = market.iloc[0:0]

    funds = []
    for spec in specs.itertuples(index=False):
        rows = rows_by_fund.get(spec.fund_name, no_rows)

        calls, distributions, marks = _unit_history_from_market_rows(
            rows,
            calls_are_negative=calls_are_negative,
        )

        funds.append(Fund(
            spec.fund_name,
            spec.fund_type,
            spec.closing_date,
            unit_calls=calls,
            unit_distributions=distributions,
            unit_nav=marks,
        ))

    return funds


def select_market_series(market: pd.DataFrame, name: str, *, label: str) -> pd.Series:
    """A market_data column by name (case-insensitively), blanks dropped."""
    wanted = canonical_name(name)
    matches = [column for column in market.columns if canonical_name(column) == wanted]

    if not matches:
        raise ValueError(f"market_data has no series {name!r} for {label}; series are {list(market.columns)}")

    return market[matches[0]].dropna()


def build_portfolio(market: pd.DataFrame, spec: SimulationSpec, rates: Any) -> Portfolio:
    """The ``Portfolio`` for a spec: liquid index and, unless the base currency is USD, the USD rate."""
    # The liquid index: levels as they are, or compounded from returns.
    liquid = select_market_series(market, spec.liquid_series, label="liquid_series")

    if spec.liquid_kind == "returns":
        liquid = returns_to_levels(liquid, spec.initial_value, inception_date=spec.inception_date)

    # A dollar portfolio has no exchange rate, and must not name one.
    is_dollar_portfolio = spec.base_currency.strip().upper() == PRIVATE_CURRENCY

    if is_dollar_portfolio:
        if spec.fx_series:
            raise ValueError("fx_series is set but base_currency is USD; drop one of them")

        return Portfolio(spec.base_currency, liquid, rates, usd_rate=None)

    # Any other needs the price of a dollar in base currency.
    if not spec.fx_series:
        raise ValueError(
            f"fx_series is required: base_currency {spec.base_currency!r} is not {PRIVATE_CURRENCY}"
        )

    usd_rate = select_market_series(market, spec.fx_series, label="fx_series")

    # EURUSD is dollars per euro; the engine wants euros per dollar.
    if spec.fx_quote == "usd_per_base":
        usd_rate = 1.0 / usd_rate

    if spec.liquid_kind == "returns":
        inception = liquid.index[0]
        usd_rate = _rate_at_inception(usd_rate, inception)

    return Portfolio(spec.base_currency, liquid, rates, usd_rate=usd_rate)


# ---------------------------------------------------------------- the orchestrator
class Orchestrator:
    """A repository and a spec, assembled into the engine's inputs on first use."""

    def __init__(self, repository: DataRepository, spec: SimulationSpec) -> None:
        if not isinstance(spec, SimulationSpec):
            raise TypeError("spec must be a SimulationSpec")

        self.repository = repository
        self.spec = spec

    @cached_property
    def funds(self) -> list[Fund]:
        return build_funds(
            self.repository.fund_specs(),
            self.repository.fund_market_data(),
            calls_are_negative=self.spec.calls_are_negative,
        )

    @cached_property
    def commitment_rates(self) -> Any:
        """The rate table: the spec's when it gives one, else the repository's."""
        rates = self.spec.commitment_rates

        if rates is None:
            rates = self.repository.commitment_rates()

        if rates is None:
            raise ValueError(
                "commitment_rates are needed: "
                "pass them in the SimulationSpec or supply a commitment_rates sheet"
            )

        return rates

    @cached_property
    def expected_return(self) -> float | None:
        """The yearly return the pacing schedule was built on.

        The spec's when it gives one, else the repository's row for the liquid series. None
        only when the repository has no expected_returns table at all; a table that exists
        but does not list this portfolio is an error, not a silent fallback.
        """
        if self.spec.expected_return is not None:
            return float(self.spec.expected_return)

        table = self.repository.expected_returns()
        if table is None:
            return None

        wanted = canonical_name(self.spec.liquid_series)
        matches = [name for name in table.index if canonical_name(name) == wanted]

        if not matches:
            raise ValueError(
                f"expected_returns has no row for portfolio {self.spec.liquid_series!r}; "
                f"portfolios are {list(table.index)}"
            )

        return float(table[matches[0]])

    @cached_property
    def draw_plans(self) -> dict[str, dict[int, float]]:
        """Which schedule years each fund draws, on calendar years.

        The spec's when it gives any, else the repository's.
        """
        if self.spec.draws is not None:
            plans = self.spec.draws
        else:
            # Optional: a repository need not have any.
            draw_plans_of_repository = getattr(self.repository, "draw_plans", None)

            if draw_plans_of_repository is None:
                plans = {}
            else:
                plans = draw_plans_of_repository() or {}

        return {name: dict(plan) for name, plan in plans.items()}

    @cached_property
    def portfolio(self) -> Portfolio:
        return build_portfolio(self.repository.market_data(), self.spec, self.commitment_rates)

    @cached_property
    def policy(self) -> AnnualRatePolicy:
        # The rate table must cover every year with a return. An inception date that falls in
        # the year before the first return carries no target, and needs no row.
        levels = self.portfolio.liquid_levels

        if self.spec.liquid_kind == "returns":
            first_date_with_a_return = levels.index[1]
        else:
            first_date_with_a_return = levels.index[0]

        first_year = first_date_with_a_return.year
        last_year = self.portfolio.last_date.year

        return AnnualRatePolicy(
            self.portfolio.commitment_rates,
            self.funds,
            self.spec.weights,
            carry_forward=self.spec.carry_forward,
            years=range(first_year, last_year + 1),
            expected_return=self.expected_return,
            draws=self.draw_plans,
            rounding_unit_usd=self.spec.commitment_rounding_unit_usd,
        )

    @cached_property
    def simulator(self) -> Simulator:
        return Simulator(
            self.portfolio,
            self.funds,
            self.policy,
            stop_on_shortfall=self.spec.stop_on_shortfall,
            cash_tolerance=self.spec.cash_tolerance,
        )

    def run(self) -> SimulationResult:
        return self.simulator.run()

    def map_events_to_observations(self) -> pd.DataFrame:
        return self.simulator.map_events_to_observations()

    def fund_summary(self) -> pd.DataFrame:
        """One row per fund: what was loaded for it, and whether it closes inside the horizon."""
        names_beyond_horizon = set(self.simulator.funds_beyond_horizon)

        rows = []
        for fund in self.funds:
            # Every day the fund has a call, a distribution or a mark on.
            event_days = (
                fund.unit_calls.index
                .union(fund.unit_distributions.index)
                .union(fund.unit_nav.index)
            )

            if len(event_days):
                first_event = event_days.min()
                last_event = event_days.max()
            else:
                first_event = pd.NaT
                last_event = pd.NaT

            if len(fund.unit_nav):
                latest_unit_nav = float(fund.unit_nav.iloc[-1])
            else:
                latest_unit_nav = float("nan")

            rows.append({
                "fund_name": fund.name,
                "fund_type": fund.fund_type,
                "closing_date": pd.Timestamp(fund.closing_date),

                # how much history was loaded
                "calls": len(fund.unit_calls),
                "distributions": len(fund.unit_distributions),
                "marks": len(fund.unit_nav),

                # per 1 committed
                "unit_called": float(fund.unit_calls.sum()),
                "unit_distributed": float(fund.unit_distributions.sum()),
                "latest_unit_nav": latest_unit_nav,

                "first_event": first_event,
                "last_event": last_event,
                "beyond_horizon": fund.name in names_beyond_horizon,
            })

        frame = pd.DataFrame(rows, columns=FUND_SUMMARY_COLUMNS)

        for date_column in ("closing_date", "first_event", "last_event"):
            frame[date_column] = pd.to_datetime(frame[date_column])

        return frame.set_index("fund_name")


# ------------------------------------------------------------------ front doors
def load_profile_workbook(path: Any, currency: str, risk: str, initial_value: float, *,
                          layout: SheetLayout = SheetLayout(), **overrides: Any) -> Orchestrator:
    """An ``Orchestrator`` for one profile of the portfolio workbook.

    ``overrides`` are ``SimulationSpec`` settings.
    """
    repository = WorkbookRepository(path, currency, risk, layout)
    spec = repository.simulation_spec(initial_value, **overrides)

    return Orchestrator(repository, spec)


def run_profile_workbook(path: Any, currency: str, risk: str, initial_value: float, *,
                         layout: SheetLayout = SheetLayout(), **overrides: Any) -> SimulationResult:
    """Load one profile of the portfolio workbook and run it."""
    orchestrator = load_profile_workbook(path, currency, risk, initial_value, layout=layout, **overrides)
    return orchestrator.run()
