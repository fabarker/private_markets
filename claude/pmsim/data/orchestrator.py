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


def infer_inception_date(dates: Any) -> pd.Timestamp:
    """One period before the first date, on the frequency inferred from the dates, rolled back to a business day.

    Month-end returns starting 30 April give 31 March; quarter-ends give the previous
    quarter end; daily returns give the previous day. A period end that falls on a weekend
    rolls back to the Friday before it. The frequency comes from ``pd.infer_freq``, which
    needs at least three regularly spaced dates.
    """
    dates = pd.DatetimeIndex(dates)
    if len(dates) < 3:
        raise ValueError("at least three dated returns are needed to infer their frequency; supply inception_date")
    frequency = pd.infer_freq(dates)
    if frequency is None:
        raise ValueError("cannot infer the frequency of the returns from their dates; supply inception_date")
    previous = dates[0] - pd.tseries.frequencies.to_offset(frequency)
    return pd.offsets.BDay().rollback(previous)


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
    if inception_date is None:
        inception = infer_inception_date(dates)
    else:
        inception = pd.Timestamp(as_date(inception_date))
        if inception >= dates[0]:
            raise ValueError(f"inception_date {inception.date()} must be before the first return on {dates[0].date()}")
    levels = float(initial_value) * np.cumprod(np.concatenate([[1.0], return_factors]))
    return pd.Series(levels, index=pd.DatetimeIndex([inception]).append(dates), name=returns.name)


def _rate_at_inception(usd_rate: pd.Series, inception: pd.Timestamp) -> pd.Series:
    """When the FX series starts after the inception date, its first rate is taken to apply there.

    Nothing is normally converted on the inception date — no commitment exists yet — so the
    rate only labels that row; without it the timeline would reject a series that starts
    on the first return date, as the workbook's FX sheet does.
    """
    if usd_rate.empty or usd_rate.index.min() <= inception:
        return usd_rate
    first = usd_rate.sort_index().iloc[0]
    return pd.concat([pd.Series([first], index=pd.DatetimeIndex([inception])), usd_rate])


def build_funds(specs: pd.DataFrame, market: pd.DataFrame, *, calls_are_negative: bool = True) -> list[Fund]:
    """One ``Fund`` per fund_spec row, with its unit history from fund_market_data.

    A fund in the spec with no market rows gets an empty history (a future closing).
    Market rows for a fund missing from the spec are an error, not silently dropped.
    """
    unknown = sorted(set(market["fund_name"]) - set(specs["fund_name"]))
    if unknown:
        raise ValueError(f"fund_market_data has rows for funds missing from fund_spec: {unknown}")
    grouped = {name: rows for name, rows in market.groupby("fund_name", sort=False)}
    funds = []
    for spec in specs.itertuples(index=False):
        calls: list[tuple[Any, float]] = []
        distributions: list[tuple[Any, float]] = []
        marks: list[tuple[Any, float]] = []
        for row in grouped.get(spec.fund_name, market.iloc[0:0]).itertuples(index=False):
            if row.kind == "nav":
                marks.append((row.date, row.unit))
            elif row.kind == "call":
                calls.append((row.date, abs(row.unit)))
            elif row.kind == "distribution":
                distributions.append((row.date, abs(row.unit)))
            else:  # flow: the sign says which
                called = -row.unit if calls_are_negative else row.unit
                if called > 0:
                    calls.append((row.date, called))
                elif called < 0:
                    distributions.append((row.date, -called))
        funds.append(Fund(spec.fund_name, spec.fund_type, spec.closing_date,
                          unit_calls=calls, unit_distributions=distributions, unit_nav=marks))
    return funds


def select_market_series(market: pd.DataFrame, name: str, *, label: str) -> pd.Series:
    """A market_data column by name (case-insensitively), blanks dropped."""
    matches = [column for column in market.columns if canonical_name(column) == canonical_name(name)]
    if not matches:
        raise ValueError(f"market_data has no series {name!r} for {label}; series are {list(market.columns)}")
    return market[matches[0]].dropna()


def build_portfolio(market: pd.DataFrame, spec: SimulationSpec, rates: Any) -> Portfolio:
    """The ``Portfolio`` for a spec: liquid index and, unless the base currency is USD, the USD rate."""
    liquid = select_market_series(market, spec.liquid_series, label="liquid_series")
    if spec.liquid_kind == "returns":
        liquid = returns_to_levels(liquid, spec.initial_value, inception_date=spec.inception_date)
    usd_rate = None
    if spec.base_currency.strip().upper() == PRIVATE_CURRENCY:
        if spec.fx_series:
            raise ValueError("fx_series is set but base_currency is USD; drop one of them")
    else:
        if not spec.fx_series:
            raise ValueError(f"fx_series is required: base_currency {spec.base_currency!r} is not {PRIVATE_CURRENCY}")
        usd_rate = select_market_series(market, spec.fx_series, label="fx_series")
        if spec.fx_quote == "usd_per_base":
            usd_rate = 1.0 / usd_rate
        if spec.liquid_kind == "returns":
            usd_rate = _rate_at_inception(usd_rate, liquid.index[0])
    return Portfolio(spec.base_currency, liquid, rates, usd_rate=usd_rate)


class Orchestrator:
    """A repository and a spec, assembled into the engine's inputs on first use."""

    def __init__(self, repository: DataRepository, spec: SimulationSpec) -> None:
        if not isinstance(spec, SimulationSpec):
            raise TypeError("spec must be a SimulationSpec")
        self.repository = repository
        self.spec = spec

    @cached_property
    def funds(self) -> list[Fund]:
        return build_funds(self.repository.fund_specs(), self.repository.fund_market_data(),
                           calls_are_negative=self.spec.calls_are_negative)

    @cached_property
    def commitment_rates(self) -> Any:
        rates = self.spec.commitment_rates
        if rates is None:
            rates = self.repository.commitment_rates()
        if rates is None:
            raise ValueError("commitment_rates are needed: pass them in the SimulationSpec or supply a commitment_rates sheet")
        return rates

    @cached_property
    def expected_return(self) -> float | None:
        """The yearly return the pacing schedule was built on: the spec's, else the repository's row for the liquid series.

        None only when the repository has no expected_returns table at all; a table that
        exists but does not list this portfolio is an error, not a silent fallback.
        """
        if self.spec.expected_return is not None:
            return float(self.spec.expected_return)
        table = self.repository.expected_returns()
        if table is None:
            return None
        matches = [name for name in table.index if canonical_name(name) == canonical_name(self.spec.liquid_series)]
        if not matches:
            raise ValueError(f"expected_returns has no row for portfolio {self.spec.liquid_series!r}; "
                             f"portfolios are {list(table.index)}")
        return float(table[matches[0]])

    @cached_property
    def draw_plans(self) -> dict[str, dict[int, float]]:
        """Which schedule years each fund draws, on calendar years: the spec's when given, else the repository's."""
        if self.spec.draws is not None:
            return {name: dict(plan) for name, plan in self.spec.draws.items()}
        from_repository = getattr(self.repository, "draw_plans", None)  # optional: a repository need not have any
        return {} if from_repository is None else {name: dict(plan) for name, plan in (from_repository() or {}).items()}

    @cached_property
    def portfolio(self) -> Portfolio:
        return build_portfolio(self.repository.market_data(), self.spec, self.commitment_rates)

    @cached_property
    def policy(self) -> AnnualRatePolicy:
        # The rate table must cover every year with a return; an inception date that falls in
        # the year before the first return carries no target and needs no row.
        levels = self.portfolio.liquid_levels
        first_year = levels.index[1 if self.spec.liquid_kind == "returns" else 0].year
        return AnnualRatePolicy(self.portfolio.commitment_rates, self.funds, self.spec.weights,
                                carry_forward=self.spec.carry_forward,
                                years=range(first_year, self.portfolio.last_date.year + 1),
                                expected_return=self.expected_return, draws=self.draw_plans)

    @cached_property
    def simulator(self) -> Simulator:
        return Simulator(self.portfolio, self.funds, self.policy,
                         stop_on_shortfall=self.spec.stop_on_shortfall, cash_tolerance=self.spec.cash_tolerance)

    def run(self) -> SimulationResult:
        return self.simulator.run()

    def map_events_to_observations(self) -> pd.DataFrame:
        return self.simulator.map_events_to_observations()

    def fund_summary(self) -> pd.DataFrame:
        """One row per fund: what was loaded for it, and whether it closes inside the horizon."""
        beyond = set(self.simulator.funds_beyond_horizon)
        rows = []
        for fund in self.funds:
            days = fund.unit_calls.index.union(fund.unit_distributions.index).union(fund.unit_nav.index)
            rows.append({
                "fund_name": fund.name, "fund_type": fund.fund_type, "closing_date": pd.Timestamp(fund.closing_date),
                "calls": len(fund.unit_calls), "distributions": len(fund.unit_distributions), "marks": len(fund.unit_nav),
                "unit_called": float(fund.unit_calls.sum()), "unit_distributed": float(fund.unit_distributions.sum()),
                "latest_unit_nav": float(fund.unit_nav.iloc[-1]) if len(fund.unit_nav) else float("nan"),
                "first_event": days.min() if len(days) else pd.NaT, "last_event": days.max() if len(days) else pd.NaT,
                "beyond_horizon": fund.name in beyond,
            })
        columns = ["fund_name", "fund_type", "closing_date", "calls", "distributions", "marks", "unit_called",
                   "unit_distributed", "latest_unit_nav", "first_event", "last_event", "beyond_horizon"]
        frame = pd.DataFrame(rows, columns=columns)
        for column in ("closing_date", "first_event", "last_event"):
            frame[column] = pd.to_datetime(frame[column])
        return frame.set_index("fund_name")


def load_profile_workbook(path: Any, currency: str, risk: str, initial_value: float, *,
                          layout: SheetLayout = SheetLayout(), **overrides: Any) -> Orchestrator:
    """An ``Orchestrator`` for one profile of the portfolio workbook; ``overrides`` are ``SimulationSpec`` settings."""
    repository = WorkbookRepository(path, currency, risk, layout)
    return Orchestrator(repository, repository.simulation_spec(initial_value, **overrides))


def run_profile_workbook(path: Any, currency: str, risk: str, initial_value: float, *,
                         layout: SheetLayout = SheetLayout(), **overrides: Any) -> SimulationResult:
    return load_profile_workbook(path, currency, risk, initial_value, layout=layout, **overrides).run()
