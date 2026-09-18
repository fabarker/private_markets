"""From tables to a run.

``SimulationSpec`` holds what the data does not say: the base currency, which market_data
series are the liquid index and the exchange rate (and how the rate is quoted), the sign
convention of flows, and the commitment-policy settings. ``Orchestrator`` joins a
repository and a spec into ``Fund`` and ``Portfolio`` objects, builds the policy and the
simulator, and runs them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from numbers import Real
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..inputs import PRIVATE_CURRENCY, Fund, Portfolio
from ..policy import AnnualRatePolicy
from ..simulator import SimulationResult, Simulator
from .repository import DataRepository, ExcelRepository, SheetNames
from .tables import canonical_name

FX_QUOTES = ("base_per_usd", "usd_per_base")
LIQUID_KINDS = ("levels", "returns")


@dataclass(frozen=True)
class SimulationSpec:
    """Everything a run needs that the tables do not carry.

    ``liquid_series`` and ``fx_series`` name columns of market_data. ``liquid_kind`` says
    what the liquid column holds: ``levels`` (used as is; the first level is the starting
    balance) or ``returns`` (simple per-period returns, compounded from ``initial_value``;
    the first row's return is the return *into* the first observation and is not applied).
    ``fx_quote`` says how the rate is quoted: ``base_per_usd`` (GBP per 1 USD, used as is)
    or ``usd_per_base`` (USD per 1 GBP, inverted). ``commitment_rates`` overrides the
    repository's rate table when given. ``calls_are_negative`` is the sign convention of
    ``Flow`` rows: negative values are calls and positive values distributions (the LP's
    view); set False for the opposite. ``Call`` and ``Distribution`` rows are read as
    magnitudes regardless.
    """

    base_currency: str
    liquid_series: str
    fx_series: str | None = None
    fx_quote: str = "base_per_usd"
    liquid_kind: str = "levels"
    initial_value: float | None = None
    commitment_rates: Any = None
    weights: Mapping[str, float] | None = None
    carry_forward: bool = False
    calls_are_negative: bool = True
    stop_on_shortfall: bool = True
    cash_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        if self.fx_quote not in FX_QUOTES:
            raise ValueError(f"fx_quote must be one of {FX_QUOTES}, got {self.fx_quote!r}")
        if self.liquid_kind not in LIQUID_KINDS:
            raise ValueError(f"liquid_kind must be one of {LIQUID_KINDS}, got {self.liquid_kind!r}")
        if not isinstance(self.liquid_series, str) or not self.liquid_series.strip():
            raise ValueError("liquid_series must name a market_data column")
        if self.liquid_kind == "returns":
            value = self.initial_value  # numbers.Real: numpy scalars count, bool is excluded explicitly
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError("initial_value (the starting liquid balance) must be a positive number when liquid_kind is 'returns'")


def returns_to_levels(returns: pd.Series, initial_value: float) -> pd.Series:
    """Total-return levels from simple per-period returns.

    ``level[0] = initial_value`` and ``level[t] = level[t-1] × (1 + returns[t])``. The first
    return is the return into the first observation — before the simulation starts — so it
    is not applied; the engine's first-period factor is 1 either way.
    """
    return_factors = 1.0 + returns.to_numpy(dtype=float)
    if len(return_factors) == 0:
        raise ValueError("returns series is empty")
    if not np.isfinite(return_factors).all() or (return_factors[1:] <= 0).any():
        raise ValueError("returns must be finite and greater than -100%")
    return_factors[0] = 1.0
    return pd.Series(float(initial_value) * np.cumprod(return_factors), index=returns.index, name=returns.name)


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
        liquid = returns_to_levels(liquid, spec.initial_value)
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
    def portfolio(self) -> Portfolio:
        return build_portfolio(self.repository.market_data(), self.spec, self.commitment_rates)

    @cached_property
    def policy(self) -> AnnualRatePolicy:
        return AnnualRatePolicy(self.portfolio.commitment_rates, self.funds, self.spec.weights,
                                carry_forward=self.spec.carry_forward, years=self.portfolio.calendar_years)

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


def load_tables_workbook(path: Any, spec: SimulationSpec, sheets: SheetNames = SheetNames()) -> Orchestrator:
    return Orchestrator(ExcelRepository(path, sheets), spec)


def run_tables_workbook(path: Any, spec: SimulationSpec, sheets: SheetNames = SheetNames()) -> SimulationResult:
    return load_tables_workbook(path, spec, sheets).run()
