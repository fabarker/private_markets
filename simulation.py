"""Deterministic liquid/private portfolio simulation with percentage carryforward.

Public entry point: Simulation(SimulationConfig(...)).run(). The first liquid
index value is initial dollar wealth. Later index ratios are investment returns
applied to the simulated cash balance. See README.md for the timing contract.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
import math
from numbers import Integral
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from simulation_events import prepare_fund_events, validate_model_dates
from vintage import FundVintage

__all__ = ["Simulation", "SimulationConfig", "SimulationResult", "LiquidityShortfall",
           "Diagnostic", "SimulationValidationError", "ValuationError"]


class SimulationValidationError(ValueError):
    """Invalid configuration or fund history; no simulation was completed."""


class ValuationError(ValueError):
    """A cash-flow-adjusted fund NAV became materially negative."""

    def __init__(self, fund_id: str, event_date: date, nav: float):
        self.fund_id, self.event_date, self.nav = fund_id, event_date, nav
        super().__init__(
            f"Negative inferred NAV for {fund_id!r} on {event_date}: {nav:,.10g}. "
            "Supply an adequate NAV mark; no growth is assumed between marks."
        )


@dataclass(frozen=True)
class SimulationConfig:
    liquid_total_return_index: pd.Series
    funds: Sequence[FundVintage]
    annual_commitment_rates: pd.DataFrame
    fund_weights: Mapping[str, float] = field(default_factory=dict)
    cash_tolerance: float = 1e-8


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    fund_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LiquidityShortfall:
    date: pd.Timestamp
    interval_start: pd.Timestamp
    liquid_open: float
    return_factor: float
    distributions_existing: float
    distributions_new: float
    cash_available_for_calls: float
    calls_required: float
    deficit: float
    calls_by_fund: pd.Series
    candidate_commitments: pd.DataFrame
    candidate_budget: pd.DataFrame


@dataclass(frozen=True)
class SimulationResult:
    status: Literal["completed", "liquidity_shortfall"]
    portfolio: pd.DataFrame
    fund_detail: pd.DataFrame
    strategy_detail: pd.DataFrame
    commitment_events: pd.DataFrame
    commitment_budget: pd.DataFrame
    shortfall: LiquidityShortfall | None
    diagnostics: list[Diagnostic]


@dataclass(frozen=True)
class _Entitlement:
    current_year_rate: float
    carried_rate: float
    pooled_rate: float
    weight: float
    effective_rate: float
    source_year_rates: dict[int, float]


PORTFOLIO_COLUMNS = [
    "date", "liquid_index", "return_factor", "period_return", "liquid_open",
    "private_nav_open", "total_open", "liquid_investment_pnl", "sizing_liquid_nav",
    "new_commitments", "distributions_existing", "distributions_new", "capital_calls",
    "distributions", "net_private_cash_flow", "liquid_close", "private_nav_close",
    "total_close", "private_valuation_pnl", "cash_rounding_adjustment",
]
FUND_COLUMNS = [
    "date", "fund_id", "strategy", "active", "commitment", "new_commitment", "nav_open",
    "nav_close", "capital_calls", "distributions", "net_cash_flow", "cumulative_calls",
    "cumulative_distributions", "latest_nav_mark_date",
]
STRATEGY_COLUMNS = [
    "date", "strategy", "commitment", "new_commitments", "nav_open", "nav_close",
    "capital_calls", "distributions", "net_cash_flow",
]
COMMITMENT_COLUMNS = [
    "date", "fund_id", "strategy", "actual_closing_date", "effective_closing_date",
    "policy_year", "current_year_rate", "carried_rate", "pooled_rate", "weight",
    "effective_rate", "source_year_rates", "sizing_liquid_nav", "commitment",
]
BUDGET_COLUMNS = [
    "date", "strategy", "cumulative_target_percentage", "cumulative_used_percentage",
    "unallocated_carried_percentage", "reserved_percentage", "target_by_year",
    "used_by_year", "carried_by_year", "reserved_by_year",
]


def _frame(rows: list[dict], columns: list[str], index: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=columns)
    date_columns = {"date", "actual_closing_date", "effective_closing_date", "latest_nav_mark_date"}
    object_columns = {"fund_id", "strategy", "source_year_rates", "target_by_year", "used_by_year",
                      "carried_by_year", "reserved_by_year"}
    for column in columns:
        if column in date_columns:
            frame[column] = pd.to_datetime(frame[column])
        elif column == "active":
            frame[column] = frame[column].astype(bool)
        elif column == "policy_year":
            frame[column] = frame[column].astype("int64")
        elif column not in object_columns:
            frame[column] = frame[column].astype(float)
    return frame.set_index(index)


def _nonnegative(value: Any, label: str) -> float:
    try:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("boolean is not a rate or amount")
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SimulationValidationError(f"{label} must be a finite non-negative number") from exc
    if not math.isfinite(result) or result < 0:
        raise SimulationValidationError(f"{label} must be a finite non-negative number")
    return result


class Simulation:
    """Snapshot configuration once; each run starts with entirely fresh state.

    Annual rates accrue once per calendar year. Years without any fund of a
    type carry their rates forward. A closing-year pool is allocated to all
    that year's funds by weight, including funds beyond a partial-year horizon.
    Closings map to the next observation but keep their actual policy year.
    """

    def __init__(self, config: SimulationConfig):
        if not isinstance(config, SimulationConfig):
            raise TypeError("config must be a SimulationConfig")
        self._config = deepcopy(config)
        self._diagnostics: list[Diagnostic] = []
        try:
            self._validate_and_prepare()
        except SimulationValidationError:
            raise
        except (TypeError, ValueError, KeyError, OverflowError) as exc:
            raise SimulationValidationError(str(exc)) from exc

    def _validate_and_prepare(self) -> None:
        config = self._config
        self._tolerance = _nonnegative(config.cash_tolerance, "cash_tolerance")
        if not isinstance(config.liquid_total_return_index, pd.Series):
            raise SimulationValidationError("liquid_total_return_index must be a pandas Series")
        self._dates = validate_model_dates(config.liquid_total_return_index.index)
        self._index_values = np.asarray(config.liquid_total_return_index, dtype=float).copy()
        if not np.isfinite(self._index_values).all() or (self._index_values <= 0).any():
            raise SimulationValidationError("Liquid index values must be finite and strictly positive")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            self._factors = np.r_[1.0, self._index_values[1:] / self._index_values[:-1]]
        if not np.isfinite(self._factors).all() or (self._factors <= 0).any():
            raise SimulationValidationError("Liquid index ratios must be finite and strictly positive")
        self._years = list(range(self._dates[0].year, self._dates[-1].year + 1))
        start, end = self._dates[0].date(), self._dates[-1].date()
        funds = list(config.funds)
        for fund in funds:
            if not isinstance(fund, FundVintage):
                raise SimulationValidationError("funds must contain FundVintage objects")
            # Revalidate the copied objects: callers may have edited public lists.
            fund.__post_init__()
            if isinstance(fund.strategy, Enum):
                fund.strategy = fund.strategy.value
            if not isinstance(fund.strategy, str) or not fund.strategy.strip():
                raise SimulationValidationError(f"Fund {fund.name!r} needs a nonempty string strategy")
            if fund.commitment_date is None:
                raise SimulationValidationError(f"Fund {fund.name!r} requires a commitment_date")
            if fund.commitment_date < start:
                raise SimulationValidationError(f"Fund {fund.name!r} closes before simulation inception")
            for entry in (*fund.normalized_realized_nav, *fund.normalized_realized_net_cash_flow):
                if entry.date < fund.commitment_date and entry.value != 0:
                    raise SimulationValidationError(
                        f"Fund {fund.name!r} has a nonzero event before its closing on {entry.date}"
                    )
            if fund.commitment_size:
                self._diagnostics.append(Diagnostic(
                    "ignored_commitment_size", "The supplied commitment_size is ignored; sizing uses policy.",
                    fund.name, {"supplied_commitment_size": fund.commitment_size}))
            if fund.commitment_date > end:
                self._diagnostics.append(Diagnostic(
                    "fund_outside_horizon", "Fund closes after the simulation horizon; its weight is retained.",
                    fund.name, {"closing_date": fund.commitment_date}))
            future_flows = sum(e.date > end for e in fund.normalized_realized_net_cash_flow)
            future_marks = sum(e.date > end for e in fund.normalized_realized_nav)
            if future_flows or future_marks:
                self._diagnostics.append(Diagnostic(
                    "events_outside_horizon", "Later flows and marks are excluded without liquidating the fund.",
                    fund.name, {"flows": future_flows, "marks": future_marks}))
        if len({f.name for f in funds}) != len(funds):
            raise SimulationValidationError("Fund names must be unique")
        self._funds = tuple(sorted(funds, key=lambda f: f.name))
        self._fund_ids = [f.name for f in self._funds]
        rates = config.annual_commitment_rates
        if not isinstance(rates, pd.DataFrame):
            raise SimulationValidationError("annual_commitment_rates must be a pandas DataFrame")
        if not rates.index.is_unique or not rates.columns.is_unique:
            raise SimulationValidationError("Annual policy years and type columns must be unique")
        if any(isinstance(y, bool) or not isinstance(y, Integral) for y in rates.index):
            raise SimulationValidationError("Annual policy index must contain integer calendar years")
        if any(not isinstance(s, str) or not s.strip() for s in rates.columns):
            raise SimulationValidationError("Annual policy columns must be nonempty fund-type strings")
        self._strategies = sorted(set(rates.columns) | {f.strategy for f in self._funds})
        self._rates: dict[tuple[int, str], float] = {}
        for year in self._years:
            for strategy in self._strategies:
                if year not in rates.index or strategy not in rates.columns:
                    raise SimulationValidationError(f"Missing commitment rate for {year} / {strategy}")
                self._rates[year, strategy] = _nonnegative(
                    rates.loc[year, strategy], f"Commitment rate for {year} / {strategy}")
        excluded = [int(y) for y in rates.index if y not in self._years]
        if excluded:
            self._diagnostics.append(Diagnostic(
                "policy_outside_horizon", "Policy years outside the model are excluded; opening carry is zero.",
                details={"years": sorted(excluded)}))
        unknown_weights = set(config.fund_weights) - set(self._fund_ids)
        if unknown_weights:
            raise SimulationValidationError(f"Weights reference unknown funds: {sorted(unknown_weights)}")
        groups: dict[tuple[int, str], list[FundVintage]] = defaultdict(list)
        for fund in self._funds:
            groups[fund.commitment_date.year, fund.strategy].append(fund)
        self._weights = {}
        for key, group in groups.items():
            for fund in group:
                if len(group) > 1 and fund.name not in config.fund_weights:
                    raise SimulationValidationError(f"Missing weight for {fund.name!r} in closing group {key}")
                self._weights[fund.name] = _nonnegative(
                    config.fund_weights.get(fund.name, 1.0), f"Weight for {fund.name!r}")
            total = math.fsum(self._weights[f.name] for f in group)
            if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-12):
                raise SimulationValidationError(f"Weights for closing group {key} must sum to 1; got {total}")
        self._prepare_budgets(groups)
        n, m = len(self._dates), len(self._funds)
        self._paths = tuple(prepare_fund_events(f, self._dates) for f in self._funds)
        self._unit_calls = np.column_stack([p.calls for p in self._paths]) if m else np.zeros((n, 0))
        self._unit_distributions = np.column_stack([p.distributions for p in self._paths]) if m else np.zeros((n, 0))
        self._unit_nav = np.column_stack([p.nav for p in self._paths]) if m else np.zeros((n, 0))
        self._closing_indices = np.array([
            self._dates.searchsorted(pd.Timestamp(f.commitment_date), side="left") for f in self._funds
        ], dtype=int)
        self._strategy_indices = {
            s: np.array([i for i, f in enumerate(self._funds) if f.strategy == s], dtype=int)
            for s in self._strategies
        }
        self._diagnostics.append(Diagnostic(
            "flow_convention", "Flows settle after period returns. Gross reporting requires separate signed "
            "call and distribution entries; previously netted source flows cannot be decomposed."))

    def _prepare_budgets(self, groups: dict) -> None:
        self._entitlements: dict[str, _Entitlement] = {}
        self._annual_budgets: dict[tuple[int, str], tuple[dict, dict]] = {}
        for strategy in self._strategies:
            carry, targets = {}, {}
            for year in self._years:
                rate = self._rates[year, strategy]
                targets[year] = rate
                carried_rate = math.fsum(carry.values())
                carry[year] = rate
                pool = math.fsum(carry.values())
                cohort = groups.get((year, strategy), ())
                for fund in cohort:
                    weight = self._weights[fund.name]
                    self._entitlements[fund.name] = _Entitlement(
                        rate, carried_rate, pool, weight, pool * weight,
                        {source: amount * weight for source, amount in carry.items()},
                    )
                if cohort:
                    carry = {}
                self._annual_budgets[year, strategy] = (targets.copy(), carry.copy())

    def _budget_rows(self, observation: pd.Timestamp, used: dict) -> list[dict]:
        rows = []
        for strategy in self._strategies:
            targets, carry = self._annual_budgets[observation.year, strategy]
            used_sources = used[strategy]
            reserved = {
                year: max(0.0, amount - carry.get(year, 0.0) - used_sources.get(year, 0.0))
                for year, amount in targets.items()
            }
            rows.append({
                "date": observation, "strategy": strategy,
                "cumulative_target_percentage": math.fsum(targets.values()),
                "cumulative_used_percentage": math.fsum(used_sources.values()),
                "unallocated_carried_percentage": math.fsum(carry.values()),
                "reserved_percentage": math.fsum(reserved.values()),
                "target_by_year": targets.copy(), "used_by_year": used_sources.copy(),
                "carried_by_year": carry.copy(), "reserved_by_year": reserved,
            })
        return rows

    def _commitment_record(self, i: int, observation: pd.Timestamp, base: float, amount: float) -> dict:
        fund = self._funds[i]
        entitlement = self._entitlements[fund.name]
        return {
            "date": observation, "fund_id": fund.name, "strategy": fund.strategy,
            "actual_closing_date": pd.Timestamp(fund.commitment_date),
            "effective_closing_date": observation, "policy_year": fund.commitment_date.year,
            "current_year_rate": entitlement.current_year_rate,
            "carried_rate": entitlement.carried_rate, "pooled_rate": entitlement.pooled_rate,
            "weight": entitlement.weight, "effective_rate": entitlement.effective_rate,
            "source_year_rates": entitlement.source_year_rates.copy(),
            "sizing_liquid_nav": base, "commitment": amount,
        }

    def run(self) -> SimulationResult:
        """Run from inception; return partial results on the first liquidity shortfall.

        Structural input errors raise SimulationValidationError at construction.
        ValuationError identifies a negative inferred NAV in a reached period.
        Neither a successful nor failed run mutates the config or input funds.
        """
        m = len(self._funds)
        commitments, nav_open, cumulative_calls, cumulative_distributions = (np.zeros(m) for _ in range(4))
        active = np.zeros(m, dtype=bool)
        used: dict[str, dict[int, float]] = {s: {} for s in self._strategies}
        liquid_open = float(self._index_values[0])
        portfolio_rows, fund_rows, strategy_rows, commitment_rows, budget_rows = [], [], [], [], []
        shortfall = None
        for k, observation in enumerate(self._dates):
            factor = float(self._factors[k])
            cohort = np.flatnonzero(self._closing_indices == k)
            with np.errstate(over="raise", invalid="raise"):
                liquid_after_return = liquid_open * factor
                distributions_existing = math.fsum((self._unit_distributions[k] * commitments).tolist())
                base = liquid_after_return + distributions_existing
                candidate = commitments.copy()
                for i in cohort:
                    candidate[i] = self._entitlements[self._funds[i].name].effective_rate * base
                calls = self._unit_calls[k] * candidate
                distributions = self._unit_distributions[k] * candidate
            candidate_events = [self._commitment_record(i, observation, base, float(candidate[i])) for i in cohort]
            new_distributions = math.fsum(distributions[cohort].tolist())
            total_calls = math.fsum(calls.tolist())
            total_distributions = math.fsum(distributions.tolist())
            available = base + new_distributions
            liquid_candidate = available - total_calls
            if not all(math.isfinite(v) for v in (base, available, liquid_candidate, *candidate)):
                raise ArithmeticError(f"Non-finite portfolio arithmetic on {observation.date()}")
            if liquid_candidate < -self._tolerance:
                pending_rows = self._budget_rows(observation, used)
                for row in pending_rows:
                    row["attempted_used_percentage"] = math.fsum(
                        e["effective_rate"] for e in candidate_events if e["strategy"] == row["strategy"])
                shortfall = LiquidityShortfall(
                    observation, self._dates[max(0, k - 1)], liquid_open, factor,
                    distributions_existing, new_distributions, available, total_calls, -liquid_candidate,
                    pd.Series(calls, index=pd.Index(self._fund_ids, name="fund_id"), name="capital_calls"),
                    _frame(candidate_events, COMMITMENT_COLUMNS, ["date", "fund_id"]),
                    _frame(pending_rows, BUDGET_COLUMNS + ["attempted_used_percentage"], ["date", "strategy"]),
                )
                break
            candidate_active = active.copy()
            candidate_active[cohort] = True
            for i in np.flatnonzero(candidate_active):
                minimum = self._paths[i].minimum_nav[k] * candidate[i]
                if minimum < -self._tolerance:
                    raise ValuationError(self._funds[i].name, self._paths[i].minimum_nav_date[k], minimum)
            with np.errstate(over="raise", invalid="raise"):
                nav_close = np.maximum(0.0, self._unit_nav[k] * candidate)
            if not np.isfinite(nav_close).all():
                raise ArithmeticError(f"Non-finite private NAV on {observation.date()}")
            liquid_close = max(0.0, liquid_candidate)
            rounding = liquid_close - liquid_candidate
            new_commitments = candidate - commitments
            cumulative_calls += calls
            cumulative_distributions += distributions
            for i in cohort:
                fund = self._funds[i]
                for source, rate in self._entitlements[fund.name].source_year_rates.items():
                    used[fund.strategy][source] = math.fsum((used[fund.strategy].get(source, 0.0), rate))
            private_open, private_close = math.fsum(nav_open.tolist()), math.fsum(nav_close.tolist())
            portfolio_rows.append({
                "date": observation, "liquid_index": self._index_values[k], "return_factor": factor,
                "period_return": factor - 1, "liquid_open": liquid_open,
                "private_nav_open": private_open, "total_open": liquid_open + private_open,
                "liquid_investment_pnl": liquid_after_return - liquid_open, "sizing_liquid_nav": base,
                "new_commitments": math.fsum(new_commitments.tolist()),
                "distributions_existing": distributions_existing, "distributions_new": new_distributions,
                "capital_calls": total_calls, "distributions": total_distributions,
                "net_private_cash_flow": total_distributions - total_calls,
                "liquid_close": liquid_close, "private_nav_close": private_close,
                "total_close": liquid_close + private_close,
                "private_valuation_pnl": private_close - private_open - total_calls + total_distributions,
                "cash_rounding_adjustment": rounding,
            })
            for i, fund in enumerate(self._funds):
                fund_rows.append({
                    "date": observation, "fund_id": fund.name, "strategy": fund.strategy,
                    "active": bool(candidate_active[i]), "commitment": candidate[i],
                    "new_commitment": new_commitments[i], "nav_open": nav_open[i], "nav_close": nav_close[i],
                    "capital_calls": calls[i], "distributions": distributions[i],
                    "net_cash_flow": distributions[i] - calls[i], "cumulative_calls": cumulative_calls[i],
                    "cumulative_distributions": cumulative_distributions[i],
                    "latest_nav_mark_date": self._paths[i].latest_mark_date[k] if candidate_active[i] else None,
                })
            for strategy, ix in self._strategy_indices.items():
                strategy_rows.append({
                    "date": observation, "strategy": strategy,
                    "commitment": math.fsum(candidate[ix].tolist()),
                    "new_commitments": math.fsum(new_commitments[ix].tolist()),
                    "nav_open": math.fsum(nav_open[ix].tolist()), "nav_close": math.fsum(nav_close[ix].tolist()),
                    "capital_calls": math.fsum(calls[ix].tolist()),
                    "distributions": math.fsum(distributions[ix].tolist()),
                    "net_cash_flow": math.fsum((distributions[ix] - calls[ix]).tolist()),
                })
            commitment_rows.extend(candidate_events)
            budget_rows.extend(self._budget_rows(observation, used))
            commitments, nav_open, active, liquid_open = candidate, nav_close, candidate_active, liquid_close
        diagnostics = deepcopy(self._diagnostics)
        for row in budget_rows[-len(self._strategies):] if self._strategies else []:
            if row["unallocated_carried_percentage"] > 1e-12 or row["reserved_percentage"] > 1e-12:
                diagnostics.append(Diagnostic(
                    "outstanding_percentage", "Percentage remains outstanding at the last completed observation.",
                    details={"strategy": row["strategy"], "date": row["date"],
                             "carried": row["unallocated_carried_percentage"], "reserved": row["reserved_percentage"]}))
        return SimulationResult(
            "liquidity_shortfall" if shortfall else "completed",
            _frame(portfolio_rows, PORTFOLIO_COLUMNS, ["date"]),
            _frame(fund_rows, FUND_COLUMNS, ["date", "fund_id"]),
            _frame(strategy_rows, STRATEGY_COLUMNS, ["date", "strategy"]),
            _frame(commitment_rows, COMMITMENT_COLUMNS, ["date", "fund_id"]),
            _frame(budget_rows, BUDGET_COLUMNS, ["date", "strategy"]),
            shortfall, diagnostics,
        )

    def simulate(self) -> SimulationResult:
        """Alias for run(), with fresh state on every invocation."""
        return self.run()
