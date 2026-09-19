"""Commitment sizing, in US dollars.

Private funds are committed to in dollars, so sizing happens exclusively in USD whatever
the portfolio's base currency. The engine asks a policy one question: given the funds
closing at this observation and a snapshot of the balances in USD, how many dollars to
commit to each. For a non-USD portfolio the engine converts the liquid balance at the
observation's exchange rate before it asks, and turns the answer into base currency only
for reporting. The policy reads balances and never moves cash. ``AnnualRatePolicy`` is the
default: a dollar budget per year and fund type — that year's rate times that year's liquid
balance — committed to the funds of the type closing that year, by weight; with
carry-forward, the budgets of years in which no fund of the type closed wait, as dollars,
for the next fund of the type.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Mapping, Protocol, Sequence

import pandas as pd

from .inputs import Fund, coerce_rate_table

WEIGHT_TOLERANCE = 1e-9
FundGroups = dict[tuple[int, str], list[Fund]]  # funds by (closing year, fund type)


@dataclass(frozen=True)
class SizingBalances:
    """What a policy may look at when sizing commitments at observation ``t``. Every amount is in US dollars."""

    t: int
    date: date
    liquid_usd: float  # liquid balance after this period's return and existing funds' distributions, at today's rate
    private_nav_usd: float  # opening private NAV; private data is natively USD, so nothing is translated
    # liquid_usd as it stood at the last observation of each completed calendar year: what a carried year's budget is sized on
    year_end_liquid_usd: Mapping[int, float] = field(default_factory=dict)

    @property
    def total_usd(self) -> float:
        return self.liquid_usd + self.private_nav_usd

    def liquid_usd_at_end_of(self, year: int) -> float | None:
        """``liquid_usd`` at the last observation on or before the end of ``year``; None before the simulation began."""
        known = [y for y in self.year_end_liquid_usd if y <= year]
        return self.year_end_liquid_usd[max(known)] if known else None


class CommitmentPolicy(Protocol):
    def size_commitments(self, cohort: Sequence[Fund], balances: SizingBalances) -> Mapping[str, float]:
        """US-dollar commitment for each fund in ``cohort``, keyed by fund name."""


@dataclass(frozen=True)
class Entitlement:
    """What a fund collects at its closing: its weight of this year's budget and of every budget carried to its year."""

    policy_year: int
    current_year_rate: float
    weight: float
    carried_year_rates: Mapping[int, float] = field(default_factory=dict)  # earlier years with no fund of the type → their rates


def _weights_by_fund(groups: FundGroups, weights: Mapping[str, float]) -> dict[str, float]:
    """Each fund's share of its group's pooled rate: given for all of a group or none (equal split), summing to 1."""
    shares: dict[str, float] = {}
    for (year, fund_type), group in groups.items():
        given = [f for f in group if f.name in weights]
        if not given:
            for f in group:
                shares[f.name] = 1.0 / len(group)
            continue
        if len(given) != len(group):
            raise ValueError(f"weights for {fund_type} funds closing in {year} must be given for all of them or none")
        for f in group:
            weight = float(weights[f.name])
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"weight for {f.name!r} must be a finite non-negative number")
            shares[f.name] = weight
        total = math.fsum(shares[f.name] for f in group)
        if abs(total - 1.0) > WEIGHT_TOLERANCE:
            raise ValueError(f"weights for {fund_type} funds closing in {year} sum to {total:.6g}, not 1")
    return shares


def _entitlements(rates: pd.DataFrame, groups: FundGroups, weights: Mapping[str, float],
                  carry_forward: bool) -> dict[str, Entitlement]:
    """Walk each fund type's years in order: a year with a closing hands its funds every year carried to it.

    Only the years and their rates are settled here. The dollars are not: a carried year's
    budget depends on that year's portfolio value, which is known only during the run.
    """
    entitlements: dict[str, Entitlement] = {}
    for fund_type in rates.columns:
        carried: dict[int, float] = {}
        for year in rates.index:
            rate = float(rates.at[year, fund_type])
            cohort = groups.get((int(year), fund_type), [])
            for f in cohort:
                entitlements[f.name] = Entitlement(int(year), rate, weights[f.name], dict(carried))
            if cohort:
                carried = {}
            elif carry_forward and rate > 0:
                carried[int(year)] = rate
    return entitlements


class AnnualRatePolicy:
    """A dollar budget per year and fund type, committed to the funds of that type closing that year.

    A year's budget is ``rate[year, type]`` × the liquid balance in USD. The closing year's
    budget is sized at the closing observation. With ``carry_forward``, a year in which no
    fund of a type closes is still sized — at that year's last observation, on that year's
    balance — and its dollars accumulate until the next year that has a fund of the type,
    whose funds collect them. Dollars are carried, never percentages: three carried years
    are three budgets, each from its own year's portfolio value. A year before the first
    observation had no portfolio to size on and carries nothing. Without ``carry_forward`` a
    year with no closing of the type is not used at all.

    ``weights`` split a year's budget, carried dollars included, among the funds of one type
    closing that year; they must be given for all funds of such a group or none (equal
    split), and sum to 1. ``years`` lists the calendar years the rate table must cover (the
    simulator passes the horizon); fund closing years are always required.
    """

    def __init__(
        self,
        rates: Any,
        funds: Sequence[Fund],
        weights: Mapping[str, float] | None = None,
        *,
        carry_forward: bool = False,
        years: Iterable[int] | None = None,
    ) -> None:
        self.rates = coerce_rate_table(rates)
        self.carry_forward = bool(carry_forward)
        funds = list(funds)
        names = [f.name for f in funds]
        if len(set(names)) != len(names):
            raise ValueError("fund names must be unique")
        weights = dict(weights or {})
        unknown = set(weights) - set(names)
        if unknown:
            raise ValueError(f"weights name funds that are not in the fund list: {sorted(unknown)}")

        missing_types = {f.fund_type for f in funds} - set(self.rates.columns)
        if missing_types:
            raise ValueError(f"commitment_rates has no column for fund type(s) {sorted(missing_types)}")
        if len(self.rates.columns):
            required_years = set(years or ()) | {f.closing_year for f in funds}
            missing_years = required_years - set(self.rates.index)
            if missing_years:
                raise ValueError(f"commitment_rates has no row for year(s) {sorted(missing_years)}")

        groups: FundGroups = {}
        for fund in funds:
            groups.setdefault((fund.closing_year, fund.fund_type), []).append(fund)
        self.weights = _weights_by_fund(groups, weights)
        self.entitlements = _entitlements(self.rates, groups, self.weights, self.carry_forward)

    def _budgets_usd(self, fund_name: str, balances: SizingBalances) -> tuple[float, float]:
        """This year's budget and the budgets carried to it, in USD, before the fund's weight is applied."""
        entitlement = self.entitlements[fund_name]
        current_year_usd = entitlement.current_year_rate * balances.liquid_usd
        carried_usd = math.fsum(
            rate * (balances.liquid_usd_at_end_of(year) or 0.0)  # each carried year on its own year-end balance
            for year, rate in entitlement.carried_year_rates.items()
        )
        return current_year_usd, carried_usd

    def size_commitments(self, cohort: Sequence[Fund], balances: SizingBalances) -> Mapping[str, float]:
        sized = {}
        for fund in cohort:
            current_year_usd, carried_usd = self._budgets_usd(fund.name, balances)
            sized[fund.name] = self.entitlements[fund.name].weight * (current_year_usd + carried_usd)
        return sized

    def explain_commitment(self, fund_name: str, balances: SizingBalances) -> dict[str, Any]:
        """How a fund's dollars were arrived at, for the commitments table: weight × (current_year_usd + carried_usd)."""
        entitlement = self.entitlements[fund_name]
        current_year_usd, carried_usd = self._budgets_usd(fund_name, balances)
        return {
            "policy_year": entitlement.policy_year, "current_year_rate": entitlement.current_year_rate,
            "weight": entitlement.weight, "current_year_usd": current_year_usd, "carried_usd": carried_usd,
            "carried_years": ", ".join(str(year) for year in entitlement.carried_year_rates),
        }
