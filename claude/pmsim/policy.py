"""Commitment sizing, in US dollars, on the liquid-only value.

Two rules fix what a commitment is sized on.

It is sized in **US dollars**: private funds are committed to in dollars, whatever the
portfolio's base currency, so the engine converts before it asks and turns the answer into
base currency only for reporting.

It is sized on the **liquid-only value**: the initial value compounded by the liquid returns,
with no capital call or distribution in it. Private assets never interfere, so every
commitment follows from the liquid returns, the initial value, the exchange rates and the
schedule alone.

The engine asks a policy one question: given the funds closing at this observation and a
snapshot of the balances, how many dollars to commit to each. The policy reads and never
moves cash, and keeps no state. ``AnnualRatePolicy`` is the default: a dollar budget per
year and fund type from a pacing schedule, committed to the funds of the type closing that
year, by weight; with carry-forward, the budgets of years in which no fund of the type
closed wait, as dollars, for the next fund of the type.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from numbers import Real
from typing import Any, Iterable, Mapping, Protocol, Sequence

import pandas as pd

from .dates import years_between
from .inputs import Fund, coerce_rate_table

WEIGHT_TOLERANCE = 1e-9
FundGroups = dict[tuple[int, str], list[Fund]]  # funds by (closing year, fund type)


@dataclass(frozen=True)
class YearEndBalance:
    """The liquid-only value at a calendar year's last observation: what that year's budget is sized on."""

    date: date
    liquid_only_usd: float


@dataclass(frozen=True)
class SizingBalances:
    """What a policy may look at when sizing commitments at observation ``t``. Every amount is in US dollars.

    ``liquid_only_usd`` is what commitments are sized on: the initial value compounded by the
    liquid returns, at today's exchange rate. No call or distribution is in it.
    ``liquid_account_usd`` is the simulated account that does pay the calls and bank the
    distributions; it and ``private_nav_usd`` are here for policies that want them, and
    ``AnnualRatePolicy`` uses neither.
    """

    t: int
    date: date
    liquid_only_usd: float
    liquid_account_usd: float  # after this period's return and the existing funds' distributions
    private_nav_usd: float  # opening private NAV; natively USD, so nothing is translated
    year_ends: Mapping[int, YearEndBalance] = field(default_factory=dict)  # completed calendar years only
    first_commitment_date: date | None = None  # the observation at which the run's first fund is committed

    @property
    def total_usd(self) -> float:
        return self.liquid_account_usd + self.private_nav_usd

    def year_end_on_or_before(self, year: int) -> YearEndBalance | None:
        """The last year end on or before ``year``: a year with no observation uses the one before. None before the run began."""
        known = [y for y in self.year_ends if y <= year]
        return self.year_ends[max(known)] if known else None


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


def validate_expected_return(value: Any, *, label: str = "expected_return") -> float:
    """A yearly expected return as a decimal: 0.05 is 5%. At or above 1 it is taken for a percentage typed as a number."""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{label} must be a number such as 0.05 for 5% a year, got {value!r}")
    if not -1.0 < value < 1.0:
        raise ValueError(f"{label} must be a decimal between -1 and 1 (write 5% as 0.05), got {value!r}")
    return float(value)


def _weights_by_fund(groups: FundGroups, weights: Mapping[str, float]) -> dict[str, float]:
    """Each fund's share of its group's budget: given for all of a group or none (equal split), summing to 1."""
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

    Only the years and their rates are settled here. The dollars are not: a year's budget
    depends on that year's liquid-only value, which the run supplies.
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

    **The budget.** ``share × liquid-only value in USD``, on the day the year is sized: the
    closing observation for a year with a fund of the type, the year's last observation
    otherwise.

    **The share.** With ``expected_return`` the rate table is a pacing schedule: amounts per
    1 of liquid value on the day of the first commitment, from a model in which the liquid
    portfolio then grows at the expected return X. The schedule rises because that portfolio
    grows, so it is turned back into a share of the liquid value before it is used::

        expected_value(d) = (1 + X) ** years from the first commitment date to d     # exactly 1 on that date
        share(d)          = schedule[year, type] / expected_value(d)

    which commits the model's planned amount scaled by how far the actual liquid value is
    ahead of, or behind, the expected one. X and the schedule are in the portfolio's own
    currency, so the share has no unit and multiplies the liquid-only value in USD directly.
    Without ``expected_return`` the table is taken to hold shares of the liquid value already.

    **Carry-forward.** With ``carry_forward``, a year in which no fund of a type closes is
    still sized, on its own liquid-only value at its own year end, and the dollars
    accumulate until the next year that has a fund of the type, whose funds collect them.
    Dollars are carried, never percentages. A year before the first observation had no
    portfolio to size on and carries nothing. Without it such a year is not used at all.

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
        expected_return: float | None = None,
    ) -> None:
        self.rates = coerce_rate_table(rates)
        self.carry_forward = bool(carry_forward)
        self.expected_return = None if expected_return is None else validate_expected_return(expected_return)
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

    def expected_value(self, on: date, first_commitment_date: date | None) -> float:
        """The pacing model's liquid value on ``on``: 1 on the first commitment date, growing at the expected return."""
        if self.expected_return is None:
            return 1.0
        if first_commitment_date is None:
            raise ValueError("expected_return needs SizingBalances.first_commitment_date: the pacing model's value is 1 on that date")
        return (1.0 + self.expected_return) ** years_between(first_commitment_date, on)

    def _budgets_usd(self, fund_name: str, balances: SizingBalances) -> tuple[float, float]:
        """This year's budget and the budgets carried to it, in USD, before the fund's weight is applied."""
        entitlement = self.entitlements[fund_name]
        first = balances.first_commitment_date
        current_year_usd = entitlement.current_year_rate / self.expected_value(balances.date, first) * balances.liquid_only_usd
        carried = []
        for year, rate in entitlement.carried_year_rates.items():
            year_end = balances.year_end_on_or_before(year)
            if year_end is not None:  # each carried year on its own year end: its own value, its own expected value
                carried.append(rate / self.expected_value(year_end.date, first) * year_end.liquid_only_usd)
        return current_year_usd, math.fsum(carried)

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
        expected_value = (float("nan") if self.expected_return is None
                          else self.expected_value(balances.date, balances.first_commitment_date))
        return {
            "policy_year": entitlement.policy_year, "current_year_rate": entitlement.current_year_rate,
            "expected_value": expected_value, "weight": entitlement.weight,
            "current_year_usd": current_year_usd, "carried_usd": carried_usd,
            "carried_years": ", ".join(str(year) for year in entitlement.carried_year_rates),
        }
