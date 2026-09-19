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
moves cash, and keeps no state. ``AnnualRatePolicy`` is the default: the pacing schedule
gives a dollar budget per year and fund type, and each fund collects the budgets of the
years it **draws**. Which years those are is its draw plan — by default its own closing
year, plus, with carry-forward, the years its type passed without a closing; or, stated
explicitly, any set of years at any multiple, including years after the closing.
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
DrawPlans = Mapping[str, Mapping[int, float]]  # fund name → calendar year → multiplier


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

    ``year_ends`` holds completed calendar years only. That is the engine's guarantee against
    look-ahead, and it is why a fund that draws a year the run has not reached is funded from
    its closing observation instead.
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
class DrawnYear:
    """One schedule year a fund draws, and the dollars that year's commitment came to.

    There are two dates, and they differ only for a year the run had not reached when the
    fund closed. ``plan_date`` is where the pacing model puts the year, and is what the rate
    is divided by, so the year contributes the share the schedule meant it to.
    ``funding_date`` is the observation whose liquid-only value the dollars are taken from:
    the year's own year end when the run has reached it, the closing observation otherwise,
    and never anything later.
    """

    year: int
    multiplier: float
    rate: float
    plan_date: date
    expected_value: float
    funding_date: date
    liquid_only_usd: float
    commitment_usd: float  # multiplier × rate / expected_value × liquid_only_usd


@dataclass(frozen=True)
class Entitlement:
    """The schedule years a fund draws, each with a multiplier, and its weight of their total."""

    policy_year: int  # the fund's closing year
    fund_type: str
    weight: float
    draws: Mapping[int, float] = field(default_factory=dict)  # calendar year → multiplier, in ascending year order


def validate_expected_return(value: Any, *, label: str = "expected_return") -> float:
    """A yearly expected return as a decimal: 0.05 is 5%. At or above 1 it is taken for a percentage typed as a number."""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{label} must be a number such as 0.05 for 5% a year, got {value!r}")
    if not -1.0 < value < 1.0:
        raise ValueError(f"{label} must be a decimal between -1 and 1 (write 5% as 0.05), got {value!r}")
    return float(value)


def format_draw_plan(draws: Mapping[int, float]) -> str:
    """``2010, 2011, 2012`` — a multiplier other than 1 shown against its year, as ``2021x3``."""
    return ", ".join(f"{year}x{multiplier:g}" if multiplier != 1 else str(year)
                     for year, multiplier in draws.items())


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


def _checked_draw_plans(plans: DrawPlans, groups: FundGroups, rates: pd.DataFrame) -> dict[str, dict[int, float]]:
    """The plans, years put in ascending order, checked against the fund list and the rate table."""
    funds_by_name = {f.name: f for group in groups.values() for f in group}
    unknown = set(plans) - set(funds_by_name)
    if unknown:
        raise ValueError(f"draws name funds that are not in the fund list: {sorted(unknown)}")

    checked: dict[str, dict[int, float]] = {}
    for name, plan in plans.items():
        if not plan:
            raise ValueError(f"draws for {name!r} is empty: leave the fund out to keep its default years")
        years: dict[int, float] = {}
        for year, multiplier in plan.items():
            year, multiplier = int(year), float(multiplier)
            if year not in rates.index:
                raise ValueError(f"draws for {name!r} name year {year}, which commitment_rates has no row for")
            if not math.isfinite(multiplier) or multiplier < 0:
                raise ValueError(f"multiplier for year {year} of {name!r} must be a finite non-negative number")
            years[year] = multiplier
        checked[name] = dict(sorted(years.items()))

    # A schedule year is one closing's to spend. Funds closing together share it by weight.
    claimed_by: dict[tuple[str, int], int] = {}
    for name, plan in checked.items():
        fund = funds_by_name[name]
        for year in plan:
            first = claimed_by.setdefault((fund.fund_type, year), fund.closing_year)
            if first != fund.closing_year:
                raise ValueError(f"{fund.fund_type} year {year} is drawn by funds closing in both {first} and "
                                 f"{fund.closing_year}; a schedule year belongs to one closing")
    return checked


def _entitlements(rates: pd.DataFrame, groups: FundGroups, weights: Mapping[str, float],
                  carry_forward: bool, plans: Mapping[str, Mapping[int, float]]) -> dict[str, Entitlement]:
    """Walk each fund type's years in order and settle which years each of its funds draws.

    A fund named in ``plans`` draws exactly what its plan says. Otherwise it draws its own
    closing year, plus — with carry-forward — every year of its type that passed without one.
    A plan switches carry-forward off for its whole fund type: once the years are named,
    pooling the remaining ones behind them would commit dollars nobody asked for.

    Only the years and their multipliers are settled here. The dollars are not: each year's
    depends on a liquid-only value the run supplies.
    """
    entitlements: dict[str, Entitlement] = {}
    for fund_type in rates.columns:
        cohorts = {year: group for (year, of_type), group in groups.items() if of_type == fund_type}
        planned = any(f.name in plans for group in cohorts.values() for f in group)
        carry = carry_forward and not planned
        waiting: dict[int, float] = {}
        for year in rates.index:
            cohort = cohorts.get(int(year), [])
            for fund in cohort:
                draws = plans.get(fund.name) or dict(sorted({int(year): 1.0, **waiting}.items()))
                entitlements[fund.name] = Entitlement(int(year), fund_type, weights[fund.name], dict(draws))
            if cohort:
                waiting = {}
            elif carry and float(rates.at[year, fund_type]) > 0:
                waiting[int(year)] = 1.0
    return entitlements


class AnnualRatePolicy:
    """A dollar budget per year and fund type, collected by the funds that draw those years.

    **The budget.** ``share × liquid-only value in USD``, on the day the year is funded: the
    closing observation for the fund's own year, the year's last observation otherwise.

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

    **Which years a fund draws.** By default its own closing year alone. With
    ``carry_forward``, also every year in which no fund of its type closed: those years are
    still sized, each on its own liquid-only value at its own year end, and their dollars
    accumulate until the next fund of the type collects them. Dollars are carried, never
    percentages. A year before the first observation had no portfolio to size on and carries
    nothing.

    ``draws`` states the years instead, per fund, as ``{calendar year: multiplier}``: a
    secondaries subscription can take four vintages' worth of budget, or three times one
    year's. Each drawn year is priced as its own dollar commitment and the dollars are added.
    A year **after** the closing is funded from the closing observation, since the run has not
    reached that year and a commitment is fixed the day it is made; its rate is still divided
    by the pacing model's value on its own date, so it contributes the share the schedule
    meant it to rather than one inflated by the growth expected in between. Naming any fund of
    a type in ``draws`` switches carry-forward off for that type.

    ``weights`` split a year's budget among the funds of one type closing that year; they must
    be given for all funds of such a group or none (equal split), and sum to 1. ``years``
    lists the calendar years the rate table must cover (the simulator passes the horizon);
    fund closing years and drawn years are always required.
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
        draws: DrawPlans | None = None,
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
        self.draws = _checked_draw_plans(dict(draws or {}), groups, self.rates)
        self.entitlements = _entitlements(self.rates, groups, self.weights, self.carry_forward, self.draws)

    def expected_value(self, on: date, first_commitment_date: date | None) -> float:
        """The pacing model's liquid value on ``on``: 1 on the first commitment date, growing at the expected return."""
        if self.expected_return is None:
            return 1.0
        if first_commitment_date is None:
            raise ValueError("expected_return needs SizingBalances.first_commitment_date: the pacing model's value is 1 on that date")
        return (1.0 + self.expected_return) ** years_between(first_commitment_date, on)

    def unclaimed_schedule_years(self) -> dict[str, list[int]]:
        """Per fund type, the years with a rate above zero that no fund draws: the schedule's unspent budget."""
        drawn: dict[str, set[int]] = {}
        for entitlement in self.entitlements.values():
            drawn.setdefault(entitlement.fund_type, set()).update(entitlement.draws)
        return {
            fund_type: [int(year) for year in self.rates.index
                        if float(self.rates.at[year, fund_type]) > 0 and int(year) not in drawn.get(fund_type, set())]
            for fund_type in self.rates.columns
        }

    def drawn_years(self, fund_name: str, balances: SizingBalances) -> list[DrawnYear]:
        """Every schedule year the fund draws, with the dollars each one's commitment came to.

        A year before the closing is funded from its own year end, as carry-forward has always
        done. A year at or after the closing is funded from the closing observation, because
        that is the last balance known when the commitment is fixed. The rate is always
        divided by the pacing model's value on the drawn year's own date.
        """
        entitlement = self.entitlements[fund_name]
        first = balances.first_commitment_date
        drawn: list[DrawnYear] = []
        for year, multiplier in entitlement.draws.items():
            rate = float(self.rates.at[year, entitlement.fund_type])
            if year < entitlement.policy_year:
                year_end = balances.year_end_on_or_before(year)
                if year_end is None:
                    continue  # a year before the run began had no portfolio to size on
                plan_date = funding_date = year_end.date
                liquid_only_usd = year_end.liquid_only_usd
            else:
                funding_date, liquid_only_usd = balances.date, balances.liquid_only_usd
                # the fund's own year is dated by the closing; a later year by the model's own calendar
                plan_date = balances.date if year == entitlement.policy_year else date(year, 12, 31)
            expected_value = self.expected_value(plan_date, first)
            drawn.append(DrawnYear(year, multiplier, rate, plan_date, expected_value, funding_date,
                                   liquid_only_usd, multiplier * rate / expected_value * liquid_only_usd))
        return drawn

    def _budgets_usd(self, fund_name: str, balances: SizingBalances) -> tuple[float, float]:
        """The dollars from the fund's own closing year, and the dollars from every other year it draws."""
        drawn = self.drawn_years(fund_name, balances)
        own_year = self.entitlements[fund_name].policy_year
        return (math.fsum(d.commitment_usd for d in drawn if d.year == own_year),
                math.fsum(d.commitment_usd for d in drawn if d.year != own_year))

    def size_commitments(self, cohort: Sequence[Fund], balances: SizingBalances) -> Mapping[str, float]:
        sized = {}
        for fund in cohort:
            own_year_usd, other_years_usd = self._budgets_usd(fund.name, balances)
            sized[fund.name] = self.entitlements[fund.name].weight * (own_year_usd + other_years_usd)
        return sized

    def explain_commitment(self, fund_name: str, balances: SizingBalances) -> dict[str, Any]:
        """How a fund's dollars were arrived at, for the commitments table: weight × (own_year_usd + other_years_usd)."""
        entitlement = self.entitlements[fund_name]
        own_year_usd, other_years_usd = self._budgets_usd(fund_name, balances)
        expected_value = (float("nan") if self.expected_return is None
                          else self.expected_value(balances.date, balances.first_commitment_date))
        own_year_rate = (float(self.rates.at[entitlement.policy_year, entitlement.fund_type])
                         if entitlement.policy_year in entitlement.draws else float("nan"))
        return {
            "policy_year": entitlement.policy_year, "own_year_rate": own_year_rate,
            "expected_value": expected_value, "weight": entitlement.weight,
            "own_year_usd": own_year_usd, "other_years_usd": other_years_usd,
            "drawn_years": format_draw_plan(entitlement.draws),
        }
