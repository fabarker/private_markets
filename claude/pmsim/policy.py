"""Commitment sizing, in US dollars, on the liquid-only value.

Two rules fix what a commitment is sized on.

It is sized in **US dollars**: private funds are committed to in dollars, whatever the
portfolio's base currency, so the engine converts before it asks and turns the answer into
base currency only for reporting.

It is sized on the **liquid-only value**: the initial value compounded by the liquid returns,
with no capital call or distribution in it. Private assets never interfere, so every
commitment follows from the liquid returns, the initial value, the exchange rates and the
schedule alone.

It can be **rounded the way the spreadsheet rounds it**: each year's dollar commitment goes to
the nearest multiple of a rounding unit, halves away from zero, which is Excel's ROUND. The
unit is one ten-thousandth of the starting value, so a 100,000,000 portfolio rounds to the
nearest 10,000 — ``ROUND(value, -4)`` — and any other starting value to the same relative
precision.

The engine asks a policy one question: given the funds closing at this observation and a
snapshot of the balances, how many dollars to commit to each. The policy reads and never
moves cash, and keeps no state. ``AnnualRatePolicy`` is the default: the pacing schedule
gives a dollar budget per year and fund type, and each fund collects the budgets of the
years it **draws**. Which years those are is its draw plan — by default its own closing
year, plus, with carry-forward, the years its type passed without a closing; or, stated
explicitly, any set of years at any multiple, including years after the closing.

Every drawn year is priced on the liquid-only value at **that year's own year end** — always,
including a year that lies after the fund's closing. Such a year is priced with hindsight:
the run looks forward to what the liquid portfolio will be worth at that year's end and uses
it. That is a deliberate choice, made so that a fund collecting years 1 to 4 collects exactly
the four dollar commitments the schedule computes for those years; it means a commitment
made in 2011 can depend on the portfolio's value in 2013.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from numbers import Real
from typing import Any, Iterable, Mapping, Protocol, Sequence

import pandas as pd

from .dates import years_between
from .inputs import Fund, coerce_rate_table

WEIGHT_TOLERANCE = 1e-9

# Excel's ROUND(value, -4) on a 100,000,000 portfolio rounds to units of 10,000: the
# starting value divided by this number.
ROUNDING_UNITS_PER_STARTING_VALUE = 10_000

FundGroups = dict[tuple[int, str], list[Fund]]  # funds by (closing year, fund type)
DrawPlans = Mapping[str, Mapping[int, float]]  # fund name → calendar year → multiplier


# ------------------------------------------------------------ what a policy is shown
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
    distributions, after this period's return and the existing funds' distributions.
    ``private_nav_usd`` is the opening private NAV; it is natively USD, so nothing is
    translated. Those two are here for policies that want them, and ``AnnualRatePolicy``
    uses neither.

    ``year_ends`` holds completed calendar years only: what the investor could have known at
    this observation. Carry-forward reads nothing else.

    ``future_year_ends`` holds the rest — the current calendar year's end and every later
    one's — which the investor could *not* have known. It is here for one purpose: a draw plan
    may name a year after the fund's closing, and such a year is priced on its own year-end
    value, with hindsight. Nothing else may read it.

    ``first_commitment_date`` is the observation at which the run's first fund is committed.
    """

    t: int
    date: date
    liquid_only_usd: float
    liquid_account_usd: float
    private_nav_usd: float
    year_ends: Mapping[int, YearEndBalance] = field(default_factory=dict)
    first_commitment_date: date | None = None
    future_year_ends: Mapping[int, YearEndBalance] = field(default_factory=dict)

    @property
    def total_usd(self) -> float:
        return self.liquid_account_usd + self.private_nav_usd

    def year_end_on_or_before(self, year: int) -> YearEndBalance | None:
        """The last year end on or before ``year``.

        A year with no observation of its own uses the one before it. None when ``year`` is
        before the run began.
        """
        years_known_by_then = [known for known in self.year_ends if known <= year]

        if not years_known_by_then:
            return None

        latest_year = max(years_known_by_then)
        return self.year_ends[latest_year]

    def year_end_seen_with_hindsight(self, year: int) -> YearEndBalance | None:
        """The year end of ``year`` itself, whether or not it has happened yet.

        None when the data has no such year. Looks among the years still to come first, then
        among the completed ones. Unlike ``year_end_on_or_before`` it never falls back to a
        different year: pricing a year on some other year's value would be wrong, and doing so
        silently would be worse.
        """
        if year in self.future_year_ends:
            return self.future_year_ends[year]

        if year in self.year_ends:
            return self.year_ends[year]

        return None


class CommitmentPolicy(Protocol):
    def size_commitments(self, cohort: Sequence[Fund], balances: SizingBalances) -> Mapping[str, float]:
        """US-dollar commitment for each fund in ``cohort``, keyed by fund name."""


# ------------------------------------------------------------- what a policy works out
@dataclass(frozen=True)
class DrawnYear:
    """One schedule year a fund draws, and the dollars that year's commitment came to.

    ``sizing_date`` is the one date the year is priced on: the pacing model's expected value
    is taken on it, and so is the liquid-only value. It is the year's own year end — except
    for the fund's own closing year, which is priced on the closing observation.
    ``looks_ahead`` is True when that date is later than the closing: the year lies after the
    fund's closing and was priced with hindsight, on a value nobody could have known on the
    day the commitment was made.

    ``year_budget_unrounded_usd`` is the year's dollar commitment as computed,
    ``rate / expected_value × liquid_only_usd``. ``year_budget_usd`` is the same amount
    rounded to the policy's rounding unit, and equal to it when nothing is rounded.
    ``commitment_usd`` is ``multiplier × year_budget_usd``.
    """

    year: int
    multiplier: float
    rate: float
    sizing_date: date
    expected_value: float
    liquid_only_usd: float
    looks_ahead: bool
    year_budget_unrounded_usd: float
    year_budget_usd: float
    commitment_usd: float


@dataclass(frozen=True)
class Entitlement:
    """The schedule years a fund draws, each with a multiplier, and its weight of their total.

    ``policy_year`` is the fund's closing year. ``draws`` maps calendar year → multiplier, in
    ascending year order.
    """

    policy_year: int
    fund_type: str
    weight: float
    draws: Mapping[int, float] = field(default_factory=dict)


# ------------------------------------------------------------------------ validation
def _is_a_finite_number(value: Any) -> bool:
    """True for a real, finite number. A bool is not a number here, though Python counts it as one."""
    if isinstance(value, bool):
        return False

    if not isinstance(value, Real):
        return False

    return math.isfinite(value)


def validate_expected_return(value: Any, *, label: str = "expected_return") -> float:
    """A yearly expected return as a decimal: 0.05 is 5%.

    At or above 1 it is taken for a percentage typed as a number, and rejected.
    """
    if not _is_a_finite_number(value):
        raise ValueError(f"{label} must be a number such as 0.05 for 5% a year, got {value!r}")

    if not -1.0 < value < 1.0:
        raise ValueError(f"{label} must be a decimal between -1 and 1 (write 5% as 0.05), got {value!r}")

    return float(value)


def validate_rounding_unit(value: Any, *, label: str = "rounding_unit_usd") -> float:
    """A rounding unit in US dollars: a finite number above zero, such as 10_000."""
    if not _is_a_finite_number(value) or value <= 0:
        raise ValueError(f"{label} must be a positive number of dollars such as 10_000, got {value!r}")

    return float(value)


# -------------------------------------------------------------------------- rounding
def commitment_rounding_unit(starting_value: float) -> float:
    """The rounding unit that gives every starting value the precision ROUND(value, -4) gives 100,000,000.

    100,000,000 → 10,000 · 1,000,000 → 100 · 100 → 0.01. Proportional, so a run started
    from any value is the 100,000,000 run scaled, rounding included.
    """
    if not _is_a_finite_number(starting_value) or starting_value <= 0:
        raise ValueError(f"starting_value must be a positive number, got {starting_value!r}")

    return float(starting_value) / ROUNDING_UNITS_PER_STARTING_VALUE


def round_like_excel(value: float, unit: float) -> float:
    """``value`` to the nearest multiple of ``unit``, halves going away from zero: Excel's ROUND.

    ``ROUND(value, -4)`` is ``round_like_excel(value, 10_000)`` and ``ROUND(value, 2)`` is
    ``round_like_excel(value, 0.01)``. Python's own ``round`` sends a half to the even
    neighbour (25,000 → 20,000); Excel sends it away from zero (25,000 → 30,000), and so
    does this. The division is done in decimal arithmetic on the digits the numbers print
    with, so 2.675 rounds to 2.68 as it does in Excel, instead of being caught by its binary
    form lying a hair below the half.
    """
    value = float(value)
    if not math.isfinite(value):
        return value

    # Work in decimal, on the digits each number prints with.
    decimal_value = Decimal(repr(value))
    decimal_unit = Decimal(repr(float(unit)))

    # How many whole units: ROUND_HALF_UP sends a half away from zero, on both sides of it.
    units = decimal_value / decimal_unit
    whole_units = units.quantize(Decimal(1), rounding=ROUND_HALF_UP)

    return float(whole_units * decimal_unit)


def format_draw_plan(draws: Mapping[int, float]) -> str:
    """``2010, 2011, 2012`` — a multiplier other than 1 shown against its year, as ``2021x3``."""
    terms = []
    for year, multiplier in draws.items():
        if multiplier != 1:
            terms.append(f"{year}x{multiplier:g}")
        else:
            terms.append(str(year))

    return ", ".join(terms)


# ------------------------------------------------- settling who draws what, before a run
def _weights_by_fund(groups: FundGroups, weights: Mapping[str, float]) -> dict[str, float]:
    """Each fund's share of its group's budget.

    A group is the funds of one type closing in one year. Its weights are given for all of
    its funds or for none (an equal split), and sum to 1.
    """
    shares: dict[str, float] = {}

    for (year, fund_type), group in groups.items():
        funds_given_a_weight = [fund for fund in group if fund.name in weights]

        # No weights for this group: split it equally.
        if not funds_given_a_weight:
            for fund in group:
                shares[fund.name] = 1.0 / len(group)
            continue

        # Some but not all: refuse to guess the rest.
        if len(funds_given_a_weight) != len(group):
            raise ValueError(
                f"weights for {fund_type} funds closing in {year} must be given for all of them or none"
            )

        for fund in group:
            weight = float(weights[fund.name])
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"weight for {fund.name!r} must be a finite non-negative number")

            shares[fund.name] = weight

        total = math.fsum(shares[fund.name] for fund in group)
        if abs(total - 1.0) > WEIGHT_TOLERANCE:
            raise ValueError(f"weights for {fund_type} funds closing in {year} sum to {total:.6g}, not 1")

    return shares


def _checked_draw_plans(plans: DrawPlans, groups: FundGroups,
                        rates: pd.DataFrame) -> dict[str, dict[int, float]]:
    """The plans, years put in ascending order, checked against the fund list and the rate table."""
    funds_by_name = {}
    for group in groups.values():
        for fund in group:
            funds_by_name[fund.name] = fund

    unknown_names = set(plans) - set(funds_by_name)
    if unknown_names:
        raise ValueError(f"draws name funds that are not in the fund list: {sorted(unknown_names)}")

    # Each plan: at least one year, every year in the rate table, every multiplier usable.
    checked: dict[str, dict[int, float]] = {}

    for name, plan in plans.items():
        if not plan:
            raise ValueError(f"draws for {name!r} is empty: leave the fund out to keep its default years")

        multiplier_by_year: dict[int, float] = {}

        for year, multiplier in plan.items():
            year = int(year)
            multiplier = float(multiplier)

            if year not in rates.index:
                raise ValueError(
                    f"draws for {name!r} name year {year}, which commitment_rates has no row for"
                )

            if not math.isfinite(multiplier) or multiplier < 0:
                raise ValueError(
                    f"multiplier for year {year} of {name!r} must be a finite non-negative number"
                )

            multiplier_by_year[year] = multiplier

        checked[name] = dict(sorted(multiplier_by_year.items()))

    # A schedule year is one closing's to spend. Funds closing together share it by weight.
    closing_year_that_claimed: dict[tuple[str, int], int] = {}

    for name, plan in checked.items():
        fund = funds_by_name[name]

        for year in plan:
            first_claim = closing_year_that_claimed.setdefault((fund.fund_type, year), fund.closing_year)

            if first_claim != fund.closing_year:
                raise ValueError(
                    f"{fund.fund_type} year {year} is drawn by funds closing in both {first_claim} and "
                    f"{fund.closing_year}; a schedule year belongs to one closing"
                )

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

        # This type's funds, by the year they close in.
        funds_closing_in = {}
        for (year, type_of_group), group in groups.items():
            if type_of_group == fund_type:
                funds_closing_in[year] = group

        # Carry-forward applies to this type only if none of its funds has a plan.
        type_has_a_plan = False
        for group in funds_closing_in.values():
            for fund in group:
                if fund.name in plans:
                    type_has_a_plan = True

        carries_forward = carry_forward and not type_has_a_plan

        # Walk the years. A year with no closing waits; a year with one hands out what waited.
        years_waiting: dict[int, float] = {}

        for year in rates.index:
            year = int(year)
            cohort = funds_closing_in.get(year, [])

            for fund in cohort:
                plan = plans.get(fund.name)

                if plan:
                    draws = dict(plan)
                else:
                    own_year_and_waiting = {year: 1.0, **years_waiting}
                    draws = dict(sorted(own_year_and_waiting.items()))

                entitlements[fund.name] = Entitlement(year, fund_type, weights[fund.name], draws)

            if cohort:
                years_waiting = {}
            elif carries_forward and float(rates.at[year, fund_type]) > 0:
                years_waiting[year] = 1.0

    return entitlements


# ------------------------------------------------------------------------- the policy
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
    Naming any fund of a type in ``draws`` switches carry-forward off for that type.

    **Every year is priced at its own year end, even one still to come.** A drawn year's
    dollars are ``rate / expected value × liquid-only value``, all three taken at that year's
    own year end. For a year after the fund's closing that means looking ahead: the liquid
    value the portfolio *will* have at that year's end is used, although nobody could have
    known it on the day the commitment was made. This is deliberate. It makes a fund that
    draws years 1 to 4 collect exactly the four dollar commitments the schedule computes for
    those years. The price is hindsight in the backtest, and ``DrawnYear.looks_ahead`` marks
    every year it touches. The fund's own closing year is priced on the closing observation,
    which for a 31 December closing is the year end. A plan naming a year the liquid series
    does not reach is an error: there is no value to look ahead to.

    **Rounding.** With ``rounding_unit_usd`` each drawn year's dollar commitment is rounded to
    the nearest multiple of the unit, halves away from zero, exactly as Excel's ROUND does,
    before its multiplier and the fund's weight are applied: a fund that draws four years
    collects four rounded amounts, and one that draws ``12x3`` three times one rounded amount.
    ``commitment_rounding_unit(starting_value)`` gives the unit that matches
    ``ROUND(value, -4)`` on a 100,000,000 portfolio at any starting value.

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
        rounding_unit_usd: float | None = None,
    ) -> None:
        # The settings.
        self.rates = coerce_rate_table(rates)
        self.carry_forward = bool(carry_forward)

        self.expected_return = None
        if expected_return is not None:
            self.expected_return = validate_expected_return(expected_return)

        self.rounding_unit_usd = None
        if rounding_unit_usd is not None:
            self.rounding_unit_usd = validate_rounding_unit(rounding_unit_usd)

        # The funds, and the weights given for them.
        funds = list(funds)
        names = [fund.name for fund in funds]
        if len(set(names)) != len(names):
            raise ValueError("fund names must be unique")

        weights = dict(weights or {})
        unknown_names = set(weights) - set(names)
        if unknown_names:
            raise ValueError(f"weights name funds that are not in the fund list: {sorted(unknown_names)}")

        # The rate table must have a column for every fund type, and a row for every year needed.
        fund_types = {fund.fund_type for fund in funds}
        missing_types = fund_types - set(self.rates.columns)
        if missing_types:
            raise ValueError(f"commitment_rates has no column for fund type(s) {sorted(missing_types)}")

        if len(self.rates.columns):
            closing_years = {fund.closing_year for fund in funds}
            required_years = set(years or ()) | closing_years
            missing_years = required_years - set(self.rates.index)
            if missing_years:
                raise ValueError(f"commitment_rates has no row for year(s) {sorted(missing_years)}")

        # Group the funds by (closing year, fund type), then settle who draws what.
        groups: FundGroups = {}
        for fund in funds:
            group_key = (fund.closing_year, fund.fund_type)
            groups.setdefault(group_key, []).append(fund)

        self.weights = _weights_by_fund(groups, weights)
        self.draws = _checked_draw_plans(dict(draws or {}), groups, self.rates)
        self.entitlements = _entitlements(self.rates, groups, self.weights, self.carry_forward, self.draws)

    def expected_value(self, on: date, first_commitment_date: date | None) -> float:
        """The pacing model's liquid value on ``on``.

        1 on the first commitment date, growing at the expected return from there. Always 1
        when the policy has no expected return: the rates are shares already.
        """
        if self.expected_return is None:
            return 1.0

        if first_commitment_date is None:
            raise ValueError(
                "expected_return needs SizingBalances.first_commitment_date: "
                "the pacing model's value is 1 on that date"
            )

        years_since_first_commitment = years_between(first_commitment_date, on)
        return (1.0 + self.expected_return) ** years_since_first_commitment

    def unclaimed_schedule_years(self) -> dict[str, list[int]]:
        """The schedule's unspent budget.

        Per fund type, the years that have a rate above zero and that no fund draws.
        """
        # Every year drawn by some fund, by fund type.
        years_drawn: dict[str, set[int]] = {}
        for entitlement in self.entitlements.values():
            years_drawn.setdefault(entitlement.fund_type, set()).update(entitlement.draws)

        # Every year that has a budget and is not among them.
        unclaimed: dict[str, list[int]] = {}
        for fund_type in self.rates.columns:
            years_drawn_by_type = years_drawn.get(fund_type, set())

            unclaimed[fund_type] = []
            for year in self.rates.index:
                has_a_budget = float(self.rates.at[year, fund_type]) > 0

                if has_a_budget and int(year) not in years_drawn_by_type:
                    unclaimed[fund_type].append(int(year))

        return unclaimed

    def drawn_years(self, fund_name: str, balances: SizingBalances) -> list[DrawnYear]:
        """Every schedule year the fund draws, with the dollars each one's commitment came to.

        Every year is priced on one date, its sizing date: the pacing model's expected value
        and the liquid-only value are both taken there. For a year before the closing that is
        the year's own year end, as carry-forward has always done. For the fund's own year it
        is the closing observation. For a year after the closing it is, again, that year's own
        year end — seen with hindsight, because the run has not reached it.
        """
        entitlement = self.entitlements[fund_name]
        first_commitment_date = balances.first_commitment_date

        drawn: list[DrawnYear] = []

        for year, multiplier in entitlement.draws.items():
            rate = float(self.rates.at[year, entitlement.fund_type])

            # 1 ── The date this year is priced on, and the liquid-only value on that date.
            if year < entitlement.policy_year:
                # A year already over: its own year end.
                year_end = balances.year_end_on_or_before(year)
                if year_end is None:
                    continue  # a year before the run began had no portfolio to size on

                sizing_date = year_end.date
                liquid_only_usd = year_end.liquid_only_usd

            elif year == entitlement.policy_year:
                # The fund's own year: the closing observation.
                sizing_date = balances.date
                liquid_only_usd = balances.liquid_only_usd

            else:
                # A year still to come: its own year end, seen with hindsight. The run looks
                # forward to what the liquid portfolio will be worth then, and uses it.
                year_end = balances.year_end_seen_with_hindsight(year)
                if year_end is None:
                    raise ValueError(
                        f"{fund_name!r} draws {year}, but the liquid series has no observation in {year} "
                        f"to take that year's value from"
                    )

                sizing_date = year_end.date
                liquid_only_usd = year_end.liquid_only_usd

            looks_ahead = sizing_date > balances.date

            # 2 ── The year's own dollar commitment: its share of the liquid-only value,
            #      with the share and the value both taken on the sizing date.
            expected_value = self.expected_value(sizing_date, first_commitment_date)
            year_budget_unrounded_usd = rate / expected_value * liquid_only_usd

            # 3 ── Rounded the way the spreadsheet rounds it, when a rounding unit is set.
            year_budget_usd = year_budget_unrounded_usd
            if self.rounding_unit_usd is not None:
                year_budget_usd = round_like_excel(year_budget_unrounded_usd, self.rounding_unit_usd)

            # 4 ── Drawn as many times as the plan says.
            commitment_usd = multiplier * year_budget_usd

            drawn.append(DrawnYear(
                year=year,
                multiplier=multiplier,
                rate=rate,
                sizing_date=sizing_date,
                expected_value=expected_value,
                liquid_only_usd=liquid_only_usd,
                looks_ahead=looks_ahead,
                year_budget_unrounded_usd=year_budget_unrounded_usd,
                year_budget_usd=year_budget_usd,
                commitment_usd=commitment_usd,
            ))

        return drawn

    def _budgets_usd(self, fund_name: str, balances: SizingBalances) -> tuple[float, float]:
        """The dollars from the fund's own closing year, and the dollars from every other year it draws."""
        drawn = self.drawn_years(fund_name, balances)
        own_year = self.entitlements[fund_name].policy_year

        own_year_usd = math.fsum(d.commitment_usd for d in drawn if d.year == own_year)
        other_years_usd = math.fsum(d.commitment_usd for d in drawn if d.year != own_year)

        return own_year_usd, other_years_usd

    def size_commitments(self, cohort: Sequence[Fund], balances: SizingBalances) -> Mapping[str, float]:
        """The dollars to commit to each fund closing now: its weight of the years it draws."""
        sized = {}

        for fund in cohort:
            own_year_usd, other_years_usd = self._budgets_usd(fund.name, balances)
            weight = self.entitlements[fund.name].weight

            sized[fund.name] = weight * (own_year_usd + other_years_usd)

        return sized

    def explain_commitment(self, fund_name: str, balances: SizingBalances) -> dict[str, Any]:
        """How a fund's dollars were arrived at, for the commitments table.

        ``commitment_usd = weight × (own_year_usd + other_years_usd)``.
        """
        entitlement = self.entitlements[fund_name]
        own_year_usd, other_years_usd = self._budgets_usd(fund_name, balances)

        # The pacing model's value today; not a number when the policy has no expected return.
        if self.expected_return is None:
            expected_value = float("nan")
        else:
            expected_value = self.expected_value(balances.date, balances.first_commitment_date)

        # The fund's own year's rate; not a number when its plan leaves its own year out.
        if entitlement.policy_year in entitlement.draws:
            own_year_rate = float(self.rates.at[entitlement.policy_year, entitlement.fund_type])
        else:
            own_year_rate = float("nan")

        return {
            "policy_year": entitlement.policy_year,
            "own_year_rate": own_year_rate,
            "expected_value": expected_value,
            "weight": entitlement.weight,
            "own_year_usd": own_year_usd,
            "other_years_usd": other_years_usd,
            "drawn_years": format_draw_plan(entitlement.draws),
        }
