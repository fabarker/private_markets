"""Commitment sizing.

The engine asks a policy one question: given the funds closing at this observation and a
snapshot of the balances, how much base currency to commit to each. The policy reads
balances and never moves cash. ``AnnualRatePolicy`` is the default: the portfolio's rate
for the fund's closing year and type, split across that year's funds of that type by
weight, optionally pooling rates from years in which no fund of the type closed.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Iterable, Mapping, Protocol, Sequence

import pandas as pd

from .inputs import Fund, coerce_rate_table

WEIGHT_TOLERANCE = 1e-9
FundGroups = dict[tuple[int, str], list[Fund]]  # funds by (closing year, fund type)


@dataclass(frozen=True)
class SizingBalances:
    """What a policy may look at when sizing commitments at observation ``t``, in base currency."""

    t: int
    date: date
    liquid: float  # after this period's return and existing funds' distributions
    private_nav: float  # opening private NAV translated at today's rate

    @property
    def total(self) -> float:
        return self.liquid + self.private_nav


class CommitmentPolicy(Protocol):
    def size_commitments(self, cohort: Sequence[Fund], balances: SizingBalances) -> Mapping[str, float]:
        """Base-currency commitment for each fund in ``cohort``, keyed by fund name."""


@dataclass(frozen=True)
class Entitlement:
    """How a fund's effective rate was arrived at; reported in the commitments table."""

    policy_year: int
    current_year_rate: float
    carried_rate: float
    pooled_rate: float
    weight: float
    effective_rate: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    """Walk each fund type's rates year by year; with carry-forward, a year with no closing pools into the next."""
    entitlements: dict[str, Entitlement] = {}
    for fund_type in rates.columns:
        carried = 0.0
        for year in rates.index:
            current = float(rates.at[year, fund_type])
            pool = current + carried
            cohort = groups.get((int(year), fund_type), [])
            for f in cohort:
                weight = weights[f.name]
                entitlements[f.name] = Entitlement(int(year), current, carried, pool, weight, pool * weight)
            carried = pool if (carry_forward and not cohort) else 0.0
    return entitlements


class AnnualRatePolicy:
    """``rate[closing year, fund type]`` × weight × sizing liquid balance.

    ``weights`` split a year's pooled rate among the funds of one type closing that year;
    they must be given for all funds of such a group or none (equal split), and sum to 1.
    With ``carry_forward`` a year in which no fund of a type closes adds its rate to the
    next year of that type that has one. ``years`` lists the calendar years the rate table
    must cover (the simulator passes the horizon); fund closing years are always required.
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

    def size_commitments(self, cohort: Sequence[Fund], balances: SizingBalances) -> Mapping[str, float]:
        return {f.name: self.entitlements[f.name].effective_rate * balances.liquid for f in cohort}

    def explain_rate(self, fund_name: str) -> dict[str, Any]:
        """The entitlement behind a fund's rate, for the commitments table."""
        return self.entitlements[fund_name].as_dict()
