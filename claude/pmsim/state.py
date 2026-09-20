"""Mutable state during a run: the cash pot in base currency, and the book of dollar commitments."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .inputs import Fund
from .timeline import AlignedFundHistory


@dataclass
class LiquidAccount:
    """The liquid pot, in base currency. The only balance that changes during a run.

    ``returns[t]`` is period ``t``'s return as a percent change, ``level[t] / level[t-1] − 1``,
    and 0 in the first period.
    """

    balance: float
    returns: np.ndarray

    def apply_return(self, t: int) -> float:
        """Apply period ``t``'s return and report the P&L."""
        period_return = float(self.returns[t])
        pnl = self.balance * period_return

        self.balance += pnl
        return pnl

    def deposit(self, amount: float) -> None:
        """Put ``amount`` into the account."""
        self.balance += amount

    def withdraw(self, amount: float) -> float:
        """Take ``amount`` out and return how much of it the balance could not cover.

        The balance is allowed to go negative: the caller decides what a shortfall means.
        """
        shortfall = max(0.0, amount - self.balance)

        self.balance -= amount
        return shortfall


@dataclass(frozen=True, eq=False)
class Commitment:
    """A fixed dollar commitment to a fund, made at its closing. Every figure here is USD.

    ``path`` is the fund's unit history, per 1 committed, on the run's observation dates.
    Scaling it by ``usd`` gives the fund's dollar calls, distributions and NAV.
    """

    fund: Fund
    path: AlignedFundHistory
    usd: float

    @property
    def closing_period(self) -> int:
        return self.path.closing_period

    def calls_in_period(self, t: int) -> float:
        unit_calls = float(self.path.unit_calls[t])
        return self.usd * unit_calls

    def distributions_in_period(self, t: int) -> float:
        unit_distributions = float(self.path.unit_distributions[t])
        return self.usd * unit_distributions

    def nav_at(self, t: int) -> float:
        """The fund's dollar NAV at observation ``t``; 0 before the first observation."""
        if t < 0:
            return 0.0

        unit_nav = float(self.path.unit_nav[t])
        return self.usd * unit_nav


@dataclass
class CommitmentBook:
    """The book of live commitments.

    Sums are exact (``math.fsum``), so the order the funds were added in cannot matter.
    """

    commitments: list[Commitment] = field(default_factory=list)

    def add(self, commitment: Commitment) -> None:
        self.commitments.append(commitment)

    def calls_in_period(self, t: int) -> float:
        """Every fund's dollar calls in period ``t``, added up."""
        return math.fsum(commitment.calls_in_period(t) for commitment in self.commitments)

    def distributions_in_period(self, t: int) -> float:
        """Every fund's dollar distributions in period ``t``, added up."""
        return math.fsum(commitment.distributions_in_period(t) for commitment in self.commitments)

    def nav_at(self, t: int) -> float:
        """Every fund's dollar NAV at observation ``t``, added up."""
        return math.fsum(commitment.nav_at(t) for commitment in self.commitments)

    def commitments_closing_in(self, t: int) -> list[Commitment]:
        """The commitments made in period ``t``."""
        return [commitment for commitment in self.commitments if commitment.closing_period == t]

    def __len__(self) -> int:
        return len(self.commitments)
