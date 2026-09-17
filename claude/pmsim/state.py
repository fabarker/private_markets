"""Mutable state during a run: the cash pot in base currency, and the book of dollar commitments."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .inputs import Fund
from .timeline import AlignedFundHistory


@dataclass
class LiquidAccount:
    """The liquid pot, in base currency. The only balance that changes during a run."""

    balance: float
    return_factors: np.ndarray  # level[t] / level[t-1]; return_factors[0] == 1

    def apply_return(self, t: int) -> float:
        """Apply period ``t``'s return and report the P&L."""
        pnl = self.balance * (float(self.return_factors[t]) - 1.0)
        self.balance += pnl
        return pnl

    def deposit(self, amount: float) -> None:
        self.balance += amount

    def withdraw(self, amount: float) -> float:
        """Take ``amount`` out and return how much of it the balance could not cover."""
        shortfall = max(0.0, amount - self.balance)
        self.balance -= amount
        return shortfall


@dataclass(frozen=True, eq=False)
class Commitment:
    """A fixed dollar commitment to a fund, made at its closing. Every figure here is USD."""

    fund: Fund
    path: AlignedFundHistory
    usd: float

    @property
    def closing_period(self) -> int:
        return self.path.closing_period

    def calls_in_period(self, t: int) -> float:
        return self.usd * float(self.path.unit_calls[t])

    def distributions_in_period(self, t: int) -> float:
        return self.usd * float(self.path.unit_distributions[t])

    def nav_at(self, t: int) -> float:
        return self.usd * float(self.path.unit_nav[t]) if t >= 0 else 0.0


@dataclass
class CommitmentBook:
    """The book of live commitments. Sums are exact (``math.fsum``), so fund order cannot matter."""

    commitments: list[Commitment] = field(default_factory=list)

    def add(self, commitment: Commitment) -> None:
        self.commitments.append(commitment)

    def calls_in_period(self, t: int) -> float:
        return math.fsum(c.calls_in_period(t) for c in self.commitments)

    def distributions_in_period(self, t: int) -> float:
        return math.fsum(c.distributions_in_period(t) for c in self.commitments)

    def nav_at(self, t: int) -> float:
        return math.fsum(c.nav_at(t) for c in self.commitments)

    def commitments_closing_in(self, t: int) -> list[Commitment]:
        return [c for c in self.commitments if c.closing_period == t]

    def __len__(self) -> int:
        return len(self.commitments)
