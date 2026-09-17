"""Mutable state during a run: the cash pot in base currency, and the book of dollar commitments."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .inputs import Fund
from .timeline import FundPath


@dataclass
class LiquidAccount:
    """The liquid pot, in base currency. The only balance that changes during a run."""

    balance: float
    factors: np.ndarray  # level[t] / level[t-1]; factors[0] == 1

    def grow(self, t: int) -> float:
        """Apply period ``t``'s return and report the P&L."""
        pnl = self.balance * (float(self.factors[t]) - 1.0)
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
    path: FundPath
    usd: float

    @property
    def start(self) -> int:
        return self.path.closing_index

    def calls(self, t: int) -> float:
        return self.usd * float(self.path.calls[t])

    def distributions(self, t: int) -> float:
        return self.usd * float(self.path.distributions[t])

    def nav(self, t: int) -> float:
        return self.usd * float(self.path.nav[t]) if t >= 0 else 0.0


@dataclass
class PrivateBook:
    """The book of live commitments. Sums are exact (``math.fsum``), so fund order cannot matter."""

    commitments: list[Commitment] = field(default_factory=list)

    def add(self, commitment: Commitment) -> None:
        self.commitments.append(commitment)

    def calls(self, t: int) -> float:
        return math.fsum(c.calls(t) for c in self.commitments)

    def distributions(self, t: int) -> float:
        return math.fsum(c.distributions(t) for c in self.commitments)

    def nav(self, t: int) -> float:
        return math.fsum(c.nav(t) for c in self.commitments)

    def closed_at(self, t: int) -> list[Commitment]:
        return [c for c in self.commitments if c.start == t]

    def __len__(self) -> int:
        return len(self.commitments)
