"""pmsim: a liquid portfolio in base currency funding USD private-market commitments.

    from pmsim import Fund, Portfolio, Simulator
    result = Simulator(portfolio, funds).run()

Inputs are split by who holds the data (Fund, Portfolio). A Timeline built from the
liquid index turns every dated input into per-period arrays once. The Simulator owns
the one loop; a CommitmentPolicy decides how much to commit when funds close.
"""
from .benchmark import annualised_irr, compare_with_liquid_only, public_market_equivalent
from .inputs import PRIVATE_CURRENCY, Fund, Portfolio
from .policy import (
    AnnualRatePolicy,
    CommitmentPolicy,
    DrawnYear,
    Entitlement,
    SizingBalances,
    YearEndBalance,
    commitment_rounding_unit,
    round_like_excel,
)
from .simulator import Shortfall, SimulationResult, Simulator
from .state import Commitment, CommitmentBook, LiquidAccount
from .timeline import AlignedFundHistory, Timeline

__all__ = [
    # the inputs
    "PRIVATE_CURRENCY",
    "Fund",
    "Portfolio",

    # dates → periods
    "Timeline",
    "AlignedFundHistory",

    # the state of a run
    "LiquidAccount",
    "Commitment",
    "CommitmentBook",

    # commitment sizing
    "SizingBalances",
    "YearEndBalance",
    "CommitmentPolicy",
    "AnnualRatePolicy",
    "Entitlement",
    "DrawnYear",
    "commitment_rounding_unit",
    "round_like_excel",

    # the loop and what it returns
    "Simulator",
    "SimulationResult",
    "Shortfall",

    # benchmarks read off a result
    "compare_with_liquid_only",
    "public_market_equivalent",
    "annualised_irr",
]
