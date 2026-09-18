"""pmsim: a liquid portfolio in base currency funding USD private-market commitments.

    from pmsim import Fund, Portfolio, Simulator
    result = Simulator(portfolio, funds).run()

Inputs are split by who holds the data (Fund, Portfolio). A Timeline built from the
liquid index turns every dated input into per-period arrays once. The Simulator owns
the one loop; a CommitmentPolicy decides how much to commit when funds close.
"""
from .inputs import PRIVATE_CURRENCY, Fund, Portfolio
from .policy import AnnualRatePolicy, CommitmentPolicy, Entitlement, SizingBalances
from .simulator import Shortfall, SimulationResult, Simulator
from .state import Commitment, LiquidAccount, CommitmentBook
from .timeline import AlignedFundHistory, Timeline

__all__ = [
    "PRIVATE_CURRENCY", "Fund", "Portfolio",
    "Timeline", "AlignedFundHistory",
    "LiquidAccount", "Commitment", "CommitmentBook",
    "SizingBalances", "CommitmentPolicy", "AnnualRatePolicy", "Entitlement",
    "Simulator", "SimulationResult", "Shortfall",
]
