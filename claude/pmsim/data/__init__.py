"""Data layer: where the engine's inputs come from.

    from pmsim.data import SimulationSpec, run_workbook
    result = run_workbook("portfolio.xlsx", SimulationSpec("GBP", liquid_series="liquid_gbp", fx_series="gbp_per_usd"))

A ``DataRepository`` hands over four normalized tables (``tables.py``); ``ExcelRepository``
reads them from a workbook and ``FrameRepository`` takes them as DataFrames — the shape a
database adapter will take. ``Orchestrator`` joins a repository with a ``SimulationSpec``
(what the data does not say: base currency, which series is which, sign convention,
policy settings) into ``Fund`` and ``Portfolio`` objects and runs the ``Simulator``.
"""
from .orchestrator import Orchestrator, SimulationSpec, build_funds, build_portfolio, load_workbook, run_workbook
from .repository import DataRepository, ExcelRepository, FrameRepository, SheetNames
from .tables import (
    normalize_commitment_rates,
    normalize_fund_market_data,
    normalize_fund_specs,
    normalize_market_data,
)

__all__ = [
    "SimulationSpec", "Orchestrator", "build_funds", "build_portfolio", "load_workbook", "run_workbook",
    "DataRepository", "ExcelRepository", "FrameRepository", "SheetNames",
    "normalize_fund_specs", "normalize_fund_market_data", "normalize_market_data", "normalize_commitment_rates",
]
