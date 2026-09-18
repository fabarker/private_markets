"""Data layer: where the engine's inputs come from.

Two workbook layouts are supported behind one seam:

    # the five-sheet portfolio workbook (Liquid, FX, Flows, Commitments, Spec), one profile at a time
    from pmsim.data import load_profile_workbook
    result = load_profile_workbook("portfolio.xlsx", "USD", "Conservative", initial_value=1_000_000).run()

    # one sheet per normalized table (fund_spec, fund_market_data, market_data, commitment_rates)
    from pmsim.data import SimulationSpec, run_tables_workbook
    result = run_tables_workbook("tables.xlsx", SimulationSpec("GBP", liquid_series="liquid_gbp", fx_series="gbp_per_usd"))

A ``DataRepository`` hands over four normalized tables (``tables.py``); ``ExcelRepository``,
``WorkbookRepository`` and ``FrameRepository`` all implement it — the last takes DataFrames
and is the shape a database adapter will take. ``Orchestrator`` joins a repository with a
``SimulationSpec`` into ``Fund`` and ``Portfolio`` objects and runs the ``Simulator``.
"""
from .orchestrator import (
    Orchestrator,
    SimulationSpec,
    build_funds,
    build_portfolio,
    load_tables_workbook,
    returns_to_levels,
    run_tables_workbook,
)
from .repository import DataRepository, ExcelRepository, FrameRepository, SheetNames
from .tables import (
    normalize_commitment_rates,
    normalize_fund_market_data,
    normalize_fund_specs,
    normalize_market_data,
)
from .workbook import (
    SheetLayout,
    WorkbookRepository,
    calendar_rates_for_profile,
    load_profile_workbook,
    run_profile_workbook,
)

__all__ = [
    "SimulationSpec", "Orchestrator", "build_funds", "build_portfolio", "returns_to_levels",
    "load_tables_workbook", "run_tables_workbook",
    "DataRepository", "ExcelRepository", "FrameRepository", "SheetNames",
    "WorkbookRepository", "SheetLayout", "calendar_rates_for_profile", "load_profile_workbook", "run_profile_workbook",
    "normalize_fund_specs", "normalize_fund_market_data", "normalize_market_data", "normalize_commitment_rates",
]
