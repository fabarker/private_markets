"""Data layer: where the engine's inputs come from.

    from pmsim.data import load_profile_workbook
    result = load_profile_workbook("portfolio.xlsx", "USD", "Conservative", initial_value=1_000_000).run()

A ``DataRepository`` hands over four normalized tables (``tables.py``). ``WorkbookRepository``
reads them from the five-sheet portfolio workbook (Liquid, FX, Flows, Commitments, Spec),
one profile at a time; ``FrameRepository`` takes them as DataFrames and is the shape a
database adapter will take. ``Orchestrator`` joins a repository with a ``SimulationSpec``
into ``Fund`` and ``Portfolio`` objects and runs the ``Simulator``.

Modules, in dependency order: ``tables`` and ``spec`` (leaves) → ``repository`` → ``orchestrator``.
"""
from .orchestrator import (
    Orchestrator,
    build_funds,
    build_portfolio,
    infer_inception_date,
    load_profile_workbook,
    returns_to_levels,
    run_profile_workbook,
)
from .repository import (
    DataRepository,
    FrameRepository,
    SheetLayout,
    WorkbookRepository,
    calendar_rates_for_profile,
)
from .spec import SimulationSpec
from .tables import (
    normalize_commitment_rates,
    normalize_fund_market_data,
    normalize_fund_specs,
    normalize_market_data,
)

__all__ = [
    "SimulationSpec", "Orchestrator", "build_funds", "build_portfolio", "returns_to_levels", "infer_inception_date",
    "load_profile_workbook", "run_profile_workbook",
    "DataRepository", "FrameRepository", "WorkbookRepository", "SheetLayout", "calendar_rates_for_profile",
    "normalize_fund_specs", "normalize_fund_market_data", "normalize_market_data", "normalize_commitment_rates",
]
