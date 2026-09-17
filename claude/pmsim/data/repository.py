"""Where the tables come from.

``DataRepository`` is the seam between the engine and its data: anything that can hand
over the normalized tables described in ``tables.py``. ``ExcelRepository`` reads them
from a workbook; ``FrameRepository`` takes them as DataFrames — for tests, notebooks, and
as the shape a database adapter will take. The orchestrator only ever talks to the protocol.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from .tables import (
    canonical_name,
    normalize_commitment_rates,
    normalize_fund_market_data,
    normalize_fund_specs,
    normalize_market_data,
)


class DataRepository(Protocol):
    def fund_specs(self) -> pd.DataFrame:
        """fund_name, fund_type, closing_date."""

    def fund_market_data(self) -> pd.DataFrame:
        """fund_name, kind, date, value, scale, unit."""

    def market_data(self) -> pd.DataFrame:
        """One column per series, indexed by date."""

    def commitment_rates(self) -> pd.DataFrame | None:
        """Calendar year × fund type, or None when the source has no rate table."""


class FrameRepository:
    """The tables handed over as DataFrames. Normalized once, at construction, so bad data fails early."""

    def __init__(self, fund_specs: Any, fund_market_data: Any, market_data: Any, commitment_rates: Any = None) -> None:
        self._fund_specs = normalize_fund_specs(fund_specs)
        self._fund_market_data = normalize_fund_market_data(fund_market_data)
        self._market_data = normalize_market_data(market_data)
        self._commitment_rates = None if commitment_rates is None else normalize_commitment_rates(commitment_rates)

    def fund_specs(self) -> pd.DataFrame:
        return self._fund_specs.copy()

    def fund_market_data(self) -> pd.DataFrame:
        return self._fund_market_data.copy()

    def market_data(self) -> pd.DataFrame:
        return self._market_data.copy()

    def commitment_rates(self) -> pd.DataFrame | None:
        return None if self._commitment_rates is None else self._commitment_rates.copy()


@dataclass(frozen=True)
class SheetNames:
    """Which sheet holds which table. Matched case-, space- and hyphen-insensitively."""

    fund_spec: str = "fund_spec"
    fund_market_data: str = "fund_market_data"
    market_data: str = "market_data"
    commitment_rates: str = "commitment_rates"  # optional sheet


class ExcelRepository:
    """The tables read from one workbook. Reads every sheet once; needs openpyxl for .xlsx."""

    def __init__(self, path: Any, sheets: SheetNames = SheetNames()) -> None:
        self.path = Path(path)
        self.sheets = sheets
        if not self.path.is_file():
            raise FileNotFoundError(f"workbook not found: {self.path}")
        try:
            self._book: dict[str, pd.DataFrame] = pd.read_excel(self.path, sheet_name=None)
        except ImportError as exc:
            raise ImportError("reading Excel workbooks needs openpyxl: pip install openpyxl") from exc

    @property
    def sheet_names(self) -> list[str]:
        return list(self._book)

    def raw_sheet(self, name: str, *, required: bool = True) -> pd.DataFrame | None:
        """A raw sheet by name, or None when absent and not required."""
        for actual, frame in self._book.items():
            if canonical_name(actual) == canonical_name(name):
                return frame.copy()
        if required:
            raise ValueError(f"{self.path.name}: no sheet named {name!r}; sheets are {self.sheet_names}")
        return None

    def fund_specs(self) -> pd.DataFrame:
        return normalize_fund_specs(self.raw_sheet(self.sheets.fund_spec))

    def fund_market_data(self) -> pd.DataFrame:
        return normalize_fund_market_data(self.raw_sheet(self.sheets.fund_market_data))

    def market_data(self) -> pd.DataFrame:
        return normalize_market_data(self.raw_sheet(self.sheets.market_data))

    def commitment_rates(self) -> pd.DataFrame | None:
        raw = self.raw_sheet(self.sheets.commitment_rates, required=False)
        return None if raw is None else normalize_commitment_rates(raw)

    def __repr__(self) -> str:
        return f"ExcelRepository({str(self.path)!r}, sheets={self.sheet_names})"
