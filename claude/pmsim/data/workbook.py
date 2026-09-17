"""The five-sheet portfolio workbook: Liquid, FX, Flows, Commitments, Spec.

    Liquid        <blank> | one column of monthly returns per profile, named "<CCY> <Risk>"
                  (USD Conservative, USD Moderate, EUR Conservative, ...)
    FX            <blank> | spot rates named "<CCY>USD" (EURUSD, GBPUSD): USD per 1 unit of CCY
    Flows         Vintage | Date | Value | Type (Flow / NAV) | Scale
                  unit = Value ÷ Scale; a negative Flow is a call, a positive one a distribution
    Commitments   Type | Year | Currency | Risk | Commitment | Rate
                  Year is years since inception (0 = the year of the first Liquid date); Rate is decimal
    Spec          Name | Year | Type
                  Year holds the fund's closing date (dd/mm/yyyy)

A *profile* is a (currency, risk) pair. It selects the Liquid return column, the
Commitments rows, and — when the currency is not USD — the FX column to invert. The
result is the same four normalized tables every repository delivers, so the orchestrator
is unchanged; ``WorkbookRepository.simulation_spec()`` builds the matching ``SimulationSpec``.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd

from ..inputs import PRIVATE_CURRENCY
from .orchestrator import Orchestrator, SimulationSpec
from .tables import (
    _dates,
    _numbers,
    _non_empty_rows,
    _texts,
    _years,
    canonical_name,
    find_column,
    normalize_fund_market_data,
    normalize_fund_specs,
    normalize_market_data,
)


@dataclass(frozen=True)
class SheetLayout:
    """Sheet names of the five-sheet workbook. Matched case-, space- and hyphen-insensitively."""

    liquid: str = "Liquid"
    fx: str = "FX"
    flows: str = "Flows"
    commitments: str = "Commitments"
    spec: str = "Spec"


def _column_named(frame: pd.DataFrame, name: str, *, table: str) -> Any:
    matches = [column for column in frame.columns if canonical_name(column) == canonical_name(name)]
    if not matches:
        raise ValueError(f"{table}: no column {name!r}; columns are {list(frame.columns)}")
    return matches[0]


def calendar_rates_for_profile(raw: Any, *, currency: str, risk: str, inception_year: int) -> pd.DataFrame:
    """Calendar year × fund type for one profile, from a schedule keyed by years since inception."""
    table = "Commitments"
    frame = _non_empty_rows(raw, table)
    type_column = find_column(frame, "fund_type", table=table)
    year_column = find_column(frame, "year", table=table)
    rate_column = find_column(frame, "rate", table=table)
    currency_column = _column_named(frame, "currency", table=table)
    risk_column = _column_named(frame, "risk", table=table)
    currencies = frame[currency_column].map(canonical_name)
    risks = frame[risk_column].map(canonical_name)
    rows = frame[(currencies == canonical_name(currency)) & (risks == canonical_name(risk))]
    if rows.empty:
        profiles = sorted({f"{c} {r}" for c, r in zip(frame[currency_column], frame[risk_column])})
        raise ValueError(f"{table}: no rows for profile {currency!r} {risk!r}; profiles are {profiles}")
    relative = _years(rows[year_column], table=table, column=year_column)
    if min(relative) < 0:
        raise ValueError(f"{table}: {year_column} counts years since inception and cannot be negative")
    long = pd.DataFrame({
        "year": [inception_year + offset for offset in relative],
        "fund_type": _texts(rows[type_column], table=table, column=type_column),
        "rate": _numbers(rows[rate_column], table=table, column=rate_column),
    })
    duplicated = long.duplicated(["year", "fund_type"])
    if duplicated.any():
        first = long[duplicated].iloc[0]
        raise ValueError(f"{table}: more than one rate for {first['fund_type']!r} in relative year "
                         f"{first['year'] - inception_year} of profile {currency} {risk}")
    wide = long.pivot(index="year", columns="fund_type", values="rate")
    if wide.isna().any().any():
        missing = [(int(year) - inception_year, fund_type) for fund_type in wide.columns
                   for year in wide.index[wide[fund_type].isna()]]
        raise ValueError(f"{table}: profile {currency} {risk} has no rate for relative year(s) {missing[:6]}")
    wide = wide.sort_index().astype(float)
    wide.index.name = "year"
    wide.columns.name = "fund_type"
    return wide


class WorkbookRepository:
    """The five-sheet workbook for one profile, as a ``DataRepository``."""

    def __init__(self, path: Any, currency: str, risk: str, layout: SheetLayout = SheetLayout()) -> None:
        if not isinstance(currency, str) or not currency.strip():
            raise ValueError("currency must be a code such as 'USD' or 'EUR'")
        if not isinstance(risk, str) or not risk.strip():
            raise ValueError("risk must be a profile name such as 'Conservative'")
        self.path = Path(path)
        self.currency = currency.strip().upper()
        self.risk = risk.strip()
        self.layout = layout
        if not self.path.is_file():
            raise FileNotFoundError(f"workbook not found: {self.path}")
        try:
            self._book: dict[str, pd.DataFrame] = pd.read_excel(self.path, sheet_name=None)
        except ImportError as exc:
            raise ImportError("reading Excel workbooks needs openpyxl: pip install openpyxl") from exc

    # ----------------------------------------------------------------- sheets
    @property
    def sheet_names(self) -> list[str]:
        return list(self._book)

    def raw_sheet(self, name: str, *, required: bool = True) -> pd.DataFrame | None:
        for actual, frame in self._book.items():
            if canonical_name(actual) == canonical_name(name):
                return frame.copy()
        if required:
            raise ValueError(f"{self.path.name}: no sheet named {name!r}; sheets are {self.sheet_names}")
        return None

    def _time_series_sheet(self, name: str, *, required: bool = True) -> pd.DataFrame | None:
        """A time-series sheet with its date column named ``date``; a blank first header counts as the date."""
        raw = self.raw_sheet(name, required=required)
        if raw is None:
            return None
        frame = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"{name}: the sheet is empty")
        date_column = find_column(frame, "date", table=name, required=False)
        if date_column is None:
            date_column = frame.columns[0]
        frame = frame.rename(columns={date_column: "date"})
        frame["date"] = [pd.Timestamp(d) for d in _dates(frame["date"], table=name, column="date")]
        frame.columns = ["date"] + [str(c).strip() for c in frame.columns[1:]]
        return frame

    @cached_property
    def _liquid(self) -> pd.DataFrame:
        return self._time_series_sheet(self.layout.liquid)

    @cached_property
    def _fx(self) -> pd.DataFrame | None:
        return self._time_series_sheet(self.layout.fx, required=False)

    # ---------------------------------------------------------------- profile
    @property
    def profile(self) -> str:
        return f"{self.currency} {self.risk}"

    @cached_property
    def liquid_column(self) -> str:
        columns = [c for c in self._liquid.columns if c != "date"]
        matches = [c for c in columns if canonical_name(c) == canonical_name(self.profile)]
        if not matches:
            raise ValueError(f"{self.layout.liquid}: no column for profile {self.profile!r}; profiles are {columns}")
        return matches[0]

    @cached_property
    def _fx_column_and_quote(self) -> tuple[str | None, str]:
        if self.currency == PRIVATE_CURRENCY:
            return None, "base_per_usd"
        if self._fx is None:
            raise ValueError(f"{self.layout.fx}: sheet is required for the non-USD profile {self.profile!r}")
        columns = [c for c in self._fx.columns if c != "date"]
        for name, quote in ((f"{self.currency}USD", "usd_per_base"), (f"USD{self.currency}", "base_per_usd")):
            matches = [c for c in columns if canonical_name(c) == canonical_name(name)]
            if matches:
                return matches[0], quote
        raise ValueError(f"{self.layout.fx}: no column {self.currency}USD or USD{self.currency}; columns are {columns}")

    @property
    def fx_column(self) -> str | None:
        return self._fx_column_and_quote[0]

    @property
    def fx_quote(self) -> str:
        return self._fx_column_and_quote[1]

    @cached_property
    def inception_year(self) -> int:
        """Relative year 0 of the Commitments schedule: the year of the first Liquid date."""
        return int(self._liquid["date"].min().year)

    # --------------------------------------------------------- DataRepository
    def fund_specs(self) -> pd.DataFrame:
        table = self.layout.spec
        frame = _non_empty_rows(self.raw_sheet(table), table)
        name_column = find_column(frame, "fund_name", table=table)
        type_column = find_column(frame, "fund_type", table=table)
        closing_column = find_column(frame, "closing_date", table=table, required=False)
        if closing_column is None:
            closing_column = _column_named(frame, "year", table=table)
        closing = frame[closing_column]
        if not pd.api.types.is_datetime64_any_dtype(closing):
            closing = pd.to_datetime(closing, dayfirst=True)  # the sheet writes dd/mm/yyyy
        return normalize_fund_specs(pd.DataFrame({
            "fund_name": frame[name_column], "fund_type": frame[type_column], "closing_date": closing,
        }))

    def fund_market_data(self) -> pd.DataFrame:
        return normalize_fund_market_data(self.raw_sheet(self.layout.flows))

    def market_data(self) -> pd.DataFrame:
        frame = self._liquid
        if self._fx is not None:
            frame = frame.merge(self._fx, on="date", how="outer", suffixes=("", " (fx)"))
        return normalize_market_data(frame)

    def commitment_rates(self) -> pd.DataFrame:
        return calendar_rates_for_profile(self.raw_sheet(self.layout.commitments), currency=self.currency, risk=self.risk,
                                   inception_year=self.inception_year)

    # ------------------------------------------------------------------- spec
    def simulation_spec(self, initial_value: float, **overrides: Any) -> SimulationSpec:
        """The ``SimulationSpec`` for this profile: returns compounded from ``initial_value``, FX inverted as needed."""
        settings: dict[str, Any] = dict(
            base_currency=self.currency, liquid_series=self.liquid_column, liquid_kind="returns",
            initial_value=initial_value, fx_series=self.fx_column, fx_quote=self.fx_quote,
        )
        settings.update(overrides)
        return SimulationSpec(**settings)

    def __repr__(self) -> str:
        return f"WorkbookRepository({str(self.path)!r}, profile={self.profile!r})"


def load_profile_workbook(path: Any, currency: str, risk: str, initial_value: float, *,
                          layout: SheetLayout = SheetLayout(), **overrides: Any) -> Orchestrator:
    """An ``Orchestrator`` for one profile of the five-sheet workbook."""
    repository = WorkbookRepository(path, currency, risk, layout)
    return Orchestrator(repository, repository.simulation_spec(initial_value, **overrides))


def run_profile_workbook(path: Any, currency: str, risk: str, initial_value: float, *,
                         layout: SheetLayout = SheetLayout(), **overrides: Any):
    return load_profile_workbook(path, currency, risk, initial_value, layout=layout, **overrides).run()
