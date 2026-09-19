"""Where the tables come from.

``DataRepository`` is the seam between the engine and its data: anything that can hand over
the four normalized tables described in ``tables.py``. The orchestrator only ever talks to
that protocol. Two repositories implement it:

``FrameRepository`` takes the tables as DataFrames — for tests, notebooks, and as the shape
a database adapter will take.

``WorkbookRepository`` reads the Excel portfolio workbook, one *profile* at a time:

    Liquid        <blank> | one column of monthly returns per profile, named "<CCY> <Risk>"
                  (USD Conservative, USD Moderate, EUR Conservative, ...)
    FX            <blank> | spot rates named "<CCY>USD" (EURUSD, GBPUSD): USD per 1 unit of CCY
    Flows         Vintage | Date | Value | Type (Flow / NAV) | Scale
                  unit = Value ÷ Scale; a negative Flow is a call, a positive one a distribution
    Commitments   Type | Year | Currency | Risk | Commitment | Rate
                  Year is years since inception (0 = the year of the first Liquid date); Rate is decimal
    Spec          Name | Year | Type
                  Year holds the fund's closing date (dd/mm/yyyy)
    Liquid Spec   Liquid | ExRet
                  one row per portfolio, named as its Liquid column; ExRet is the yearly return X
                  its pacing schedule was built on (5.4%, which Excel stores as 0.054)

A profile is a (currency, risk) pair. It selects the Liquid return column, the Commitments
rows, its expected return and — when the currency is not USD — the FX column to invert. What comes out is the
same four tables, and ``WorkbookRepository.simulation_spec()`` builds the matching
``SimulationSpec``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from ..inputs import PRIVATE_CURRENCY
from .spec import SimulationSpec
from .tables import (
    _dates,
    _non_empty_rows,
    _numbers,
    _texts,
    _years,
    canonical_name,
    find_column,
    normalize_commitment_rates,
    normalize_expected_returns,
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

    def expected_returns(self) -> pd.Series | None:
        """Portfolio name → the yearly expected return its pacing schedule was built on; None when the source has none."""


# ------------------------------------------------------------------ DataFrames
class FrameRepository:
    """The tables handed over as DataFrames. Normalized once, at construction, so bad data fails early."""

    def __init__(self, fund_specs: Any, fund_market_data: Any, market_data: Any, commitment_rates: Any = None,
                 expected_returns: Any = None) -> None:
        self._fund_specs = normalize_fund_specs(fund_specs)
        self._fund_market_data = normalize_fund_market_data(fund_market_data)
        self._market_data = normalize_market_data(market_data)
        self._commitment_rates = None if commitment_rates is None else normalize_commitment_rates(commitment_rates)
        self._expected_returns = None if expected_returns is None else normalize_expected_returns(expected_returns)

    def fund_specs(self) -> pd.DataFrame:
        return self._fund_specs.copy()

    def fund_market_data(self) -> pd.DataFrame:
        return self._fund_market_data.copy()

    def market_data(self) -> pd.DataFrame:
        return self._market_data.copy()

    def commitment_rates(self) -> pd.DataFrame | None:
        return None if self._commitment_rates is None else self._commitment_rates.copy()

    def expected_returns(self) -> pd.Series | None:
        return None if self._expected_returns is None else self._expected_returns.copy()


# ------------------------------------------------------- the Excel portfolio workbook
@dataclass(frozen=True)
class SheetLayout:
    """Sheet names of the portfolio workbook. Matched ignoring case, spaces, hyphens and underscores."""

    liquid: str = "Liquid"
    fx: str = "FX"
    flows: str = "Flows"
    commitments: str = "Commitments"
    spec: str = "Spec"
    liquid_spec: str = "Liquid Spec"


def _sheet_key(name: Any) -> str:
    """Sheet names compared ignoring case, spaces, hyphens and underscores: LiquidSpec is Liquid Spec."""
    return re.sub(r"[\s\-_]+", "", str(name).strip().lower())


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
    """The Excel portfolio workbook for one profile. Reads every sheet once; needs openpyxl for .xlsx."""

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
        """A copy of the sheet whose name matches ``name`` loosely, or None when absent and not required."""
        for actual, frame in self._book.items():
            if _sheet_key(actual) == _sheet_key(name):
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
        frame.columns = [c if c == "date" else str(c).strip() for c in frame.columns]  # wherever the date column sits
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
        return calendar_rates_for_profile(self.raw_sheet(self.layout.commitments),
                                          currency=self.currency, risk=self.risk, inception_year=self.inception_year)

    def expected_returns(self) -> pd.Series:
        """The Liquid Spec sheet. Required: the Commitments sheet is a pacing schedule, and means nothing without its X."""
        return normalize_expected_returns(self.raw_sheet(self.layout.liquid_spec))

    @cached_property
    def expected_return(self) -> float:
        """This profile's X: the yearly return the pacing model assumed for its liquid portfolio."""
        table = self.expected_returns()
        matches = [name for name in table.index if canonical_name(name) == canonical_name(self.profile)]
        if not matches:
            raise ValueError(f"{self.layout.liquid_spec}: no row for portfolio {self.profile!r}; "
                             f"portfolios are {list(table.index)}")
        return float(table[matches[0]])

    # ------------------------------------------------------------------- spec
    def simulation_spec(self, initial_value: float, **overrides: Any) -> SimulationSpec:
        """The ``SimulationSpec`` for this profile: returns compounded from ``initial_value``, FX inverted, its expected return."""
        settings: dict[str, Any] = dict(
            base_currency=self.currency, liquid_series=self.liquid_column, liquid_kind="returns",
            initial_value=initial_value, fx_series=self.fx_column, fx_quote=self.fx_quote,
            expected_return=self.expected_return,
        )
        settings.update(overrides)
        return SimulationSpec(**settings)

    def __repr__(self) -> str:
        return f"WorkbookRepository({str(self.path)!r}, profile={self.profile!r})"
