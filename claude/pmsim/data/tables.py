"""The tables a data source must deliver, and the column aliases accepted on the way in.

Every repository — the Excel workbook today, a database later — hands over the same
tables in the same shape, so the orchestrator never sees a source-specific column name.
Normalization lives here and every repository applies it:

    fund_spec          fund_name, fund_type, closing_date
    fund_market_data   fund_name, kind, date, value, scale, unit
                       kind is flow | nav | call | distribution; unit = value / scale
    market_data        one column per series, indexed by date — from a wide sheet
                       (date + one column per series) or a long one (date, series, value)
    commitment_rates   calendar year (index) × fund type (columns) — from a wide sheet
                       (year + one column per type) or a long one (year, type, rate); optional
    expected_returns   portfolio name → the yearly expected return its pacing schedule was built
                       on, as a decimal (0.054 is 5.4%); optional. The workbook's Liquid Spec
                       sheet holds it as Liquid | ExRet

Column names are matched case-, space- and hyphen-insensitively against ALIASES.
"""
from __future__ import annotations

import math
import re
from datetime import date
from numbers import Integral
from typing import Any, Iterable

import pandas as pd

from ..dates import as_date
from ..policy import validate_expected_return

FUND_SPEC_COLUMNS = ["fund_name", "fund_type", "closing_date"]
FUND_MARKET_COLUMNS = ["fund_name", "kind", "date", "value", "scale"]

KINDS = {
    "flow": "flow", "flows": "flow", "cash_flow": "flow", "cashflow": "flow",
    "nav": "nav", "mark": "nav", "valuation": "nav",
    "call": "call", "calls": "call", "capital_call": "call", "contribution": "call",
    "distribution": "distribution", "distributions": "distribution", "dist": "distribution",
}

ALIASES: dict[str, frozenset[str]] = {
    "fund_name": frozenset({"fund_name", "fund", "name", "fundname", "fund_id", "vintage"}),
    "fund_type": frozenset({"fund_type", "type", "strategy", "fundtype", "asset_class"}),
    "closing_date": frozenset({"closing_date", "closingdate", "close_date", "closing", "commitment_date"}),
    "kind": frozenset({"kind", "type", "entry_type", "record_type", "data_type"}),
    "date": frozenset({"date", "observation_date", "as_of", "asof"}),
    "value": frozenset({"value", "amount"}),
    "scale": frozenset({"scale", "divisor", "commitment", "commitment_size"}),
    "year": frozenset({"year", "calendar_year", "policy_year"}),
    "rate": frozenset({"rate", "commitment_rate", "target", "value"}),
    "series": frozenset({"series", "name", "ticker", "field", "variable", "item"}),
    "portfolio": frozenset({"portfolio", "portfolio_name", "profile", "name", "liquid"}),
    "expected_return": frozenset({"expected_return", "expected_returns", "expected_ret", "exret", "ex_ret", "return", "x"}),
}


def canonical_name(name: Any) -> str:
    """Column names compared case-, space- and hyphen-insensitively."""
    return re.sub(r"[\s\-]+", "_", str(name).strip().lower())


def _is_missing(value: Any) -> bool:
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def find_column(frame: pd.DataFrame, wanted: str, *, table: str, required: bool = True) -> Any:
    """The frame column that ALIASES says is ``wanted``; None when absent and not required."""
    by_canon: dict[str, Any] = {}
    for column in frame.columns:
        key = canonical_name(column)
        if key in by_canon:
            raise ValueError(f"{table}: columns {by_canon[key]!r} and {column!r} are the same name")
        by_canon[key] = column
    matches = [by_canon[alias] for alias in sorted(ALIASES[wanted]) if alias in by_canon]
    if len(matches) > 1:
        raise ValueError(f"{table}: columns {matches} all look like {wanted!r}; keep one")
    if not matches:
        if required:
            raise ValueError(f"{table}: no column for {wanted!r}; columns are {list(frame.columns)}")
        return None
    return matches[0]


def _non_empty_rows(raw: Any, table: str) -> pd.DataFrame:
    if not isinstance(raw, pd.DataFrame):
        raise TypeError(f"{table}: expected a DataFrame, got {type(raw).__name__}")
    return raw.dropna(how="all").reset_index(drop=True)


def _texts(values: Iterable[Any], *, table: str, column: Any) -> list[str]:
    out = []
    for i, value in enumerate(values):
        if _is_missing(value):
            raise ValueError(f"{table}: {column} is missing in row {i + 1}")
        out.append(str(value).strip())
    return out


def _dates(values: Iterable[Any], *, table: str, column: Any) -> list[date]:
    out = []
    for i, value in enumerate(values):
        try:
            out.append(as_date(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{table}: {column} in row {i + 1}: {exc}") from None
    return out


def _numbers(values: Iterable[Any], *, table: str, column: Any,
             allow_missing: bool = False, positive: bool = False) -> list[float]:
    out = []
    for i, value in enumerate(values):
        if _is_missing(value):
            if allow_missing:
                out.append(float("nan"))
                continue
            raise ValueError(f"{table}: {column} is missing in row {i + 1}")
        if isinstance(value, bool):
            raise ValueError(f"{table}: {column} in row {i + 1} is a boolean, not a number")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{table}: {column} in row {i + 1} is {value!r}, not a number") from None
        if not math.isfinite(number):
            raise ValueError(f"{table}: {column} in row {i + 1} must be finite")
        if positive and number <= 0:
            raise ValueError(f"{table}: {column} in row {i + 1} must be positive, got {number!r}")
        out.append(number)
    return out


def _years(values: Iterable[Any], *, table: str, column: Any) -> list[int]:
    out = []
    for i, value in enumerate(values):
        if isinstance(value, bool) or _is_missing(value):
            raise ValueError(f"{table}: {column} in row {i + 1} is not a calendar year")
        if isinstance(value, Integral):
            out.append(int(value))
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{table}: {column} in row {i + 1} is {value!r}, not a calendar year") from None
        if not number.is_integer():
            raise ValueError(f"{table}: {column} in row {i + 1} is {value!r}, not a calendar year")
        out.append(int(number))
    return out


def _kinds(values: Iterable[Any], *, table: str) -> list[str]:
    out = []
    for i, value in enumerate(values):
        key = "" if _is_missing(value) else canonical_name(value)
        if key not in KINDS:
            raise ValueError(f"{table}: row {i + 1} has type {value!r}; expected Flow, NAV, Call or Distribution")
        out.append(KINDS[key])
    return out


# ------------------------------------------------------------------ tables
def normalize_fund_specs(raw: Any) -> pd.DataFrame:
    """fund_name, fund_type, closing_date — one row per fund, names unique."""
    table = "fund_spec"
    frame = _non_empty_rows(raw, table)
    columns = {wanted: find_column(frame, wanted, table=table) for wanted in FUND_SPEC_COLUMNS}
    out = pd.DataFrame({
        "fund_name": _texts(frame[columns["fund_name"]], table=table, column="fund_name"),
        "fund_type": _texts(frame[columns["fund_type"]], table=table, column="fund_type"),
        "closing_date": _dates(frame[columns["closing_date"]], table=table, column="closing_date"),
    })
    duplicates = sorted(set(out["fund_name"][out["fund_name"].duplicated()]))
    if duplicates:
        raise ValueError(f"{table}: fund_name must be unique; duplicated: {duplicates}")
    return out


def normalize_fund_market_data(raw: Any) -> pd.DataFrame:
    """fund_name, kind, date, value, scale, unit — one row per fund event; unit = value / scale."""
    table = "fund_market_data"
    frame = _non_empty_rows(raw, table)
    columns = {wanted: find_column(frame, wanted, table=table) for wanted in FUND_MARKET_COLUMNS}
    kinds = _kinds(frame[columns["kind"]], table=table)
    out = pd.DataFrame({
        "fund_name": _texts(frame[columns["fund_name"]], table=table, column="fund_name"),
        "kind": kinds,
        "date": _dates(frame[columns["date"]], table=table, column="date"),
        "value": _numbers(frame[columns["value"]], table=table, column="value"),
        "scale": _numbers(frame[columns["scale"]], table=table, column="scale", positive=True),
    })
    out["unit"] = out["value"] / out["scale"]
    return out.sort_values(["fund_name", "date"], kind="stable").reset_index(drop=True)


def normalize_market_data(raw: Any) -> pd.DataFrame:
    """One float column per series, indexed by date. Blank cells stay NaN (sparse series are fine)."""
    table = "market_data"
    frame = _non_empty_rows(raw, table)
    date_column = find_column(frame, "date", table=table)
    value_column = find_column(frame, "value", table=table, required=False)
    series_column = find_column(frame, "series", table=table, required=False)
    stamps = pd.DatetimeIndex(
        [pd.Timestamp(d) for d in _dates(frame[date_column], table=table, column=date_column)], name="date"
    )
    if value_column is not None and series_column is not None:
        long = pd.DataFrame({
            "date": stamps,
            "series": _texts(frame[series_column], table=table, column=series_column),
            "value": _numbers(frame[value_column], table=table, column=value_column, allow_missing=True),
        })
        duplicated = long.duplicated(["date", "series"])
        if duplicated.any():
            first = long[duplicated].iloc[0]
            raise ValueError(f"{table}: more than one value for {first['series']!r} on {first['date'].date()}")
        wide = long.pivot(index="date", columns="series", values="value")
    else:
        wide = pd.DataFrame(index=stamps)
        for column in frame.columns:
            if column == date_column:
                continue
            wide[str(column).strip()] = _numbers(frame[column], table=table, column=column, allow_missing=True)
        if wide.index.has_duplicates:
            first = wide.index[wide.index.duplicated()][0].date()
            raise ValueError(f"{table}: more than one row for {first}")
    wide = wide.sort_index().astype(float)
    wide.columns.name = None
    wide.index.name = "date"
    return wide


def normalize_commitment_rates(raw: Any) -> pd.DataFrame:
    """Calendar year (index) × fund type (columns). Every cell must be given; use 0 for no target."""
    table = "commitment_rates"
    frame = _non_empty_rows(raw, table)
    year_column = find_column(frame, "year", table=table)
    type_column = find_column(frame, "fund_type", table=table, required=False)
    rate_column = find_column(frame, "rate", table=table, required=False)
    years = _years(frame[year_column], table=table, column=year_column)
    if type_column is not None and rate_column is not None:
        long = pd.DataFrame({
            "year": years,
            "fund_type": _texts(frame[type_column], table=table, column=type_column),
            "rate": _numbers(frame[rate_column], table=table, column=rate_column),
        })
        duplicated = long.duplicated(["year", "fund_type"])
        if duplicated.any():
            first = long[duplicated].iloc[0]
            raise ValueError(f"{table}: more than one rate for {first['fund_type']!r} in {first['year']}")
        wide = long.pivot(index="year", columns="fund_type", values="rate")
    else:
        wide = pd.DataFrame(index=pd.Index(years, name="year"))
        for column in frame.columns:
            if column == year_column:
                continue
            wide[str(column).strip()] = _numbers(frame[column], table=table, column=column, allow_missing=True)
        if wide.index.has_duplicates:
            first = wide.index[wide.index.duplicated()][0]
            raise ValueError(f"{table}: year {first} appears more than once")
    if wide.isna().any().any():
        missing = [(int(year), fund_type) for fund_type in wide.columns for year in wide.index[wide[fund_type].isna()]]
        raise ValueError(f"{table}: no rate for {missing[:6]}; use 0 for years with no target")
    wide = wide.sort_index().astype(float)
    wide.index.name = "year"
    wide.columns.name = "fund_type"
    return wide


def _percent_to_decimal(values: Iterable[Any]) -> list[Any]:
    """A cell typed as the text "5.4%" read as 0.054. A percentage-formatted number is already 0.054 and passes through."""
    out = []
    for value in values:
        if isinstance(value, str) and value.strip().endswith("%"):
            try:
                out.append(float(value.strip()[:-1]) / 100.0)
                continue
            except ValueError:
                pass
        out.append(value)
    return out


def normalize_expected_returns(raw: Any) -> pd.Series:
    """Portfolio name → yearly expected return as a decimal: the X each portfolio's pacing schedule was built on.

    The workbook's Liquid Spec sheet is ``Liquid | ExRet``; the column names are matched by
    alias, so ``Portfolio | Expected Return`` works too.
    """
    table = "expected returns"
    frame = _non_empty_rows(raw, table)
    portfolio_column = find_column(frame, "portfolio", table=table)
    return_column = find_column(frame, "expected_return", table=table)
    portfolios = _texts(frame[portfolio_column], table=table, column=portfolio_column)
    duplicates = sorted({name for name in portfolios if portfolios.count(name) > 1})
    if duplicates:
        raise ValueError(f"{table}: each portfolio needs one expected return; duplicated: {duplicates}")
    numbers = _numbers(_percent_to_decimal(frame[return_column]), table=table, column=return_column)
    returns = [
        validate_expected_return(value, label=f"{table}: expected return of {portfolio!r}")
        for portfolio, value in zip(portfolios, numbers)
    ]
    return pd.Series(returns, index=pd.Index(portfolios, name="portfolio"), name="expected_return", dtype=float)
