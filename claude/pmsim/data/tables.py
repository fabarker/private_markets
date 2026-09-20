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

# What a fund_market_data row may call itself, and the kind that means.
KINDS = {
    "flow": "flow", "flows": "flow", "cash_flow": "flow", "cashflow": "flow",
    "nav": "nav", "mark": "nav", "valuation": "nav",
    "call": "call", "calls": "call", "capital_call": "call", "contribution": "call",
    "distribution": "distribution", "distributions": "distribution", "dist": "distribution",
}

# The column names accepted for each thing a table must hold.
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
    "expected_return": frozenset({
        "expected_return", "expected_returns", "expected_ret", "exret", "ex_ret", "return", "x",
    }),
    # deliberately not "years": the Spec sheet's own "Year" column holds the closing date
    "draws": frozenset({"draws", "draw", "draw_plan", "drawn_years", "schedule_years", "commitment_years"}),
}


# ------------------------------------------------------------------ finding columns
def canonical_name(name: Any) -> str:
    """Column names compared case-, space- and hyphen-insensitively."""
    lower_case = str(name).strip().lower()
    return re.sub(r"[\s\-]+", "_", lower_case)


def _is_missing(value: Any) -> bool:
    """True for a blank cell: empty text, None, NaN or NaT."""
    if isinstance(value, str):
        return not value.strip()

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def find_column(frame: pd.DataFrame, wanted: str, *, table: str, required: bool = True) -> Any:
    """The frame column that ALIASES says is ``wanted``; None when absent and not required."""
    # The frame's columns, by canonical name. Two that canonicalise alike cannot be told apart.
    column_by_canonical_name: dict[str, Any] = {}

    for column in frame.columns:
        key = canonical_name(column)

        if key in column_by_canonical_name:
            already_there = column_by_canonical_name[key]
            raise ValueError(f"{table}: columns {already_there!r} and {column!r} are the same name")

        column_by_canonical_name[key] = column

    # The columns whose name is one of the aliases of what is wanted.
    matches = []
    for alias in sorted(ALIASES[wanted]):
        if alias in column_by_canonical_name:
            matches.append(column_by_canonical_name[alias])

    if len(matches) > 1:
        raise ValueError(f"{table}: columns {matches} all look like {wanted!r}; keep one")

    if not matches:
        if required:
            raise ValueError(f"{table}: no column for {wanted!r}; columns are {list(frame.columns)}")
        return None

    return matches[0]


# ------------------------------------------------------------------ reading columns
def _drop_blank_rows(raw: Any, table: str) -> pd.DataFrame:
    """The table without its entirely blank rows, renumbered from zero."""
    if not isinstance(raw, pd.DataFrame):
        raise TypeError(f"{table}: expected a DataFrame, got {type(raw).__name__}")

    return raw.dropna(how="all").reset_index(drop=True)


def _read_text_column(values: Iterable[Any], *, table: str, column: Any) -> list[str]:
    """Every cell as text without surrounding spaces; a blank cell is an error."""
    texts = []

    for row_number, value in enumerate(values, start=1):
        if _is_missing(value):
            raise ValueError(f"{table}: {column} is missing in row {row_number}")

        texts.append(str(value).strip())

    return texts


def _read_date_column(values: Iterable[Any], *, table: str, column: Any) -> list[date]:
    """Every cell as a calendar date."""
    dates = []

    for row_number, value in enumerate(values, start=1):
        try:
            dates.append(as_date(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{table}: {column} in row {row_number}: {exc}") from None

    return dates


def _read_number_column(values: Iterable[Any], *, table: str, column: Any,
                        allow_missing: bool = False, positive: bool = False) -> list[float]:
    """Every cell as a finite float.

    A blank cell is NaN when ``allow_missing`` and an error otherwise. With ``positive``
    every number must be above zero.
    """
    numbers = []

    for row_number, value in enumerate(values, start=1):
        if _is_missing(value):
            if allow_missing:
                numbers.append(float("nan"))
                continue
            raise ValueError(f"{table}: {column} is missing in row {row_number}")

        if isinstance(value, bool):
            raise ValueError(f"{table}: {column} in row {row_number} is a boolean, not a number")

        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{table}: {column} in row {row_number} is {value!r}, not a number") from None

        if not math.isfinite(number):
            raise ValueError(f"{table}: {column} in row {row_number} must be finite")

        if positive and number <= 0:
            raise ValueError(f"{table}: {column} in row {row_number} must be positive, got {number!r}")

        numbers.append(number)

    return numbers


def _read_year_column(values: Iterable[Any], *, table: str, column: Any) -> list[int]:
    """Every cell as a whole-number year. 2027 and 2027.0 are years; 2027.5 and "FY27" are not."""
    years = []

    for row_number, value in enumerate(values, start=1):
        if isinstance(value, bool) or _is_missing(value):
            raise ValueError(f"{table}: {column} in row {row_number} is not a calendar year")

        if isinstance(value, Integral):
            years.append(int(value))
            continue

        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"{table}: {column} in row {row_number} is {value!r}, not a calendar year"
            ) from None

        if not number.is_integer():
            raise ValueError(f"{table}: {column} in row {row_number} is {value!r}, not a calendar year")

        years.append(int(number))

    return years


def _read_kind_column(values: Iterable[Any], *, table: str) -> list[str]:
    """Every cell as one of flow, nav, call or distribution."""
    kinds = []

    for row_number, value in enumerate(values, start=1):
        if _is_missing(value):
            key = ""
        else:
            key = canonical_name(value)

        if key not in KINDS:
            raise ValueError(
                f"{table}: row {row_number} has type {value!r}; expected Flow, NAV, Call or Distribution"
            )

        kinds.append(KINDS[key])

    return kinds


# ------------------------------------------------------------------ tables
def normalize_fund_specs(raw: Any) -> pd.DataFrame:
    """fund_name, fund_type, closing_date — one row per fund, names unique."""
    table = "fund_spec"
    frame = _drop_blank_rows(raw, table)

    name_column = find_column(frame, "fund_name", table=table)
    type_column = find_column(frame, "fund_type", table=table)
    closing_column = find_column(frame, "closing_date", table=table)

    specs = pd.DataFrame({
        "fund_name": _read_text_column(frame[name_column], table=table, column="fund_name"),
        "fund_type": _read_text_column(frame[type_column], table=table, column="fund_type"),
        "closing_date": _read_date_column(frame[closing_column], table=table, column="closing_date"),
    })

    is_repeated = specs["fund_name"].duplicated()
    duplicates = sorted(set(specs["fund_name"][is_repeated]))
    if duplicates:
        raise ValueError(f"{table}: fund_name must be unique; duplicated: {duplicates}")

    return specs


def normalize_fund_market_data(raw: Any) -> pd.DataFrame:
    """fund_name, kind, date, value, scale, unit — one row per fund event; unit = value / scale."""
    table = "fund_market_data"
    frame = _drop_blank_rows(raw, table)

    columns = {}
    for wanted in FUND_MARKET_COLUMNS:
        columns[wanted] = find_column(frame, wanted, table=table)

    kinds = _read_kind_column(frame[columns["kind"]], table=table)

    events = pd.DataFrame({
        "fund_name": _read_text_column(frame[columns["fund_name"]], table=table, column="fund_name"),
        "kind": kinds,
        "date": _read_date_column(frame[columns["date"]], table=table, column="date"),
        "value": _read_number_column(frame[columns["value"]], table=table, column="value"),
        "scale": _read_number_column(frame[columns["scale"]], table=table, column="scale", positive=True),
    })

    # Per 1 committed: the workbook quotes each fund's figures against a commitment size.
    events["unit"] = events["value"] / events["scale"]

    in_fund_and_date_order = events.sort_values(["fund_name", "date"], kind="stable")
    return in_fund_and_date_order.reset_index(drop=True)


def _market_data_from_long_sheet(frame: pd.DataFrame, stamps: pd.DatetimeIndex,
                                 series_column: Any, value_column: Any, table: str) -> pd.DataFrame:
    """A sheet of (date, series, value) rows, pivoted to one column per series."""
    long = pd.DataFrame({
        "date": stamps,
        "series": _read_text_column(frame[series_column], table=table, column=series_column),
        "value": _read_number_column(
            frame[value_column], table=table, column=value_column, allow_missing=True,
        ),
    })

    is_repeated = long.duplicated(["date", "series"])
    if is_repeated.any():
        first = long[is_repeated].iloc[0]
        raise ValueError(f"{table}: more than one value for {first['series']!r} on {first['date'].date()}")

    return long.pivot(index="date", columns="series", values="value")


def _market_data_from_wide_sheet(frame: pd.DataFrame, stamps: pd.DatetimeIndex,
                                 date_column: Any, table: str) -> pd.DataFrame:
    """A sheet with a date column and one column per series, read as it is."""
    wide = pd.DataFrame(index=stamps)

    for column in frame.columns:
        if column == date_column:
            continue

        series_name = str(column).strip()
        wide[series_name] = _read_number_column(frame[column], table=table, column=column, allow_missing=True)

    if wide.index.has_duplicates:
        first_repeated_day = wide.index[wide.index.duplicated()][0].date()
        raise ValueError(f"{table}: more than one row for {first_repeated_day}")

    return wide


def normalize_market_data(raw: Any) -> pd.DataFrame:
    """One float column per series, indexed by date. Blank cells stay NaN (sparse series are fine)."""
    table = "market_data"
    frame = _drop_blank_rows(raw, table)

    date_column = find_column(frame, "date", table=table)
    value_column = find_column(frame, "value", table=table, required=False)
    series_column = find_column(frame, "series", table=table, required=False)

    dates = _read_date_column(frame[date_column], table=table, column=date_column)
    stamps = pd.DatetimeIndex([pd.Timestamp(day) for day in dates], name="date")

    # A long sheet names its series in a column; a wide one has a column per series.
    is_long_sheet = value_column is not None and series_column is not None
    if is_long_sheet:
        wide = _market_data_from_long_sheet(frame, stamps, series_column, value_column, table)
    else:
        wide = _market_data_from_wide_sheet(frame, stamps, date_column, table)

    wide = wide.sort_index().astype(float)
    wide.columns.name = None
    wide.index.name = "date"
    return wide


def _commitment_rates_from_long_sheet(frame: pd.DataFrame, years: list[int],
                                      type_column: Any, rate_column: Any, table: str) -> pd.DataFrame:
    """A sheet of (year, fund type, rate) rows, pivoted to one column per fund type."""
    long = pd.DataFrame({
        "year": years,
        "fund_type": _read_text_column(frame[type_column], table=table, column=type_column),
        "rate": _read_number_column(frame[rate_column], table=table, column=rate_column),
    })

    is_repeated = long.duplicated(["year", "fund_type"])
    if is_repeated.any():
        first = long[is_repeated].iloc[0]
        raise ValueError(f"{table}: more than one rate for {first['fund_type']!r} in {first['year']}")

    return long.pivot(index="year", columns="fund_type", values="rate")


def _commitment_rates_from_wide_sheet(frame: pd.DataFrame, years: list[int],
                                      year_column: Any, table: str) -> pd.DataFrame:
    """A sheet with a year column and one column per fund type, read as it is."""
    wide = pd.DataFrame(index=pd.Index(years, name="year"))

    for column in frame.columns:
        if column == year_column:
            continue

        fund_type = str(column).strip()
        wide[fund_type] = _read_number_column(frame[column], table=table, column=column, allow_missing=True)

    if wide.index.has_duplicates:
        first_repeated_year = wide.index[wide.index.duplicated()][0]
        raise ValueError(f"{table}: year {first_repeated_year} appears more than once")

    return wide


def normalize_commitment_rates(raw: Any) -> pd.DataFrame:
    """Calendar year (index) × fund type (columns). Every cell must be given; use 0 for no target."""
    table = "commitment_rates"
    frame = _drop_blank_rows(raw, table)

    year_column = find_column(frame, "year", table=table)
    type_column = find_column(frame, "fund_type", table=table, required=False)
    rate_column = find_column(frame, "rate", table=table, required=False)

    years = _read_year_column(frame[year_column], table=table, column=year_column)

    # A long sheet names its fund types in a column; a wide one has a column per fund type.
    is_long_sheet = type_column is not None and rate_column is not None
    if is_long_sheet:
        wide = _commitment_rates_from_long_sheet(frame, years, type_column, rate_column, table)
    else:
        wide = _commitment_rates_from_wide_sheet(frame, years, year_column, table)

    # Every year needs a rate for every fund type.
    if wide.isna().any().any():
        missing = []
        for fund_type in wide.columns:
            years_without_a_rate = wide.index[wide[fund_type].isna()]
            for year in years_without_a_rate:
                missing.append((int(year), fund_type))

        raise ValueError(f"{table}: no rate for {missing[:6]}; use 0 for years with no target")

    wide = wide.sort_index().astype(float)
    wide.index.name = "year"
    wide.columns.name = "fund_type"
    return wide


def _percent_to_decimal(values: Iterable[Any]) -> list[Any]:
    """A cell typed as the text "5.4%" read as 0.054.

    A percentage-formatted number is already 0.054 and passes through, as does anything else.
    """
    converted = []

    for value in values:
        is_percent_text = isinstance(value, str) and value.strip().endswith("%")

        if is_percent_text:
            digits = value.strip()[:-1]
            try:
                converted.append(float(digits) / 100.0)
                continue
            except ValueError:
                pass  # not a number before the % sign: left as it is, and rejected further on

        converted.append(value)

    return converted


def normalize_expected_returns(raw: Any) -> pd.Series:
    """Portfolio name → yearly expected return, as a decimal.

    This is the X each portfolio's pacing schedule was built on. The workbook's Liquid Spec
    sheet is ``Liquid | ExRet``; the column names are matched by alias, so
    ``Portfolio | Expected Return`` works too.
    """
    table = "expected returns"
    frame = _drop_blank_rows(raw, table)

    portfolio_column = find_column(frame, "portfolio", table=table)
    return_column = find_column(frame, "expected_return", table=table)

    # One row per portfolio.
    portfolios = _read_text_column(frame[portfolio_column], table=table, column=portfolio_column)

    duplicates = sorted({name for name in portfolios if portfolios.count(name) > 1})
    if duplicates:
        raise ValueError(f"{table}: each portfolio needs one expected return; duplicated: {duplicates}")

    # Each return as a decimal, whether it was typed as 0.054 or as "5.4%".
    cells = _percent_to_decimal(frame[return_column])
    numbers = _read_number_column(cells, table=table, column=return_column)

    returns = []
    for portfolio, number in zip(portfolios, numbers):
        label = f"{table}: expected return of {portfolio!r}"
        returns.append(validate_expected_return(number, label=label))

    index = pd.Index(portfolios, name="portfolio")
    return pd.Series(returns, index=index, name="expected_return", dtype=float)


# --------------------------------------------------------------- draw plans
DRAW_TERM = re.compile(r"^(\d{1,4})(?:\s*-\s*(\d{1,4}))?(?:\s*[x*×]\s*(\d+(?:\.\d+)?))?$", re.IGNORECASE)
NO_DRAW_PLAN = frozenset({"", "-", "--", "none", "n/a", "na", "default", "nan"})


def parse_draw_plan(text: Any, *, fund_name: str) -> dict[int, float] | None:
    """One fund's draw plan from one cell: the schedule years it draws, each with a multiplier.

    Years are counted from inception, as the Commitments sheet counts them. The cell holds
    comma-separated terms, each a year (``7``), a run of years (``1-4``), or either of those
    with a multiplier (``12x3``, ``1-4x2``): so ``1-4`` draws years 1, 2, 3 and 4 once each,
    and ``12x3`` draws three times year 12's commitment. Blank, ``-`` and ``none`` all mean no
    plan at all — the fund keeps whatever years ``carry_forward`` gives it.
    """
    if _is_missing(text):
        return None

    written = str(text).strip()
    if written.lower() in NO_DRAW_PLAN:
        return None

    plan: dict[int, float] = {}

    for term in written.split(","):
        term = term.strip()
        if not term:
            continue

        match = DRAW_TERM.match(term)
        if match is None:
            raise ValueError(
                f"draws for {fund_name!r}: cannot read {term!r}; write a year (7), a run of years "
                f"(1-4), or either with a multiplier (12x3)"
            )

        # The three parts of a term: the first year, the last year of a run, the multiplier.
        first_year_text = match.group(1)
        last_year_text = match.group(2)
        multiplier_text = match.group(3)

        first_year = int(first_year_text)
        last_year = int(last_year_text or first_year_text)

        if multiplier_text is None:
            multiplier = 1.0
        else:
            multiplier = float(multiplier_text)

        if last_year < first_year:
            raise ValueError(
                f"draws for {fund_name!r}: {term!r} runs backwards; write the earlier year first"
            )

        for year in range(first_year, last_year + 1):
            if year in plan:
                raise ValueError(
                    f"draws for {fund_name!r}: year {year} appears twice; "
                    f"use a multiplier to draw a year more than once"
                )

            plan[year] = multiplier

    if not plan:
        return None

    return dict(sorted(plan.items()))


def relative_draw_plans_to_calendar(plans: Any, *, inception_year: int) -> dict[str, dict[int, float]]:
    """Draw plans keyed by years since inception, remapped onto calendar years."""
    calendar_plans = {}

    for name, plan in plans.items():
        calendar_plan = {}
        for years_since_inception, multiplier in plan.items():
            calendar_plan[inception_year + years_since_inception] = multiplier

        calendar_plans[name] = calendar_plan

    return calendar_plans
