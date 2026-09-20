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
    Spec          Name | Year | Type | Draws (optional)
                  Year holds the fund's closing date (dd/mm/yyyy); Draws names the schedule
                  years the fund collects, counted from inception: 1-4, 12x3, blank for none
    Liquid Spec   Liquid | ExRet          (also read as Return Spec, or Expected Returns)
                  one row per portfolio, named as its Liquid column; ExRet is the yearly return X
                  its pacing schedule was built on (5.4%, which Excel stores as 0.054)

A profile is a (currency, risk) pair. It selects the Liquid return column, the Commitments
rows, its expected return and — when the currency is not USD — the FX column to invert.
What comes out is the same four tables, and ``WorkbookRepository.simulation_spec()`` builds
the matching ``SimulationSpec``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Mapping, Protocol

import pandas as pd

from ..inputs import PRIVATE_CURRENCY
from ..policy import commitment_rounding_unit
from .spec import SimulationSpec
from .tables import (
    _drop_blank_rows,
    _read_date_column,
    _read_number_column,
    _read_text_column,
    _read_year_column,
    canonical_name,
    find_column,
    normalize_commitment_rates,
    normalize_expected_returns,
    normalize_fund_market_data,
    normalize_fund_specs,
    normalize_market_data,
    parse_draw_plan,
    relative_draw_plans_to_calendar,
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
        """Portfolio name → the yearly expected return its pacing schedule was built on.

        None when the source has none.
        """

    def draw_plans(self) -> Mapping[str, Mapping[int, float]]:
        """Fund name → calendar year → multiplier: which schedule years each fund draws.

        Empty when the source says nothing.
        """


def _copy_of_draw_plans(plans: Mapping[str, Mapping[int, float]]) -> dict[str, dict[int, float]]:
    """A fresh copy, so nobody holding the result can alter the repository's own plans."""
    return {name: dict(plan) for name, plan in plans.items()}


# ------------------------------------------------------------------ DataFrames
class FrameRepository:
    """The tables handed over as DataFrames. Normalized once, at construction, so bad data fails early."""

    def __init__(
        self,
        fund_specs: Any,
        fund_market_data: Any,
        market_data: Any,
        commitment_rates: Any = None,
        expected_returns: Any = None,
        draw_plans: Mapping[str, Mapping[int, float]] | None = None,
    ) -> None:
        # The three tables every source has.
        self._fund_specs = normalize_fund_specs(fund_specs)
        self._fund_market_data = normalize_fund_market_data(fund_market_data)
        self._market_data = normalize_market_data(market_data)

        # The optional ones.
        self._commitment_rates = None
        if commitment_rates is not None:
            self._commitment_rates = normalize_commitment_rates(commitment_rates)

        self._expected_returns = None
        if expected_returns is not None:
            self._expected_returns = normalize_expected_returns(expected_returns)

        # In calendar years, like commitment_rates: these tables are handed over ready to use.
        self._draw_plans = _copy_of_draw_plans(draw_plans or {})

    def fund_specs(self) -> pd.DataFrame:
        return self._fund_specs.copy()

    def fund_market_data(self) -> pd.DataFrame:
        return self._fund_market_data.copy()

    def market_data(self) -> pd.DataFrame:
        return self._market_data.copy()

    def commitment_rates(self) -> pd.DataFrame | None:
        if self._commitment_rates is None:
            return None
        return self._commitment_rates.copy()

    def expected_returns(self) -> pd.Series | None:
        if self._expected_returns is None:
            return None
        return self._expected_returns.copy()

    def draw_plans(self) -> dict[str, dict[int, float]]:
        return _copy_of_draw_plans(self._draw_plans)


# ------------------------------------------------------- the Excel portfolio workbook
@dataclass(frozen=True)
class SheetLayout:
    """Sheet names of the portfolio workbook. Matched ignoring case, spaces, hyphens and underscores.

    ``liquid_spec`` is the sheet of expected returns, under any of the names it goes by; set
    one name to pin it.
    """

    liquid: str = "Liquid"
    fx: str = "FX"
    flows: str = "Flows"
    commitments: str = "Commitments"
    spec: str = "Spec"
    liquid_spec: str | tuple[str, ...] = ("Liquid Spec", "Return Spec", "Expected Returns")


def _sheet_key(name: Any) -> str:
    """Sheet names compared ignoring case, spaces, hyphens and underscores: LiquidSpec is Liquid Spec."""
    lower_case = str(name).strip().lower()
    return re.sub(r"[\s\-_]+", "", lower_case)


def _column_named(frame: pd.DataFrame, name: str, *, table: str) -> Any:
    """The frame column called ``name``, compared loosely; an error listing the columns when there is none."""
    wanted = canonical_name(name)
    matches = [column for column in frame.columns if canonical_name(column) == wanted]

    if not matches:
        raise ValueError(f"{table}: no column {name!r}; columns are {list(frame.columns)}")

    return matches[0]


def _columns_matching(columns: list[Any], name: str) -> list[Any]:
    """The columns whose name is ``name``, compared loosely."""
    wanted = canonical_name(name)
    return [column for column in columns if canonical_name(column) == wanted]


def calendar_rates_for_profile(raw: Any, *, currency: str, risk: str, inception_year: int) -> pd.DataFrame:
    """Calendar year × fund type for one profile, from a schedule keyed by years since inception."""
    table = "Commitments"
    frame = _drop_blank_rows(raw, table)

    type_column = find_column(frame, "fund_type", table=table)
    year_column = find_column(frame, "year", table=table)
    rate_column = find_column(frame, "rate", table=table)
    currency_column = _column_named(frame, "currency", table=table)
    risk_column = _column_named(frame, "risk", table=table)

    # The rows of this profile: its currency and its risk level.
    currencies = frame[currency_column].map(canonical_name)
    risks = frame[risk_column].map(canonical_name)

    is_this_profile = (currencies == canonical_name(currency)) & (risks == canonical_name(risk))
    rows = frame[is_this_profile]

    if rows.empty:
        profiles_in_the_sheet = sorted({
            f"{row_currency} {row_risk}"
            for row_currency, row_risk in zip(frame[currency_column], frame[risk_column])
        })
        raise ValueError(
            f"{table}: no rows for profile {currency!r} {risk!r}; profiles are {profiles_in_the_sheet}"
        )

    # Years since inception become calendar years: year 0 is the inception year.
    years_since_inception = _read_year_column(rows[year_column], table=table, column=year_column)
    if min(years_since_inception) < 0:
        raise ValueError(f"{table}: {year_column} counts years since inception and cannot be negative")

    long = pd.DataFrame({
        "year": [inception_year + offset for offset in years_since_inception],
        "fund_type": _read_text_column(rows[type_column], table=table, column=type_column),
        "rate": _read_number_column(rows[rate_column], table=table, column=rate_column),
    })

    # One rate per year and fund type.
    is_repeated = long.duplicated(["year", "fund_type"])
    if is_repeated.any():
        first = long[is_repeated].iloc[0]
        raise ValueError(
            f"{table}: more than one rate for {first['fund_type']!r} in relative year "
            f"{first['year'] - inception_year} of profile {currency} {risk}"
        )

    wide = long.pivot(index="year", columns="fund_type", values="rate")

    # And none missing: every fund type needs a rate in every year the profile lists.
    if wide.isna().any().any():
        missing = []
        for fund_type in wide.columns:
            years_without_a_rate = wide.index[wide[fund_type].isna()]
            for year in years_without_a_rate:
                missing.append((int(year) - inception_year, fund_type))

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

        # Every sheet, read once: sheet name → DataFrame.
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
        wanted = _sheet_key(name)

        for actual_name, frame in self._book.items():
            if _sheet_key(actual_name) == wanted:
                return frame.copy()

        if required:
            raise ValueError(f"{self.path.name}: no sheet named {name!r}; sheets are {self.sheet_names}")

        return None

    def _time_series_sheet(self, name: str, *, required: bool = True) -> pd.DataFrame | None:
        """A time-series sheet with its date column named ``date``.

        A blank first header counts as the date: the real Liquid and FX sheets leave it blank.
        """
        raw = self.raw_sheet(name, required=required)
        if raw is None:
            return None

        # Drop the blank rows and the blank columns.
        frame = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"{name}: the sheet is empty")

        # The date column: the one named like a date, else the first (the real sheet leaves its header blank).
        date_column = find_column(frame, "date", table=name, required=False)
        if date_column is None:
            date_column = frame.columns[0]

        frame = frame.rename(columns={date_column: "date"})

        dates = _read_date_column(frame["date"], table=name, column="date")
        frame["date"] = [pd.Timestamp(day) for day in dates]

        # Tidy the other column names, wherever the date column sits among them.
        tidied_names = []
        for column in frame.columns:
            if column == "date":
                tidied_names.append(column)
            else:
                tidied_names.append(str(column).strip())

        frame.columns = tidied_names
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
        """The Liquid sheet's column for this profile, e.g. "EUR Moderate"."""
        profile_columns = [column for column in self._liquid.columns if column != "date"]
        matches = _columns_matching(profile_columns, self.profile)

        if not matches:
            raise ValueError(
                f"{self.layout.liquid}: no column for profile {self.profile!r}; "
                f"profiles are {profile_columns}"
            )

        return matches[0]

    @cached_property
    def _fx_column_and_quote(self) -> tuple[str | None, str]:
        """The FX sheet's column for this profile's currency, and which way round it is quoted."""
        # A dollar profile converts nothing.
        if self.currency == PRIVATE_CURRENCY:
            return None, "base_per_usd"

        if self._fx is None:
            raise ValueError(f"{self.layout.fx}: sheet is required for the non-USD profile {self.profile!r}")

        rate_columns = [column for column in self._fx.columns if column != "date"]

        # EURUSD is dollars per euro; USDEUR is euros per dollar. Either will do.
        names_and_quotes = (
            (f"{self.currency}USD", "usd_per_base"),
            (f"USD{self.currency}", "base_per_usd"),
        )
        for name, quote in names_and_quotes:
            matches = _columns_matching(rate_columns, name)
            if matches:
                return matches[0], quote

        raise ValueError(
            f"{self.layout.fx}: no column {self.currency}USD or USD{self.currency}; "
            f"columns are {rate_columns}"
        )

    @property
    def fx_column(self) -> str | None:
        return self._fx_column_and_quote[0]

    @property
    def fx_quote(self) -> str:
        return self._fx_column_and_quote[1]

    @cached_property
    def inception_year(self) -> int:
        """Relative year 0 of the Commitments schedule: the year of the first Liquid date."""
        first_liquid_date = self._liquid["date"].min()
        return int(first_liquid_date.year)

    # --------------------------------------------------------- DataRepository
    def fund_specs(self) -> pd.DataFrame:
        """The Spec sheet: Name | Year | Type, where Year holds the closing date."""
        table = self.layout.spec
        frame = _drop_blank_rows(self.raw_sheet(table), table)

        name_column = find_column(frame, "fund_name", table=table)
        type_column = find_column(frame, "fund_type", table=table)

        # The closing date sits under a closing-date heading, or under "Year" as the real sheet has it.
        closing_column = find_column(frame, "closing_date", table=table, required=False)
        if closing_column is None:
            closing_column = _column_named(frame, "year", table=table)

        # The sheet writes dd/mm/yyyy, which pandas would otherwise read month first.
        closing_dates = frame[closing_column]
        if not pd.api.types.is_datetime64_any_dtype(closing_dates):
            closing_dates = pd.to_datetime(closing_dates, dayfirst=True)

        return normalize_fund_specs(pd.DataFrame({
            "fund_name": frame[name_column],
            "fund_type": frame[type_column],
            "closing_date": closing_dates,
        }))

    def fund_market_data(self) -> pd.DataFrame:
        return normalize_fund_market_data(self.raw_sheet(self.layout.flows))

    def market_data(self) -> pd.DataFrame:
        """The Liquid returns and the FX rates together, one column per series."""
        frame = self._liquid

        if self._fx is not None:
            frame = frame.merge(self._fx, on="date", how="outer", suffixes=("", " (fx)"))

        return normalize_market_data(frame)

    def commitment_rates(self) -> pd.DataFrame:
        return calendar_rates_for_profile(
            self.raw_sheet(self.layout.commitments),
            currency=self.currency,
            risk=self.risk,
            inception_year=self.inception_year,
        )

    def draw_plans(self) -> dict[str, dict[int, float]]:
        """The Spec sheet's optional Draws column, on calendar years.

        Empty when the column is absent, so a workbook that does not have it leaves every fund
        with the years ``carry_forward`` gives it.
        """
        table = self.layout.spec
        frame = _drop_blank_rows(self.raw_sheet(table), table)

        draws_column = find_column(frame, "draws", table=table, required=False)
        if draws_column is None:
            return {}

        name_column = find_column(frame, "fund_name", table=table)
        names = _read_text_column(frame[name_column], table=table, column="fund_name")

        # One cell per fund, in years since inception. A blank cell is no plan at all.
        plans_in_years_since_inception = {}
        for name, cell in zip(names, frame[draws_column]):
            plan = parse_draw_plan(cell, fund_name=name)

            if plan is not None:
                plans_in_years_since_inception[name] = plan

        return relative_draw_plans_to_calendar(
            plans_in_years_since_inception,
            inception_year=self.inception_year,
        )

    def expected_returns(self) -> pd.Series:
        """The Liquid Spec sheet.

        Required: the Commitments sheet is a pacing schedule, and means nothing without its X.
        """
        # The sheet goes by several names; the layout lists them, or pins one.
        if isinstance(self.layout.liquid_spec, str):
            names = (self.layout.liquid_spec,)
        else:
            names = tuple(self.layout.liquid_spec)

        for name in names:
            sheet = self.raw_sheet(name, required=False)
            if sheet is not None:
                return normalize_expected_returns(sheet)

        names_tried = " or ".join(repr(name) for name in names)
        raise ValueError(f"{self.path.name}: no sheet named {names_tried}; sheets are {self.sheet_names}")

    @cached_property
    def expected_return(self) -> float:
        """This profile's X: the yearly return the pacing model assumed for its liquid portfolio."""
        table = self.expected_returns()
        matches = _columns_matching(list(table.index), self.profile)

        if not matches:
            raise ValueError(
                f"expected returns: no row for portfolio {self.profile!r}; portfolios are {list(table.index)}"
            )

        return float(table[matches[0]])

    # ------------------------------------------------------------------- spec
    def simulation_spec(self, initial_value: float, **overrides: Any) -> SimulationSpec:
        """The ``SimulationSpec`` for this profile.

        Returns compounded from ``initial_value``, the FX rate inverted when it is quoted as
        dollars per unit of base currency, and the profile's expected return.

        Commitments are rounded as the spreadsheet rounds them: to one ten-thousandth of
        ``initial_value``, which is ``ROUND(value, -4)`` when that is 100,000,000. Pass
        ``commitment_rounding_unit_usd`` to choose another unit, or None to round nothing.
        """
        settings: dict[str, Any] = {
            "base_currency": self.currency,
            "liquid_series": self.liquid_column,
            "liquid_kind": "returns",
            "initial_value": initial_value,
            "fx_series": self.fx_column,
            "fx_quote": self.fx_quote,
            "expected_return": self.expected_return,
            "commitment_rounding_unit_usd": commitment_rounding_unit(initial_value),
        }

        # Anything the caller passes wins over the profile's own settings.
        settings.update(overrides)
        return SimulationSpec(**settings)

    def __repr__(self) -> str:
        return f"WorkbookRepository({str(self.path)!r}, profile={self.profile!r})"
