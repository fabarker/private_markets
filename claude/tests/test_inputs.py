from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from pmsim import Fund, Portfolio
from pmsim.inputs import coerce_rate_table

D = pd.Timestamp


# ------------------------------------------------------------------- Fund
def test_fund_accepts_pairs_mapping_and_series_alike():
    pairs = Fund("A", "BUYOUT", "2027-01-01", unit_calls=[("2027-02-01", 0.2), ("2027-03-01", 0.1)])
    mapping = Fund("A", "BUYOUT", "2027-01-01", unit_calls={"2027-02-01": 0.2, date(2027, 3, 1): 0.1})
    series = Fund("A", "BUYOUT", "2027-01-01", unit_calls=pd.Series([0.1, 0.2], index=pd.to_datetime(["2027-03-01", "2027-02-01"])))
    for fund in (mapping, series):
        pd.testing.assert_series_equal(fund.unit_calls, pairs.unit_calls)
    assert list(pairs.unit_calls.index) == [D("2027-02-01"), D("2027-03-01")]  # sorted
    assert pairs.unit_distributions.empty and pairs.unit_nav.empty
    assert pairs.closing_date == date(2027, 1, 1) and pairs.closing_year == 2027


def test_fund_copies_its_inputs():
    calls = pd.Series([0.2], index=pd.to_datetime(["2027-02-01"]))
    fund = Fund("A", "BUYOUT", "2027-01-01", unit_calls=calls)
    calls.iloc[0] = 99.0
    assert fund.unit_calls.iloc[0] == 0.2 and fund.unit_calls is not calls


def test_same_day_flows_are_summed_but_same_day_marks_are_rejected():
    fund = Fund("A", "BUYOUT", "2027-01-01", unit_calls=[("2027-02-01", 0.2), ("2027-02-01", 0.1)])
    assert fund.unit_calls.tolist() == [pytest.approx(0.3)]
    with pytest.raises(ValueError, match="more than one entry on 2027-02-01"):
        Fund("A", "BUYOUT", "2027-01-01", unit_nav=[("2027-02-01", 0.5), ("2027-02-01", 0.6)])


@pytest.mark.parametrize("field", ["unit_calls", "unit_distributions", "unit_nav"])
@pytest.mark.parametrize("value, message", [(-0.1, "negative"), (float("nan"), "finite"), (float("inf"), "finite"),
                                            ("abc", "not a number"), (None, "not a number"), (True, "boolean")])
def test_fund_rejects_bad_values(field, value, message):
    with pytest.raises(ValueError, match=message):
        Fund("A", "BUYOUT", "2027-01-01", **{field: [("2027-02-01", value)]})


@pytest.mark.parametrize("field", ["unit_calls", "unit_distributions", "unit_nav"])
def test_fund_rejects_history_before_its_closing(field):
    with pytest.raises(ValueError, match="2026-12-31 precedes the closing on 2027-01-01"):
        Fund("A", "BUYOUT", "2027-01-01", **{field: [("2026-12-31", 0.1)]})
    Fund("A", "BUYOUT", "2027-01-01", **{field: [("2027-01-01", 0.1)]})  # same day is fine


@pytest.mark.parametrize("bad", [datetime(2027, 1, 1, 12), pd.Timestamp("2027-01-01 00:00:01"),
                                 pd.Timestamp("2027-01-01", tz="UTC"), "not a date", 2027, None])
def test_fund_rejects_ambiguous_dates(bad):
    with pytest.raises(ValueError):
        Fund("A", "BUYOUT", bad)
    with pytest.raises(ValueError):
        Fund("A", "BUYOUT", "2027-01-01", unit_calls=[(bad, 0.1)])


def test_fund_accepts_midnight_datetimes_and_strips_names():
    fund = Fund(" A ", " BUYOUT ", datetime(2027, 1, 1), unit_calls=[(pd.Timestamp("2027-02-01"), 0.1)])
    assert (fund.name, fund.fund_type, fund.closing_date) == ("A", "BUYOUT", date(2027, 1, 1))


@pytest.mark.parametrize("name, fund_type", [("", "BUYOUT"), ("A", ""), (None, "BUYOUT"), ("A", 3)])
def test_fund_requires_names(name, fund_type):
    with pytest.raises(ValueError, match="non-empty string"):
        Fund(name, fund_type, "2027-01-01")


def test_fund_events_merge_flows_and_marks_by_day():
    fund = Fund("A", "BUYOUT", "2027-01-01", unit_calls=[("2027-02-01", 0.2)],
                unit_distributions=[("2027-02-01", 0.05), ("2027-03-01", 0.1)], unit_nav=[("2027-03-01", 0.4)])
    assert fund.events_by_day() == [(D("2027-02-01"), 0.2, 0.05, None), (D("2027-03-01"), 0.0, 0.1, 0.4)]


# --------------------------------------------------------------- Portfolio
LEVELS = [("2027-01-01", 100.0), ("2027-06-30", 110.0)]
RATES = {"BUYOUT": {2027: 0.1}}


def test_base_currency_decides_whether_a_rate_is_needed():
    usd = Portfolio("usd", LEVELS, RATES)
    assert usd.base_currency == "USD" and usd.usd_rate is None and not usd.requires_fx_conversion
    gbp = Portfolio("GBP", LEVELS, RATES, usd_rate=[("2027-01-01", 0.8)])
    assert gbp.requires_fx_conversion and gbp.usd_rate.tolist() == [0.8]
    with pytest.raises(ValueError, match="usd_rate is required: base_currency 'GBP' is not USD"):
        Portfolio("GBP", LEVELS, RATES)
    with pytest.raises(ValueError, match="usd_rate must be omitted when base_currency is USD"):
        Portfolio("USD", LEVELS, RATES, usd_rate=[("2027-01-01", 0.8)])


@pytest.mark.parametrize("rate", [[], [("2027-01-01", 0.0)], [("2027-01-01", -1.0)]])
def test_usd_rate_must_be_positive(rate):
    with pytest.raises(ValueError, match="strictly positive"):
        Portfolio("EUR", LEVELS, RATES, usd_rate=rate)


@pytest.mark.parametrize("levels, message", [
    ([], "at least one observation"),
    ([("2027-01-01", 0.0)], "strictly positive"),
    ([("2027-01-01", 100.0), ("2027-01-01", 101.0)], "more than one entry"),
    ([("2027-01-01", 100.0), ("2027-02-01", float("nan"))], "finite"),
])
def test_liquid_levels_validation(levels, message):
    with pytest.raises(ValueError, match=message):
        Portfolio("USD", levels, RATES)


def test_portfolio_dates_and_years():
    portfolio = Portfolio("USD", [("2027-06-30", 100.0), ("2027-01-01", 90.0), ("2029-01-31", 120.0)], {"BUYOUT": {2027: 0, 2028: 0, 2029: 0.1}})
    assert portfolio.first_date == date(2027, 1, 1) and portfolio.last_date == date(2029, 1, 31)
    assert list(portfolio.calendar_years) == [2027, 2028, 2029]
    assert portfolio.liquid_levels.tolist() == [90.0, 100.0, 120.0]


def test_base_currency_must_be_a_code():
    with pytest.raises(ValueError, match="currency code"):
        Portfolio("", LEVELS, RATES)


# -------------------------------------------------------------- rate table
def test_rate_table_from_mapping_and_frame():
    table = coerce_rate_table({"BUYOUT": {2028: 0.08, 2027: 0.1}, "VC": {2027: 0.0, 2028: 0.02}})
    assert list(table.index) == [2027, 2028] and list(table.columns) == ["BUYOUT", "VC"]
    assert table.index.name == "year" and table.columns.name == "fund_type"
    assert table.at[2028, "BUYOUT"] == 0.08 and table.dtypes.eq(float).all()
    frame = pd.DataFrame({"BUYOUT": [0.1]}, index=[np.int64(2027)])
    assert coerce_rate_table(frame).at[2027, "BUYOUT"] == 0.1
    assert coerce_rate_table(pd.DataFrame()).empty


@pytest.mark.parametrize("value, message", [
    ({"BUYOUT": {2027: 0.1, 2029: 0.1}}, "every calendar year from 2027 to 2029"),
    ({"BUYOUT": {2027: -0.1}}, "non-negative"),
    ({"BUYOUT": {2027: float("nan")}}, "finite"),
    ({"BUYOUT": {"2027": 0.1}}, "integer calendar years"),
    ({"BUYOUT": {2027: "x"}}, "numeric"),
    ({"": {2027: 0.1}}, "non-empty fund-type strings"),
    (pd.DataFrame({"BUYOUT": [0.1]}), "outside 1900..9999"),  # the RangeIndex year 0 mistake
])
def test_rate_table_validation(value, message):
    with pytest.raises(ValueError, match=message):
        coerce_rate_table(value)
