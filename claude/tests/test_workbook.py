"""The portfolio workbook: profile selection, returns → levels, relative-year schedule, FX inversion, expected returns."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("openpyxl")

from examples.profile_workbook import FUNDS, LIQUID_SPEC, sample_tables, write_sample_workbook  # noqa: E402
from pmsim import round_like_excel  # noqa: E402
from pmsim.data import (  # noqa: E402
    Orchestrator,
    SheetLayout,
    SimulationSpec,
    WorkbookRepository,
    calendar_rates_for_profile,
    infer_inception_date,
    load_profile_workbook,
    returns_to_levels,
    run_profile_workbook,
)
from tests.conftest import check_identities  # noqa: E402


@pytest.fixture(scope="module")
def workbook(tmp_path_factory):
    return write_sample_workbook(tmp_path_factory.mktemp("book") / "portfolio.xlsx")


@pytest.fixture
def usd(workbook):
    return WorkbookRepository(workbook, "USD", "Conservative")


@pytest.fixture
def eur(workbook):
    return WorkbookRepository(workbook, "eur", " Conservative ")


# --------------------------------------------------------------- the sheets
def test_repository_reads_the_sheets_and_selects_the_profile(usd, eur):
    assert usd.sheet_names == ["Liquid", "Liquid Spec", "FX", "Flows", "Commitments", "Spec"]
    assert usd.profile == "USD Conservative" and usd.liquid_column == "USD Conservative"
    assert usd.fx_column is None and usd.inception_year == 2009
    assert eur.profile == "EUR Conservative" and eur.fx_column == "EURUSD" and eur.fx_quote == "usd_per_base"
    gbp = WorkbookRepository(usd.path, "GBP", "Moderate")
    assert gbp.liquid_column == "GBP Moderate" and gbp.fx_column == "GBPUSD"
    with pytest.raises(ValueError, match=r"Liquid: no column for profile 'USD Balanced'; profiles are \['USD Conservative'"):
        WorkbookRepository(usd.path, "USD", "Balanced").liquid_column
    with pytest.raises(ValueError, match="FX: no column CHFUSD or USDCHF"):
        WorkbookRepository(usd.path, "CHF", "Conservative").fx_column


def test_spec_sheet_gives_fund_specs_with_day_first_closing_dates(usd):
    specs = usd.fund_specs()
    assert len(specs) == 19 and specs["fund_name"].tolist()[:3] == ["PEM2011", "SEC_VI", "PEM2012"]
    assert specs["fund_type"].tolist()[:3] == ["BUYOUT", "SECONDARIES", "BUYOUT"]
    assert specs["closing_date"].tolist()[:2] == [date(2010, 12, 31), date(2011, 12, 31)]  # "31/12/2010" read day-first
    closings = dict(zip(specs["fund_name"], specs["closing_date"]))
    assert closings["PEM2017"] == date(2016, 12, 30)  # "30/12/2016": there is no 30th month, so only day-first reads it
    assert closings["SEC_X"] == date(2025, 12, 31)


def test_flows_sheet_uses_vintage_as_the_fund_name(usd):
    market = usd.fund_market_data()
    assert set(market["fund_name"]) == {name for name, _, _, _ in FUNDS} and len(market) == 577
    pem = market[market["fund_name"] == "PEM2011"]
    # each year of a fund's life: its calls, then the mark, then that year's distribution once the harvest starts
    assert pem["kind"].tolist()[:6] == ["flow", "flow", "nav", "flow", "flow", "nav"]
    assert pem["unit"].iloc[0] == pytest.approx(-0.15) and pem["unit"].iloc[2] == pytest.approx(0.235)


def test_market_data_joins_liquid_returns_with_fx(usd):
    market = usd.market_data()
    assert list(market.columns) == list(LIQUID_SPEC) + ["EURUSD", "GBPUSD"]
    assert market.index[0] == pd.Timestamp("2009-04-30") and market.index[-1] == pd.Timestamp("2026-12-31")
    assert len(market) == 213 and market.index.name == "date"


def test_commitment_schedule_maps_relative_years_onto_the_calendar(usd, eur):
    rates = usd.commitment_rates()
    assert list(rates.index) == list(range(2009, 2030)) and list(rates.columns) == ["BUYOUT", "SECONDARIES"]
    assert rates.loc[2009].tolist() == [0.0, 0.0]  # relative year 0 = inception year 2009
    assert rates.loc[2010, "BUYOUT"] == 0.022  # relative year 1
    # the sample's schedule grows at the profile's own expected return, so the share it stands for is the same each year
    assert rates.loc[2011, "BUYOUT"] == pytest.approx(0.022 * 1.054) and usd.expected_return == 0.054
    assert eur.commitment_rates().loc[2011, "BUYOUT"] == pytest.approx(0.022 * 1.047)  # another profile, another X
    raw = usd.raw_sheet("Commitments")
    with pytest.raises(ValueError, match=r"no rows for profile 'USD' 'Wild'; profiles are \['EUR Aggressive', 'EUR Conservative'"):
        calendar_rates_for_profile(raw, currency="USD", risk="Wild", inception_year=2009)
    doubled = pd.concat([raw, raw.iloc[[1]]])
    with pytest.raises(ValueError, match="more than one rate for 'BUYOUT' in relative year 1"):
        calendar_rates_for_profile(doubled, currency="USD", risk="Conservative", inception_year=2009)
    gap = raw[~((raw["Type"] == "BUYOUT") & (raw["Year"] == 3) & (raw["Currency"] == "USD") & (raw["Risk"] == "Conservative"))]
    with pytest.raises(ValueError, match=r"no rate for relative year\(s\) \[\(3, 'BUYOUT'\)\]"):
        calendar_rates_for_profile(gap, currency="USD", risk="Conservative", inception_year=2009)


# ------------------------------------------------------- returns and levels
def test_returns_to_levels_starts_one_period_before_the_first_return():
    month_ends = pd.to_datetime(["2027-01-31", "2027-02-28", "2027-03-31"])
    levels = returns_to_levels(pd.Series([0.05, 0.10, -0.50], index=month_ends), 1000.0)
    assert list(levels.index) == [pd.Timestamp("2026-12-31"), *month_ends]  # a month back; 31 Dec 2026 is a Thursday
    np.testing.assert_allclose(levels, [1000.0, 1050.0, 1155.0, 577.5])  # every return applied
    with pytest.raises(ValueError, match="greater than -100%"):
        returns_to_levels(pd.Series([0.0, -1.0, 0.0], index=month_ends), 1.0)


def test_inception_date_follows_the_inferred_frequency():
    weekend = pd.to_datetime(["2027-02-28", "2027-03-31", "2027-04-30"])
    assert infer_inception_date(weekend) == pd.Timestamp("2027-01-29")  # 31 Jan 2027 is a Sunday: back to the Friday
    assert infer_inception_date(pd.to_datetime(["2027-06-30", "2027-09-30", "2027-12-31"])) == pd.Timestamp("2027-03-31")
    assert infer_inception_date(pd.bdate_range("2027-01-04", periods=4)) == pd.Timestamp("2027-01-01")  # Monday: back to Friday
    assert infer_inception_date(pd.date_range("2027-01-08", periods=3, freq="W-FRI")) == pd.Timestamp("2027-01-01")
    irregular = pd.Series([0.1, 0.1, 0.1], index=pd.to_datetime(["2027-01-05", "2027-02-17", "2027-05-30"]))
    with pytest.raises(ValueError, match="cannot infer the frequency"):
        returns_to_levels(irregular, 1.0)
    with pytest.raises(ValueError, match="at least three"):
        returns_to_levels(irregular.iloc[:2], 1.0)
    explicit = returns_to_levels(irregular, 1.0, inception_date="2027-01-01")  # the escape hatch
    assert explicit.index[0] == pd.Timestamp("2027-01-01") and explicit.iloc[1] == pytest.approx(1.1)
    with pytest.raises(ValueError, match="must be before the first return"):
        returns_to_levels(irregular, 1.0, inception_date="2027-01-05")
    with pytest.raises(ValueError, match="inception_date only applies"):
        SimulationSpec("USD", "x", inception_date="2027-01-01")


def test_initial_value_validation():
    for bad in (None, 0, -1, float("nan"), True, "100"):
        with pytest.raises(ValueError, match="initial_value"):
            SimulationSpec("USD", "x", liquid_kind="returns", initial_value=bad)
    for good in (100, 100.0, np.int64(100), np.float64(100.0)):  # numpy scalars from a DataFrame are numbers too
        assert SimulationSpec("USD", "x", liquid_kind="returns", initial_value=good).initial_value == 100
    with pytest.raises(ValueError, match="liquid_kind must be one of"):
        SimulationSpec("USD", "x", liquid_kind="prices")


def test_profile_spec_builds_levels_and_inverts_fx(usd, eur):
    spec = usd.simulation_spec(1_000_000)
    assert (spec.base_currency, spec.liquid_series, spec.liquid_kind, spec.initial_value, spec.fx_series) == \
        ("USD", "USD Conservative", "returns", 1_000_000, None)
    portfolio = Orchestrator(usd, spec).portfolio
    returns = usd.market_data()["USD Conservative"]
    levels = portfolio.liquid_levels
    assert levels.index[0] == pd.Timestamp("2009-03-31") and levels.index[1] == pd.Timestamp("2009-04-30")  # inception: a month back
    np.testing.assert_allclose(levels.to_numpy(), 1_000_000 * np.cumprod(np.r_[1.0, 1.0 + returns.to_numpy()]))
    assert portfolio.usd_rate is None
    eur_portfolio = Orchestrator(eur, eur.simulation_spec(2_000_000)).portfolio
    rate = eur_portfolio.usd_rate
    assert rate.index[0] == pd.Timestamp("2009-03-31") and rate.iloc[0] == rate.iloc[1]  # first known rate applies at inception
    np.testing.assert_allclose(rate.to_numpy()[1:], 1.0 / eur.market_data()["EURUSD"].to_numpy())
    assert eur_portfolio.base_currency == "EUR" and eur_portfolio.liquid_levels.iloc[0] == 2_000_000
    overridden = usd.simulation_spec(1_000_000, carry_forward=True, stop_on_shortfall=False)
    assert overridden.carry_forward and not overridden.stop_on_shortfall


# ----------------------------------------------------------------- end to end
def test_usd_conservative_runs_end_to_end(workbook):
    orchestrator = load_profile_workbook(workbook, "USD", "Conservative", 1_000_000)
    result = orchestrator.run()
    assert result.status == "completed" and result.base_currency == "USD"
    assert result.funds_beyond_horizon == ()  # the liquid series runs to 2026, past every closing in the Spec sheet
    c = result.commitments
    assert len(c) == 19 and [fund for _, fund in c.index][:3] == ["PEM2011", "SEC_VI", "PEM2012"]

    pem2011 = c.loc[(pd.Timestamp("2010-12-31"), "PEM2011")]
    level = orchestrator.portfolio.liquid_levels.loc["2010-12-31"]  # a USD profile: the liquid-only value is the levels
    assert pem2011["policy_year"] == 2010 and pem2011["own_year_rate"] == 0.022  # relative year 1 of the schedule
    assert pem2011["expected_value"] == 1.0 and pem2011["other_years_usd"] == 0  # the first commitment: the model is worth 1 here
    # from 1,000,000 the commitments are rounded to the nearest 100: ROUND(value, -4) on 100,000,000, scaled
    assert orchestrator.spec.commitment_rounding_unit_usd == 100.0
    assert pem2011["sizing_base"] == pytest.approx(level)
    assert pem2011["commitment_usd"] == round_like_excel(0.022 * level, 100.0)
    assert orchestrator.simulator.first_commitment_date == date(2010, 12, 31) and orchestrator.expected_return == 0.054

    # the sample's schedule grows at X and is then divided by X, so before rounding every drawn
    # year is the same share of the value that funds it: 2.2% for buyout, 0.8% for secondaries
    draws = result.draws.reset_index()
    share = draws["year_budget_unrounded_usd"] / draws["liquid_only_usd"]
    np.testing.assert_allclose(share[draws["fund_type"] == "BUYOUT"], 0.022, rtol=1e-4)
    np.testing.assert_allclose(share[draws["fund_type"] == "SECONDARIES"], 0.008, rtol=1e-4)
    # and rounding moves each of them by half a unit at most
    assert ((draws["year_budget_usd"] - draws["year_budget_unrounded_usd"]).abs() <= 50.0).all()
    pem2012 = c.loc[(pd.Timestamp("2011-12-31"), "PEM2012")]  # same date as SEC_VI, different type: no weights needed
    assert pem2012["own_year_rate"] == pytest.approx(0.022 * 1.054) and pem2012["expected_value"] == pytest.approx(1.054)
    # SEC_VI's plan is years 1-4: 2010 funded at its own year end, 2011-2013 at the closing
    sec_vi = c.loc[(pd.Timestamp("2011-12-31"), "SEC_VI")]
    assert sec_vi["drawn_years"] == "2010, 2011, 2012, 2013" and sec_vi["other_years_usd"] > 0
    assert c.loc[(pd.Timestamp("2016-12-31"), "PEM2017"), "policy_year"] == 2016  # closed 30 Dec, observed on the 31st

    # PEM2011's first call, 14 March 2011, pooled onto the March month end and scaled by its commitment
    march = result.funds.loc[(pd.Timestamp("2011-03-31"), "PEM2011")]
    assert march["calls_usd"] == pytest.approx(0.15 * pem2011["commitment_usd"])
    summary = orchestrator.fund_summary()
    assert summary.loc["PEM2011", "marks"] == 13 and summary.loc["PEM2011", "unit_called"] == pytest.approx(0.95)
    check_identities(result)


def test_eur_conservative_translates_at_the_inverted_rate(workbook):
    result = run_profile_workbook(workbook, "EUR", "Conservative", 1_000_000)
    assert result.status == "completed" and result.base_currency == "EUR"
    row = result.commitments.loc[(pd.Timestamp("2010-12-31"), "PEM2011")]
    eurusd = sample_tables()["FX"].loc["2010-12-31", "EURUSD"]
    assert row["usd_rate"] == pytest.approx(1 / eurusd)
    assert row["commitment_usd"] == pytest.approx(row["commitment_base"] * eurusd)
    assert (result.periods["fx_translation"] != 0).any()
    check_identities(result)


def test_eur_moderate_script_defaults_to_a_hundred_million_euros_with_rounded_commitments(workbook):
    from examples import eur_moderate
    assert eur_moderate.START_VALUE == 100_000_000 and eur_moderate.START_IN_BASE_CURRENCY is True
    orchestrator, result = eur_moderate.run(workbook)
    assert result.base_currency == "EUR" and result.status == "completed"
    assert result.periods["liquid_open"].iloc[0] == 100_000_000.0  # euros, used as they are: no conversion
    # ROUND(value, -4): every year's commitment, and so every fund's, is a whole number of 10,000 dollars
    assert orchestrator.spec.commitment_rounding_unit_usd == 10_000.0
    assert (result.draws["year_budget_usd"] % 10_000 == 0).all()
    assert (result.commitments["commitment_usd"] % 10_000 == 0).all()
    sec_ix = result.draws.xs("SEC_IX", level="fund").iloc[0]  # 12x3: three times one rounded year, not the rounding of three
    assert sec_ix["commitment_usd"] == 3 * round_like_excel(sec_ix["year_budget_unrounded_usd"], 10_000.0)
    check_identities(result)


def test_eur_moderate_script_starts_from_dollars_converted_at_the_first_rate(workbook, tmp_path, capsys):
    from examples import eur_moderate
    # the draw-plan arithmetic below is exact only before rounding, so rounding is switched off here
    from_dollars = dict(start_value=100.0, start_in_base_currency=False, round_commitments=False)
    orchestrator, result = eur_moderate.run(workbook, out_dir=tmp_path / "out", **from_dollars)
    eurusd = sample_tables()["FX"].loc["2009-04-30", "EURUSD"]
    assert orchestrator.repository.profile == "EUR Moderate"
    assert orchestrator.spec.initial_value == pytest.approx(100.0 / eurusd)
    assert result.base_currency == "EUR" and result.status == "completed"
    assert result.periods["liquid_open"].iloc[0] == pytest.approx(100.0 / eurusd)
    assert orchestrator.policy.rates.at[2010, "BUYOUT"] == 0.030  # EUR Moderate's year-1 BUYOUT rate
    assert orchestrator.expected_return == 0.059  # EUR Moderate's ExRet

    # SEC_VI's Draws cell says years 1-4. It closes in year 2, so 2010 is funded from its own year
    # end — the first commitment date, where the expected value is 1 — and 2011, 2012 and 2013 from
    # the closing. The schedule grows at X, so every one of them is the same 1.1% share.
    sec_vi = result.commitments.loc[(pd.Timestamp("2011-12-31"), "SEC_VI")]
    balance_at_end_of_2010 = result.periods.loc[pd.Timestamp("2010-12-31"), "liquid_only_usd"]
    assert sec_vi["drawn_years"] == "2010, 2011, 2012, 2013"
    assert sec_vi["expected_value"] == pytest.approx(1.059)
    assert sec_vi["own_year_usd"] == pytest.approx(0.011 * sec_vi["sizing_base_usd"], rel=1e-3)
    assert sec_vi["commitment_usd"] == pytest.approx(
        0.011 * balance_at_end_of_2010 + 3 * 0.011 * sec_vi["sizing_base_usd"], rel=1e-3)

    # with the plans switched off, the script's carry-forward switch decides the years again:
    # no secondaries fund closes in 2010, so SEC_VI collects that year's budget and no more
    assert orchestrator.spec.carry_forward is True
    _, carried = eur_moderate.run(workbook, draws={}, **from_dollars)
    row = carried.commitments.loc[(pd.Timestamp("2011-12-31"), "SEC_VI")]
    assert row["drawn_years"] == "2010, 2011"
    assert row["commitment_usd"] == pytest.approx(0.011 * balance_at_end_of_2010 + 0.011 * row["sizing_base_usd"], rel=1e-3)
    _, without = eur_moderate.run(workbook, carry_forward=False, draws={}, **from_dollars)
    assert without.commitments.loc[(pd.Timestamp("2011-12-31"), "SEC_VI"), "commitment_usd"] == \
        pytest.approx(0.011 * row["sizing_base_usd"], rel=1e-3)
    assert {p.name for p in (tmp_path / "out").iterdir()} == {
        "periods.csv", "funds.csv", "commitments.csv", "draws.csv", "map_events_to_observations.csv",
        "fund_summary.csv", "tracked_values.csv", "liquid_only_comparison.csv", "public_market_equivalent.csv"}
    assert "Starting balance: USD 100.00 = EUR" in capsys.readouterr().out
    _, in_euros = eur_moderate.run(workbook, start_value=100.0, start_in_base_currency=True)
    assert in_euros.periods["liquid_open"].iloc[0] == 100.0
    assert (in_euros.draws["year_budget_usd"].mul(100).round(6) % 1 == 0).all()  # from 100: whole cents, ROUND(value, 2)
    check_identities(result)


def test_january_start_puts_inception_in_the_previous_year_without_needing_its_rate(tmp_path):
    tables = sample_tables()
    shifted = pd.date_range("2010-01-31", periods=len(tables["Liquid"]), freq="ME")
    tables["Liquid"].index = shifted
    tables["FX"].index = shifted
    path = tmp_path / "january.xlsx"
    with pd.ExcelWriter(path) as writer:
        tables["Liquid"].to_excel(writer, sheet_name="Liquid")
        tables["FX"].to_excel(writer, sheet_name="FX")
        for sheet in ("Liquid Spec", "Flows", "Commitments", "Spec"):
            tables[sheet].to_excel(writer, sheet_name=sheet, index=False)
    orchestrator = load_profile_workbook(path, "EUR", "Conservative", 1_000_000)
    assert orchestrator.repository.inception_year == 2010  # schedule Year 0: the first Liquid year, not the inception row's
    assert orchestrator.portfolio.first_date == date(2009, 12, 31)  # a Thursday
    assert list(orchestrator.repository.commitment_rates().index)[:2] == [2010, 2011]
    result = orchestrator.run()
    assert result.status == "completed" and result.periods.index[0] == pd.Timestamp("2009-12-31")
    assert result.periods["usd_rate"].iloc[0] == result.periods["usd_rate"].iloc[1]  # first known rate at inception
    check_identities(result)


def test_date_column_need_not_be_first(tmp_path):
    path = tmp_path / "date_second.xlsx"
    tables = sample_tables()
    liquid = tables["Liquid"].reset_index().rename(columns={"index": "Date"})
    profiles = [c for c in liquid.columns if c != "Date"]
    liquid = liquid[[profiles[0], "Date"] + profiles[1:]]  # the date column second, not first
    fx = tables["FX"].reset_index().rename(columns={"index": "Observation Date"})[["EURUSD", "GBPUSD", "Observation Date"]]
    with pd.ExcelWriter(path) as writer:
        liquid.to_excel(writer, sheet_name="Liquid", index=False)
        fx.to_excel(writer, sheet_name="FX", index=False)
        for sheet in ("Liquid Spec", "Flows", "Commitments", "Spec"):
            tables[sheet].to_excel(writer, sheet_name=sheet, index=False)
    repository = WorkbookRepository(path, "EUR", "Conservative")
    assert repository.inception_year == 2009 and repository.liquid_column == "EUR Conservative"
    assert repository.fx_column == "EURUSD"
    market = repository.market_data()
    assert list(market.columns) == list(LIQUID_SPEC) + ["EURUSD", "GBPUSD"]
    pd.testing.assert_frame_equal(market, WorkbookRepository(write_sample_workbook(tmp_path / "usual.xlsx"),
                                                             "EUR", "Conservative").market_data())
    check_identities(Orchestrator(repository, repository.simulation_spec(1_000_000)).run())


def test_liquid_spec_sheet_gives_each_portfolio_its_expected_return(usd, eur, tmp_path):
    table = usd.expected_returns()  # Liquid | ExRet, one row per portfolio
    assert table.to_dict() == {"USD Conservative": 0.054, "USD Moderate": 0.064, "USD Aggressive": 0.074,
                               "EUR Conservative": 0.047, "EUR Moderate": 0.059, "EUR Aggressive": 0.070,
                               "GBP Conservative": 0.054, "GBP Moderate": 0.064, "GBP Aggressive": 0.075}
    assert table.index.name == "portfolio" and table.name == "expected_return"
    assert usd.expected_return == 0.054 and eur.expected_return == 0.047  # only this profile's row is used
    assert usd.simulation_spec(100).expected_return == 0.054 and usd.simulation_spec(100, expected_return=0.07).expected_return == 0.07

    def workbook_with(liquid_spec, sheet_name="Liquid Spec"):
        path = tmp_path / f"{sheet_name or 'none'}-{len(list(tmp_path.iterdir()))}.xlsx"
        tables = sample_tables()
        with pd.ExcelWriter(path) as writer:
            tables["Liquid"].to_excel(writer, sheet_name="Liquid")
            tables["FX"].to_excel(writer, sheet_name="FX")
            for sheet in ("Flows", "Commitments", "Spec"):
                tables[sheet].to_excel(writer, sheet_name=sheet, index=False)
            if liquid_spec is not None:
                liquid_spec.to_excel(writer, sheet_name=sheet_name, index=False)
        return path

    # the sheet name and both column names are matched loosely, so the older Portfolio / Expected Return spelling still reads
    aliased = workbook_with(pd.DataFrame({"portfolio name": ["usd conservative"], "Expected Return": [0.03]}), sheet_name="LiquidSpec")
    assert WorkbookRepository(aliased, "USD", "Conservative").expected_return == 0.03
    typed_as_text = workbook_with(pd.DataFrame({"Liquid": ["USD Conservative"], "ExRet": ["5.4%"]}))
    assert WorkbookRepository(typed_as_text, "USD", "Conservative").expected_return == pytest.approx(0.054)
    # the schedule means nothing without the return it assumed, so the sheet is required
    with pytest.raises(ValueError, match=r"no sheet named 'Liquid Spec' or 'Return Spec' or 'Expected Returns'"):
        load_profile_workbook(workbook_with(None), "USD", "Conservative", 1_000_000)
    # the sheet is read under any of the names it has gone by, so a renamed tab keeps working
    for sheet_name in ("Liquid Spec", "Return Spec", "Expected Returns", "returnspec"):
        renamed = workbook_with(pd.DataFrame({"Liquid": ["USD Conservative"], "Return": [0.03]}), sheet_name=sheet_name)
        assert WorkbookRepository(renamed, "USD", "Conservative").expected_return == 0.03
    pinned = SheetLayout(liquid_spec="Return Spec")  # or pin exactly one
    assert WorkbookRepository(workbook_with(pd.DataFrame({"Liquid": ["USD Conservative"], "ExRet": [0.03]}),
                                            sheet_name="Return Spec"), "USD", "Conservative", pinned).expected_return == 0.03
    with pytest.raises(ValueError, match=r"expected returns: no row for portfolio 'EUR Moderate'; portfolios are \['USD Conservative'\]"):
        load_profile_workbook(workbook_with(pd.DataFrame({"Liquid": ["USD Conservative"], "ExRet": [0.054]})),
                              "EUR", "Moderate", 1_000_000)
    with pytest.raises(ValueError, match=r"expected return of 'USD Conservative' must be a decimal .* \(write 5% as 0.05\), got 5.4"):
        load_profile_workbook(workbook_with(pd.DataFrame({"Liquid": ["USD Conservative"], "ExRet": [5.4]})),
                              "USD", "Conservative", 1_000_000)
    with pytest.raises(ValueError, match=r"each portfolio needs one expected return; duplicated: \['USD Conservative'\]"):
        load_profile_workbook(workbook_with(pd.DataFrame({"Liquid": ["USD Conservative"] * 2, "ExRet": [0.054, 0.064]})),
                              "USD", "Conservative", 1_000_000)


def test_layout_override_and_missing_sheets(tmp_path):
    path = tmp_path / "renamed.xlsx"
    tables = sample_tables()
    with pd.ExcelWriter(path) as writer:
        tables["Liquid"].to_excel(writer, sheet_name="Returns")
        tables["FX"].to_excel(writer, sheet_name="Spot")
        tables["Flows"].to_excel(writer, sheet_name="Fund Data", index=False)
        tables["Commitments"].to_excel(writer, sheet_name="Schedule", index=False)
        tables["Spec"].to_excel(writer, sheet_name="Funds", index=False)
        tables["Liquid Spec"].to_excel(writer, sheet_name="PacingAssumptions", index=False)
    layout = SheetLayout(liquid="returns", fx="spot", flows="fund-data", commitments="schedule", spec="funds",
                         liquid_spec="pacing assumptions")
    result = run_profile_workbook(path, "USD", "Conservative", 1_000_000, layout=layout)
    assert result.status == "completed"
    with pytest.raises(ValueError, match=r"no sheet named 'Liquid'; sheets are \['Returns'"):
        WorkbookRepository(path, "USD", "Conservative").liquid_column
    with pytest.raises(FileNotFoundError):
        WorkbookRepository(tmp_path / "missing.xlsx", "USD", "Conservative")
    with pytest.raises(ValueError, match="currency must be a code"):
        WorkbookRepository(path, "", "Conservative")


# ------------------------------------------------- the Spec sheet's Draws column
SECONDARIES_DRAWS = {  # the sample workbook's plan, in relative years
    "SEC_VI": "1-4", "SEC_VII": "5-8", "SEC_VIII": "9-11", "SEC_IX": "12x3", "SEC_X": "16x3",
}


def test_the_spec_sheets_draws_column_is_read_onto_calendar_years(workbook):
    plans = WorkbookRepository(workbook, "USD", "Conservative").draw_plans()
    assert set(plans) == set(SECONDARIES_DRAWS)  # only the secondaries name years; the buyout cells are blank
    assert plans["SEC_VI"] == {2010: 1.0, 2011: 1.0, 2012: 1.0, 2013: 1.0}  # relative 1-4, inception 2009
    assert plans["SEC_VIII"] == {2018: 1.0, 2019: 1.0, 2020: 1.0}
    assert plans["SEC_IX"] == {2021: 3.0} and plans["SEC_X"] == {2025: 3.0}
    assert [name for name, _, fund_type, draws in FUNDS if draws] == list(SECONDARIES_DRAWS)


def test_a_workbook_without_a_draws_column_leaves_every_fund_on_its_default_years(workbook, tmp_path):
    tables = sample_tables()
    tables["Spec"] = tables["Spec"].drop(columns=["Draws"])
    path = tmp_path / "no-draws.xlsx"
    with pd.ExcelWriter(path) as writer:
        for sheet, frame in tables.items():
            frame.to_excel(writer, sheet_name=sheet, index=sheet in ("Liquid", "FX"))
    repository = WorkbookRepository(path, "USD", "Conservative")
    assert repository.draw_plans() == {}
    policy = Orchestrator(repository, repository.simulation_spec(initial_value=100.0)).policy
    assert policy.entitlements["SEC_VI"].draws == {2011: 1.0}  # its own closing year, nothing else


def test_the_secondaries_draw_four_vintages_then_three_times_one_year(workbook):
    # exact shares are asserted below, so nothing is rounded here; the rounding has its own tests
    orchestrator = load_profile_workbook(workbook, "USD", "Conservative", 100.0, commitment_rounding_unit_usd=None)
    result = orchestrator.run()
    c = result.commitments.reset_index().set_index("fund")
    secondaries = c[c["fund_type"] == "SECONDARIES"]

    assert secondaries.loc["SEC_VI", "drawn_years"] == "2010, 2011, 2012, 2013"
    assert secondaries.loc["SEC_IX", "drawn_years"] == "2021x3"
    # the sample's schedule grows at X, so under this convention each drawn year is the same 0.8%
    # share of whatever value funds it: four years for SEC_VI, three years' worth for SEC_IX
    assert secondaries.loc["SEC_IX", "rate"] == pytest.approx(3 * 0.008, rel=1e-3)
    assert secondaries.loc["SEC_VIII", "rate"] == pytest.approx(3 * 0.008, rel=1e-3)
    # SEC_VI draws one year at end-2010 and three at end-2011, so its share spans two balances
    assert secondaries.loc["SEC_VI", "commitment_usd"] == pytest.approx(2.930, abs=5e-4)
    assert secondaries["commitment_usd"].sum() == pytest.approx(20.495, abs=5e-4)

    # relative years 13, 14, 15 and 17 onwards are drawn by nobody: the plan stops at year 16
    assert orchestrator.policy.unclaimed_schedule_years()["SECONDARIES"] == [2022, 2023, 2024, 2026, 2027, 2028, 2029]
    check_identities(result)


def test_every_drawn_year_is_funded_from_a_date_no_later_than_the_closing(workbook):
    """The no-look-ahead guarantee, over the whole sample run."""
    result = run_profile_workbook(workbook, "EUR", "Moderate", 100.0)
    draws = result.draws.reset_index()
    assert (draws["funding_date"] <= draws["date"]).all()
    # a forward draw is exactly a row whose plan date is later than the date funding it
    forward = draws[draws["plan_date"] > draws["funding_date"]]
    assert set(forward["fund"]) == {"SEC_VI", "SEC_VII", "SEC_VIII"}  # the three that reach past their closing
    assert (forward["funding_date"] == forward["date"]).all()  # and all of them funded at the closing

    # the dollars of a fund's drawn years add up to its commitment
    totals = draws.groupby(["date", "fund"])["commitment_usd"].sum().reindex(result.commitments.index)
    np.testing.assert_allclose(totals, result.commitments["commitment_usd"])


# ------------------------------------------------------ rounding the commitments
def test_the_workbook_rounds_commitments_like_excel_round_at_any_starting_value(workbook):
    """ROUND(value, -4) on 100,000,000, and the same relative precision everywhere else."""
    repository = WorkbookRepository(workbook, "USD", "Conservative")
    assert repository.simulation_spec(100_000_000).commitment_rounding_unit_usd == 10_000.0  # ROUND(value, -4)
    assert repository.simulation_spec(1_000_000).commitment_rounding_unit_usd == 100.0       # ROUND(value, -2)
    assert repository.simulation_spec(100).commitment_rounding_unit_usd == 0.01              # ROUND(value, 2)
    assert repository.simulation_spec(100, commitment_rounding_unit_usd=None).commitment_rounding_unit_usd is None

    large = run_profile_workbook(workbook, "USD", "Conservative", 100_000_000)
    small = run_profile_workbook(workbook, "USD", "Conservative", 100)
    assert (large.draws["year_budget_usd"] % 10_000 == 0).all() and (large.commitments["commitment_usd"] % 10_000 == 0).all()
    # the same relative precision: the run from 100 is the run from 100,000,000 divided by a million, rounding included
    np.testing.assert_allclose(small.commitments["commitment_usd"] * 1_000_000, large.commitments["commitment_usd"])
    np.testing.assert_allclose(small.periods["total_close"] * 1_000_000, large.periods["total_close"])
    check_identities(large)


def test_each_drawn_year_is_rounded_before_its_multiplier_and_the_years_are_then_added(workbook):
    rounded = run_profile_workbook(workbook, "USD", "Conservative", 100_000_000)
    exact = run_profile_workbook(workbook, "USD", "Conservative", 100_000_000, commitment_rounding_unit_usd=None)
    draws = rounded.draws
    # every year's budget is the Excel rounding of the unrounded one, which is what the unrounded run computes
    np.testing.assert_allclose(draws["year_budget_unrounded_usd"], exact.draws["year_budget_usd"])
    expected = [round_like_excel(value, 10_000.0) for value in draws["year_budget_unrounded_usd"]]
    assert draws["year_budget_usd"].tolist() == expected
    assert draws["commitment_usd"].tolist() == (draws["multiplier"] * draws["year_budget_usd"]).tolist()
    # SEC_VI draws four years: four rounded amounts added, which need not be the rounding of their sum
    sec_vi = draws.xs("SEC_VI", level="fund")
    assert len(sec_vi) == 4 and rounded.commitments.xs("SEC_VI", level="fund")["commitment_usd"].iloc[0] == sec_vi["year_budget_usd"].sum()
