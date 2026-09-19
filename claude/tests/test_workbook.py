"""The portfolio workbook: profile selection, returns → levels, relative-year schedule, FX inversion, expected returns."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("openpyxl")

from examples.profile_workbook import FUNDS, LIQUID_SPEC, sample_tables, write_sample_workbook  # noqa: E402
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
    assert set(market["fund_name"]) == {name for name, _, _ in FUNDS} and len(market) == 577
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
    assert pem2011["policy_year"] == 2010 and pem2011["current_year_rate"] == 0.022  # relative year 1 of the schedule
    assert pem2011["expected_value"] == 1.0 and pem2011["carried_usd"] == 0  # the first commitment: the model is worth 1 here
    assert pem2011["sizing_base"] == pytest.approx(level) and pem2011["commitment_usd"] == pytest.approx(0.022 * level)
    assert orchestrator.simulator.first_commitment_date == date(2010, 12, 31) and orchestrator.expected_return == 0.054

    # the sample's schedule grows at X and is then divided by X, so every commitment is the same share of the value
    rate_of = c.reset_index().groupby("fund_type")["rate"]
    np.testing.assert_allclose(rate_of.get_group("BUYOUT"), 0.022, rtol=1e-4)
    np.testing.assert_allclose(rate_of.get_group("SECONDARIES"), 0.008, rtol=1e-4)
    pem2012 = c.loc[(pd.Timestamp("2011-12-31"), "PEM2012")]  # same date as SEC_VI, different type: no weights needed
    assert pem2012["current_year_rate"] == pytest.approx(0.022 * 1.054) and pem2012["expected_value"] == pytest.approx(1.054)
    assert c.loc[(pd.Timestamp("2011-12-31"), "SEC_VI"), "carried_usd"] == 0  # carry-forward off by default: 2010's budget is lost
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


def test_eur_moderate_script_starts_from_dollars_converted_at_the_first_rate(workbook, tmp_path, capsys):
    from examples import eur_moderate
    orchestrator, result = eur_moderate.run(workbook, start_usd=100.0, out_dir=tmp_path / "out")
    eurusd = sample_tables()["FX"].loc["2009-04-30", "EURUSD"]
    assert orchestrator.repository.profile == "EUR Moderate"
    assert orchestrator.spec.initial_value == pytest.approx(100.0 / eurusd)
    assert result.base_currency == "EUR" and result.status == "completed"
    assert result.periods["liquid_open"].iloc[0] == pytest.approx(100.0 / eurusd)
    assert orchestrator.policy.entitlements["PEM2011"].current_year_rate == 0.030  # EUR Moderate's year-1 BUYOUT rate
    # the script switches carry-forward on: no secondaries fund closes in 2010, so SEC_VI collects 2010's budget in 2011
    assert orchestrator.spec.carry_forward is True
    sec_vi = result.commitments.loc[(pd.Timestamp("2011-12-31"), "SEC_VI")]
    balance_at_end_of_2010 = result.periods.loc[pd.Timestamp("2010-12-31"), "liquid_only_usd"]
    # 2010's budget was sized on 31 Dec 2010, the first commitment date, where the expected value is 1; 2011's a year on, at 1.059
    assert sec_vi["carried_years"] == "2010" and sec_vi["carried_usd"] == pytest.approx(0.011 * balance_at_end_of_2010)
    assert sec_vi["expected_value"] == pytest.approx(1.059) and orchestrator.expected_return == 0.059  # EUR Moderate's ExRet
    assert sec_vi["commitment_usd"] == pytest.approx(sec_vi["carried_usd"] + 0.011 * sec_vi["sizing_base_usd"])
    _, without = eur_moderate.run(workbook, start_usd=100.0, carry_forward=False)
    assert without.commitments.loc[(pd.Timestamp("2011-12-31"), "SEC_VI"), "commitment_usd"] == \
        pytest.approx(0.011 * sec_vi["sizing_base_usd"])
    assert {p.name for p in (tmp_path / "out").iterdir()} == {
        "periods.csv", "funds.csv", "commitments.csv", "map_events_to_observations.csv", "fund_summary.csv",
        "tracked_values.csv", "liquid_only_comparison.csv", "public_market_equivalent.csv"}
    assert "Starting balance: USD 100.00 = EUR" in capsys.readouterr().out
    _, in_euros = eur_moderate.run(workbook, start_usd=100.0, start_in_base_currency=True)
    assert in_euros.periods["liquid_open"].iloc[0] == 100.0
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
