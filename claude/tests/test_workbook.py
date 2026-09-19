"""The portfolio workbook: profile selection, returns → levels, relative-year schedule, FX inversion, expected returns."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("openpyxl")

from examples.profile_workbook import sample_tables, write_sample_workbook  # noqa: E402
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
    assert usd.sheet_names == ["Liquid", "FX", "Flows", "Commitments", "Spec", "Expected Returns"]
    assert usd.profile == "USD Conservative" and usd.liquid_column == "USD Conservative"
    assert usd.fx_column is None and usd.inception_year == 2009
    assert eur.profile == "EUR Conservative" and eur.fx_column == "EURUSD" and eur.fx_quote == "usd_per_base"
    with pytest.raises(ValueError, match=r"Liquid: no column for profile 'GBP Conservative'; profiles are \['USD Conservative'"):
        WorkbookRepository(usd.path, "GBP", "Conservative").liquid_column
    with pytest.raises(ValueError, match="FX: no column CHFUSD or USDCHF"):
        WorkbookRepository(usd.path, "CHF", "Conservative").fx_column


def test_spec_sheet_gives_fund_specs_with_day_first_closing_dates(usd):
    specs = usd.fund_specs()
    assert specs["fund_name"].tolist() == ["PEM2011", "SEC_VI", "PEM2012", "PEM2013", "SEC_VII"]
    assert specs["fund_type"].tolist() == ["BUYOUT", "SECONDARIES", "BUYOUT", "BUYOUT", "SECONDARIES"]
    assert specs["closing_date"].tolist()[:2] == [date(2010, 12, 31), date(2011, 12, 31)]  # "31/12/2010" read day-first


def test_flows_sheet_uses_vintage_as_the_fund_name(usd):
    market = usd.fund_market_data()
    assert set(market["fund_name"]) == {"PEM2011", "SEC_VI"}
    pem = market[market["fund_name"] == "PEM2011"]
    assert pem["kind"].tolist() == ["flow"] * 4 + ["nav"] + ["flow"] * 2 + ["nav"]
    assert pem["unit"].iloc[0] == pytest.approx(-0.025) and pem["unit"].iloc[4] == pytest.approx(0.045)


def test_market_data_joins_liquid_returns_with_fx(usd):
    market = usd.market_data()
    assert list(market.columns) == ["USD Conservative", "USD Moderate", "USD Aggressive", "EUR Conservative",
                                    "EUR Moderate", "EURUSD", "GBPUSD"]
    assert market.index[0] == pd.Timestamp("2009-04-30") and market.index[-1] == pd.Timestamp("2012-12-31")
    assert len(market) == 45 and market.index.name == "date"
    assert market["USD Conservative"].iloc[0] == pytest.approx(0.004)  # k = 0: sin(0) = 0


def test_commitment_schedule_maps_relative_years_onto_the_calendar(usd, eur):
    rates = usd.commitment_rates()
    assert list(rates.index) == list(range(2009, 2020)) and list(rates.columns) == ["BUYOUT", "SECONDARIES"]
    assert rates.loc[2009].tolist() == [0.0, 0.0]  # relative year 0 = inception year 2009
    assert rates.loc[2010, "BUYOUT"] == 0.022 and rates.loc[2019, "SECONDARIES"] == 0.008
    assert eur.commitment_rates().loc[2010, "BUYOUT"] == 0.020  # a different profile, different rates
    raw = usd.raw_sheet("Commitments")
    with pytest.raises(ValueError, match=r"no rows for profile 'USD' 'Wild'; profiles are \['EUR Conservative', 'EUR Moderate', 'USD Conservative', 'USD Moderate'\]"):
        calendar_rates_for_profile(raw, currency="USD", risk="Wild", inception_year=2009)
    doubled = pd.concat([raw, raw.iloc[[1]]])
    with pytest.raises(ValueError, match="more than one rate for 'SECONDARIES' in relative year 1"):
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
    assert result.funds_beyond_horizon == ("SEC_VII",)
    c = result.commitments
    assert [fund for _, fund in c.index] == ["PEM2011", "SEC_VI", "PEM2012", "PEM2013"]
    pem2011 = c.loc[(pd.Timestamp("2010-12-31"), "PEM2011")]
    level = orchestrator.portfolio.liquid_levels.loc["2010-12-31"]
    assert pem2011["policy_year"] == 2010 and pem2011["current_year_rate"] == 0.022  # relative year 1 of the schedule
    assert pem2011["rate"] == pytest.approx(0.022) and pem2011["carried_usd"] == 0
    assert pem2011["sizing_base"] == pytest.approx(level) and pem2011["commitment_usd"] == pytest.approx(0.022 * level)
    assert c.loc[(pd.Timestamp("2011-12-31"), "SEC_VI"), "current_year_rate"] == 0.008
    assert c.loc[(pd.Timestamp("2011-12-31"), "SEC_VI"), "carried_usd"] == 0  # carry-forward is off by default: 2010's budget is lost
    pem2012 = c.loc[(pd.Timestamp("2011-12-31"), "PEM2012")]
    assert pem2012["current_year_rate"] == 0.022  # same date, different type: no weights needed
    # the pacing model's value is 1 when PEM2011 is committed; a year later it expects 1.04 (this profile's X is 4%)
    assert orchestrator.simulator.first_commitment_date == date(2010, 12, 31) and orchestrator.expected_return == 0.04
    assert pem2011["expected_value"] == 1.0 and pem2012["expected_value"] == pytest.approx(1.04)
    assert pem2012["commitment_usd"] == pytest.approx(0.022 / 1.04 * pem2012["sizing_base_usd"])
    levels_usd = orchestrator.portfolio.liquid_levels  # a USD profile: the liquid-only value is the level series itself
    assert pem2012["sizing_base_usd"] == pytest.approx(levels_usd.loc["2011-12-31"])
    # PEM2011's June 2011 call pooled onto the June month end, scaled by its commitment
    june = result.funds.loc[(pd.Timestamp("2011-06-30"), "PEM2011")]
    assert june["calls_usd"] == pytest.approx(0.025 * pem2011["commitment_usd"])
    summary = orchestrator.fund_summary()
    assert summary.loc["PEM2011", "marks"] == 2 and summary.loc["SEC_VII", "beyond_horizon"]
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
    assert orchestrator.policy.entitlements["PEM2011"].current_year_rate == 0.026  # EUR Moderate's BUYOUT rate
    # the script switches carry-forward on: no secondaries fund closes in 2010, so SEC_VI collects 2010's budget in 2011
    assert orchestrator.spec.carry_forward is True
    sec_vi = result.commitments.loc[(pd.Timestamp("2011-12-31"), "SEC_VI")]
    balance_at_end_of_2010 = result.periods.loc[pd.Timestamp("2010-12-31"), "sizing_base_usd"]
    # 2010's budget was sized on 31 Dec 2010, the first commitment date, where the expected value is 1; 2011's a year on, at 1.045
    assert sec_vi["carried_years"] == "2010" and sec_vi["carried_usd"] == pytest.approx(0.009 * balance_at_end_of_2010)
    assert sec_vi["expected_value"] == pytest.approx(1.045) and orchestrator.expected_return == 0.045
    assert sec_vi["commitment_usd"] == pytest.approx(sec_vi["carried_usd"] + 0.009 / 1.045 * sec_vi["sizing_base_usd"])
    _, without = eur_moderate.run(workbook, start_usd=100.0, carry_forward=False)
    assert without.commitments.loc[(pd.Timestamp("2011-12-31"), "SEC_VI"), "commitment_usd"] == \
        pytest.approx(0.009 / 1.045 * sec_vi["sizing_base_usd"])
    assert {p.name for p in (tmp_path / "out").iterdir()} == {
        "periods.csv", "funds.csv", "commitments.csv", "map_events_to_observations.csv", "fund_summary.csv",
        "liquid_only_comparison.csv", "public_market_equivalent.csv"}
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
        for sheet in ("Flows", "Commitments", "Spec", "Expected Returns"):
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
    liquid = liquid[["USD Conservative", "Date", "USD Moderate", "USD Aggressive", "EUR Conservative", "EUR Moderate"]]
    fx = tables["FX"].reset_index().rename(columns={"index": "Observation Date"})[["EURUSD", "GBPUSD", "Observation Date"]]
    with pd.ExcelWriter(path) as writer:
        liquid.to_excel(writer, sheet_name="Liquid", index=False)
        fx.to_excel(writer, sheet_name="FX", index=False)
        for sheet in ("Flows", "Commitments", "Spec", "Expected Returns"):
            tables[sheet].to_excel(writer, sheet_name=sheet, index=False)
    repository = WorkbookRepository(path, "EUR", "Conservative")
    assert repository.inception_year == 2009 and repository.liquid_column == "EUR Conservative"
    assert repository.fx_column == "EURUSD"
    market = repository.market_data()
    assert list(market.columns) == ["USD Conservative", "USD Moderate", "USD Aggressive", "EUR Conservative",
                                    "EUR Moderate", "EURUSD", "GBPUSD"]
    pd.testing.assert_frame_equal(market, WorkbookRepository(write_sample_workbook(tmp_path / "usual.xlsx"),
                                                             "EUR", "Conservative").market_data())
    check_identities(Orchestrator(repository, repository.simulation_spec(1_000_000)).run())


def test_expected_returns_sheet_gives_each_portfolio_its_x(usd, eur, tmp_path):
    table = usd.expected_returns()
    assert table.to_dict() == {"USD Conservative": 0.04, "USD Moderate": 0.05, "USD Aggressive": 0.06,
                               "EUR Conservative": 0.035, "EUR Moderate": 0.045}
    assert table.index.name == "portfolio" and table.name == "expected_return"
    assert usd.expected_return == 0.04 and eur.expected_return == 0.035
    assert usd.simulation_spec(100).expected_return == 0.04 and usd.simulation_spec(100, expected_return=0.07).expected_return == 0.07

    def workbook_with(expected_returns, sheet_name="Expected Returns"):
        path = tmp_path / f"{sheet_name or 'none'}-{len(list(tmp_path.iterdir()))}.xlsx"
        tables = sample_tables()
        with pd.ExcelWriter(path) as writer:
            tables["Liquid"].to_excel(writer, sheet_name="Liquid")
            tables["FX"].to_excel(writer, sheet_name="FX")
            for sheet in ("Flows", "Commitments", "Spec"):
                tables[sheet].to_excel(writer, sheet_name=sheet, index=False)
            if expected_returns is not None:
                expected_returns.to_excel(writer, sheet_name=sheet_name, index=False)
        return path

    squashed = workbook_with(pd.DataFrame({"portfolio name": ["usd conservative"], "X": [0.03]}), sheet_name="ExpectedReturns")
    assert WorkbookRepository(squashed, "USD", "Conservative").expected_return == 0.03  # sheet, columns and name matched loosely
    # the schedule means nothing without the return it assumed, so the sheet is required
    with pytest.raises(ValueError, match=r"no sheet named 'Expected Returns'"):
        load_profile_workbook(workbook_with(None), "USD", "Conservative", 1_000_000)
    with pytest.raises(ValueError, match=r"Expected Returns: no row for portfolio 'EUR Moderate'; portfolios are \['USD Conservative'\]"):
        load_profile_workbook(workbook_with(pd.DataFrame({"Portfolio": ["USD Conservative"], "Expected Return": [0.04]})),
                              "EUR", "Moderate", 1_000_000)
    with pytest.raises(ValueError, match=r"expected return of 'USD Conservative' must be a decimal .* \(write 5% as 0.05\), got 4.0"):
        load_profile_workbook(workbook_with(pd.DataFrame({"Portfolio": ["USD Conservative"], "Expected Return": [4]})),
                              "USD", "Conservative", 1_000_000)
    with pytest.raises(ValueError, match=r"each portfolio needs one expected return; duplicated: \['USD Conservative'\]"):
        load_profile_workbook(workbook_with(pd.DataFrame({"Portfolio": ["USD Conservative"] * 2, "Expected Return": [0.04, 0.05]})),
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
        tables["Expected Returns"].to_excel(writer, sheet_name="PacingAssumptions", index=False)
    layout = SheetLayout(liquid="returns", fx="spot", flows="fund-data", commitments="schedule", spec="funds",
                         expected_returns="pacing assumptions")
    result = run_profile_workbook(path, "USD", "Conservative", 1_000_000, layout=layout)
    assert result.status == "completed"
    with pytest.raises(ValueError, match=r"no sheet named 'Liquid'; sheets are \['Returns'"):
        WorkbookRepository(path, "USD", "Conservative").liquid_column
    with pytest.raises(FileNotFoundError):
        WorkbookRepository(tmp_path / "missing.xlsx", "USD", "Conservative")
    with pytest.raises(ValueError, match="currency must be a code"):
        WorkbookRepository(path, "", "Conservative")
