"""The five-sheet portfolio workbook: profile selection, returns → levels, relative-year schedule, FX inversion."""
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
    commitment_schedule,
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
def test_repository_reads_the_five_sheets_and_selects_the_profile(usd, eur):
    assert usd.sheet_names == ["Liquid", "FX", "Flows", "Commitments", "Spec"]
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
    assert list(market.columns) == ["USD Conservative", "USD Moderate", "USD Aggressive", "EUR Conservative", "EURUSD", "GBPUSD"]
    assert market.index[0] == pd.Timestamp("2009-04-30") and market.index[-1] == pd.Timestamp("2012-12-31")
    assert len(market) == 45 and market.index.name == "date"
    assert market["USD Conservative"].iloc[0] == pytest.approx(0.004)  # k = 0: sin(0) = 0


def test_commitment_schedule_maps_relative_years_onto_the_calendar(usd, eur):
    rates = usd.commitment_rates()
    assert list(rates.index) == list(range(2009, 2020)) and list(rates.columns) == ["BUYOUT", "SECONDARIES"]
    assert rates.loc[2009].tolist() == [0.0, 0.0]  # relative year 0 = inception year 2009
    assert rates.loc[2010, "BUYOUT"] == 0.022 and rates.loc[2019, "SECONDARIES"] == 0.008
    assert eur.commitment_rates().loc[2010, "BUYOUT"] == 0.020  # a different profile, different rates
    raw = usd.sheet("Commitments")
    with pytest.raises(ValueError, match=r"no rows for profile 'USD' 'Wild'; profiles are \['EUR Conservative', 'USD Conservative', 'USD Moderate'\]"):
        commitment_schedule(raw, currency="USD", risk="Wild", inception_year=2009)
    doubled = pd.concat([raw, raw.iloc[[1]]])
    with pytest.raises(ValueError, match="more than one rate for 'SECONDARIES' in relative year 1"):
        commitment_schedule(doubled, currency="USD", risk="Conservative", inception_year=2009)
    gap = raw[~((raw["Type"] == "BUYOUT") & (raw["Year"] == 3) & (raw["Currency"] == "USD") & (raw["Risk"] == "Conservative"))]
    with pytest.raises(ValueError, match=r"no rate for relative year\(s\) \[\(3, 'BUYOUT'\)\]"):
        commitment_schedule(gap, currency="USD", risk="Conservative", inception_year=2009)


# ------------------------------------------------------- returns and levels
def test_returns_to_levels_compounds_from_the_initial_value():
    returns = pd.Series([0.05, 0.10, -0.50], index=pd.to_datetime(["2027-01-31", "2027-02-28", "2027-03-31"]))
    levels = returns_to_levels(returns, 1000.0)
    np.testing.assert_allclose(levels, [1000.0, 1100.0, 550.0])  # the first return is not applied
    with pytest.raises(ValueError, match="greater than -100%"):
        returns_to_levels(pd.Series([0.0, -1.0]), 1.0)
    for bad in (None, 0, -1, float("nan"), True):
        with pytest.raises(ValueError, match="initial_value"):
            SimulationSpec("USD", "x", liquid_kind="returns", initial_value=bad)
    with pytest.raises(ValueError, match="liquid_kind must be one of"):
        SimulationSpec("USD", "x", liquid_kind="prices")


def test_profile_spec_builds_levels_and_inverts_fx(usd, eur):
    spec = usd.spec(1_000_000)
    assert (spec.base_currency, spec.liquid_series, spec.liquid_kind, spec.initial_value, spec.fx_series) == \
        ("USD", "USD Conservative", "returns", 1_000_000, None)
    portfolio = Orchestrator(usd, spec).portfolio
    returns = usd.market_data()["USD Conservative"]
    np.testing.assert_allclose(portfolio.liquid_levels.to_numpy(),
                               1_000_000 * np.cumprod(np.r_[1.0, 1.0 + returns.to_numpy()[1:]]))
    assert portfolio.usd_rate is None
    eur_portfolio = Orchestrator(eur, eur.spec(2_000_000)).portfolio
    np.testing.assert_allclose(eur_portfolio.usd_rate.to_numpy(), 1.0 / eur.market_data()["EURUSD"].to_numpy())
    assert eur_portfolio.base_currency == "EUR" and eur_portfolio.liquid_levels.iloc[0] == 2_000_000
    overridden = usd.spec(1_000_000, carry_forward=True, stop_on_shortfall=False)
    assert overridden.carry_forward and not overridden.stop_on_shortfall


# ----------------------------------------------------------------- end to end
def test_usd_conservative_runs_end_to_end(workbook):
    orchestrator = load_profile_workbook(workbook, "USD", "Conservative", 1_000_000)
    result = orchestrator.run()
    assert result.status == "completed" and result.base_currency == "USD"
    assert result.beyond_horizon == ("SEC_VII",)
    c = result.commitments
    assert [fund for _, fund in c.index] == ["PEM2011", "SEC_VI", "PEM2012", "PEM2013"]
    pem2011 = c.loc[(pd.Timestamp("2010-12-31"), "PEM2011")]
    level = orchestrator.portfolio.liquid_levels.loc["2010-12-31"]
    assert pem2011["policy_year"] == 2010 and pem2011["rate"] == 0.022  # relative year 1 of the schedule
    assert pem2011["sizing_base"] == pytest.approx(level) and pem2011["commitment_usd"] == pytest.approx(0.022 * level)
    assert c.loc[(pd.Timestamp("2011-12-31"), "SEC_VI"), "rate"] == 0.008
    assert c.loc[(pd.Timestamp("2011-12-31"), "PEM2012"), "rate"] == 0.022  # same date, different type: no weights needed
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


def test_layout_override_and_missing_sheets(tmp_path):
    path = tmp_path / "renamed.xlsx"
    tables = sample_tables()
    with pd.ExcelWriter(path) as writer:
        tables["Liquid"].to_excel(writer, sheet_name="Returns")
        tables["FX"].to_excel(writer, sheet_name="Spot")
        tables["Flows"].to_excel(writer, sheet_name="Fund Data", index=False)
        tables["Commitments"].to_excel(writer, sheet_name="Schedule", index=False)
        tables["Spec"].to_excel(writer, sheet_name="Funds", index=False)
    layout = SheetLayout(liquid="returns", fx="spot", flows="fund-data", commitments="schedule", spec="funds")
    result = run_profile_workbook(path, "USD", "Conservative", 1_000_000, layout=layout)
    assert result.status == "completed"
    with pytest.raises(ValueError, match=r"no sheet named 'Liquid'; sheets are \['Returns'"):
        WorkbookRepository(path, "USD", "Conservative").liquid_column
    with pytest.raises(FileNotFoundError):
        WorkbookRepository(tmp_path / "missing.xlsx", "USD", "Conservative")
    with pytest.raises(ValueError, match="currency must be a code"):
        WorkbookRepository(path, "", "Conservative")
