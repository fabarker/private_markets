"""Data layer: normalized tables, the in-memory repository, and the orchestrator."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from pmsim import STARTING_VALUE, AnnualRatePolicy, Portfolio, Simulator
from pmsim.data import FrameRepository, Orchestrator, SimulationSpec, build_funds, build_portfolio
from pmsim.data.tables import (
    normalize_commitment_rates,
    normalize_fund_market_data,
    normalize_fund_specs,
    normalize_market_data,
    parse_draw_plan,
    relative_draw_plans_to_calendar,
)
from tests.conftest import check_identities

SPEC = SimulationSpec("GBP", liquid_series="liquid_gbp", fx_series="gbp_per_usd", weights={"A": 0.6, "B": 0.4})


def tables():
    """The worked example as workbook-style tables. Flows are dollars; scale is the fund's commitment."""
    fund_spec = pd.DataFrame({"fund_name": ["A", "B"], "type": ["BUYOUT", "BUYOUT"],
                              "closing_date": ["2027-02-15", "2027-05-10"]})
    fund_market = pd.DataFrame([
        ("A", "Flow", -250_000.0, "2027-03-01", 1_000_000),
        ("A", "NAV", 250_000.0, "2027-03-31", 1_000_000),
        ("A", "Flow", 50_000.0, "2027-06-01", 1_000_000),
        ("B", "Flow", -250_000.0, "2027-05-20", 1_000_000),
    ], columns=["fund_name", "type", "value", "date", "scale"])
    market = pd.DataFrame({"date": ["2027-01-01", "2027-03-31", "2027-06-30"],
                           "liquid_gbp": [1e6, 1.1e6, 1.21e6], "gbp_per_usd": [0.8, 0.8, 0.75]})
    rates = pd.DataFrame({"year": [2027], "BUYOUT": [0.1]})
    return fund_spec, fund_market, market, rates


def started_at_the_starting_value(portfolio):
    """The same portfolio as the data layer always builds it: its levels rescaled to start at 100,000,000."""
    levels = portfolio.liquid_levels * (STARTING_VALUE / portfolio.liquid_levels.iloc[0])
    usd_rate = portfolio.usd_rate if portfolio.requires_fx_conversion else None
    return Portfolio(portfolio.base_currency, levels, portfolio.commitment_rates, usd_rate=usd_rate)


def expected(gbp_portfolio, worked_funds):
    """The worked example run on the engine directly, from the 100,000,000 every assembled run starts with."""
    portfolio = started_at_the_starting_value(gbp_portfolio)
    policy = AnnualRatePolicy(portfolio.commitment_rates, worked_funds, {"A": 0.6, "B": 0.4})
    return Simulator(portfolio, worked_funds, policy).run()


# ------------------------------------------------------------ end to end
def test_tables_reproduce_the_worked_example(gbp_portfolio, worked_funds):
    result = Orchestrator(FrameRepository(*tables()), SPEC).run()
    want = expected(gbp_portfolio, worked_funds)
    pd.testing.assert_frame_equal(result.periods, want.periods)
    pd.testing.assert_frame_equal(result.funds, want.funds)
    pd.testing.assert_frame_equal(result.commitments, want.commitments)
    assert result.periods["liquid_open"].iloc[0] == 100_000_000.0  # the tables quote 1,000,000; the run starts at 100,000,000
    assert result.periods["total_close"].iloc[-1] == pytest.approx(120_731_875.0)  # the worked example's 1,207,318.75, × 100
    check_identities(result)


def test_column_aliases_are_case_and_space_insensitive(gbp_portfolio, worked_funds):
    fund_spec, fund_market, market, rates = tables()
    fund_spec.columns = ["Fund Name", "Strategy", "Closing Date"]
    fund_market.columns = ["Fund", "Entry Type", "Amount", "Date", "Divisor"]
    market.columns = ["Date", "Liquid GBP", "GBP-per-USD"]
    rates.columns = ["Year", "BUYOUT"]
    spec = SimulationSpec("gbp", liquid_series="liquid gbp", fx_series="gbp per usd", weights={"A": 0.6, "B": 0.4})
    result = Orchestrator(FrameRepository(fund_spec, fund_market, market, rates), spec).run()
    pd.testing.assert_frame_equal(result.periods, expected(gbp_portfolio, worked_funds).periods)


def test_orchestrator_exposes_the_assembled_pieces():
    orchestrator = Orchestrator(FrameRepository(*tables()), SPEC)
    assert [f.name for f in orchestrator.funds] == ["A", "B"]
    assert orchestrator.portfolio.base_currency == "GBP" and orchestrator.portfolio.requires_fx_conversion
    assert orchestrator.policy.entitlements["A"].weight == 0.6 and orchestrator.policy.rates.at[2027, "BUYOUT"] == 0.1
    assert orchestrator.simulator.timeline.n_observations == 3
    assert orchestrator.map_events_to_observations().loc[("A", pd.Timestamp("2027-03-01")), "observation_date"] == pd.Timestamp("2027-03-31")
    with pytest.raises(TypeError, match="SimulationSpec"):
        Orchestrator(FrameRepository(*tables()), {"base_currency": "GBP"})


def test_fund_summary():
    summary = Orchestrator(FrameRepository(*tables()), SPEC).fund_summary()
    assert list(summary.index) == ["A", "B"]
    a = summary.loc["A"]
    assert (a["fund_type"], a["calls"], a["distributions"], a["marks"]) == ("BUYOUT", 1, 1, 1)
    assert a["unit_called"] == pytest.approx(0.25) and a["unit_distributed"] == pytest.approx(0.05)
    assert a["latest_unit_nav"] == pytest.approx(0.25) and a["closing_date"] == pd.Timestamp("2027-02-15")
    assert a["first_event"] == pd.Timestamp("2027-03-01") and a["last_event"] == pd.Timestamp("2027-06-01")
    assert not a["beyond_horizon"] and np.isnan(summary.loc["B", "latest_unit_nav"])


# --------------------------------------------------------- fund histories
def test_flow_sign_convention_and_explicit_kinds():
    specs = normalize_fund_specs(pd.DataFrame({"fund_name": ["F"], "type": ["VC"], "closing_date": ["2027-01-01"]}))
    market = normalize_fund_market_data(pd.DataFrame([
        ("F", "flow", -20.0, "2027-02-01", 100), ("F", "flow", 5.0, "2027-03-01", 100),
        ("F", "Call", -10.0, "2027-04-01", 100), ("F", "Distribution", -2.0, "2027-05-01", 100),
        ("F", "flow", 0.0, "2027-06-01", 100), ("F", "NAV", 30.0, "2027-06-30", 100),
    ], columns=["fund_name", "type", "value", "date", "scale"]))
    [fund] = build_funds(specs, market)
    assert fund.unit_calls.to_dict() == {pd.Timestamp("2027-02-01"): 0.2, pd.Timestamp("2027-04-01"): 0.1}
    assert fund.unit_distributions.to_dict() == {pd.Timestamp("2027-03-01"): 0.05, pd.Timestamp("2027-05-01"): 0.02}
    assert fund.unit_nav.to_dict() == {pd.Timestamp("2027-06-30"): 0.3}
    [flipped] = build_funds(specs, market, calls_are_negative=False)
    assert flipped.unit_calls.to_dict() == {pd.Timestamp("2027-03-01"): 0.05, pd.Timestamp("2027-04-01"): 0.1}
    assert flipped.unit_distributions.to_dict() == {pd.Timestamp("2027-02-01"): 0.2, pd.Timestamp("2027-05-01"): 0.02}


def test_scale_divides_and_must_be_positive():
    rows = pd.DataFrame([("F", "flow", -250_000.0, "2027-02-01", 1_000_000)], columns=["fund_name", "type", "value", "date", "scale"])
    assert normalize_fund_market_data(rows)["unit"].tolist() == [-0.25]
    for bad in (0, -1, None, "x"):
        rows["scale"] = [bad]
        with pytest.raises(ValueError, match="scale"):
            normalize_fund_market_data(rows)


@pytest.mark.parametrize("kind", ["Foo", "", None, 3])
def test_unknown_kind_is_an_error(kind):
    rows = pd.DataFrame([("F", kind, 1.0, "2027-02-01", 1)], columns=["fund_name", "type", "value", "date", "scale"])
    with pytest.raises(ValueError, match="expected Flow, NAV, Call or Distribution"):
        normalize_fund_market_data(rows)


def test_market_rows_for_unknown_funds_fail_and_spec_only_funds_are_empty():
    fund_spec, fund_market, market, rates = tables()
    fund_spec = pd.concat([fund_spec, pd.DataFrame({"fund_name": ["C"], "type": ["VC"], "closing_date": ["2027-06-30"]})])
    rates["VC"] = [0.0]
    funds = Orchestrator(FrameRepository(fund_spec, fund_market, market, rates), SPEC).funds
    assert [f.name for f in funds] == ["A", "B", "C"] and funds[2].unit_calls.empty
    stray = pd.concat([fund_market, pd.DataFrame([("Z", "Flow", -1.0, "2027-02-01", 1)], columns=fund_market.columns)])
    with pytest.raises(ValueError, match=r"missing from fund_spec: \['Z'\]"):
        FrameRepository(fund_spec, stray, market, rates) and Orchestrator(FrameRepository(fund_spec, stray, market, rates), SPEC).funds


def test_fund_spec_validation():
    with pytest.raises(ValueError, match=r"duplicated: \['A'\]"):
        normalize_fund_specs(pd.DataFrame({"fund_name": ["A", "A"], "type": ["X", "X"], "closing_date": ["2027-01-01"] * 2}))
    with pytest.raises(ValueError, match=r"no column for 'closing_date'; columns are \['fund_name', 'type'\]"):
        normalize_fund_specs(pd.DataFrame({"fund_name": ["A"], "type": ["X"]}))
    with pytest.raises(ValueError, match="fund_type is missing in row 1"):
        normalize_fund_specs(pd.DataFrame({"fund_name": ["A"], "type": [None], "closing_date": ["2027-01-01"]}))
    with pytest.raises(ValueError, match="closing_date in row 1"):
        normalize_fund_specs(pd.DataFrame({"fund_name": ["A"], "type": ["X"], "closing_date": ["soon"]}))
    with pytest.raises(ValueError, match="both look like|all look like"):
        normalize_fund_specs(pd.DataFrame({"fund_name": ["A"], "fund": ["A"], "type": ["X"], "closing_date": ["2027-01-01"]}))
    blank_rows = pd.DataFrame({"fund_name": ["A", None], "type": ["X", None], "closing_date": ["2027-01-01", None]})
    assert len(normalize_fund_specs(blank_rows)) == 1  # fully blank rows are ignored


# ------------------------------------------------------------ market data
def test_market_data_long_form_pivots_to_wide(gbp_portfolio, worked_funds):
    fund_spec, fund_market, wide, rates = tables()
    long = wide.melt(id_vars="date", var_name="series", value_name="value")
    result = Orchestrator(FrameRepository(fund_spec, fund_market, long, rates), SPEC).run()
    pd.testing.assert_frame_equal(result.periods, expected(gbp_portfolio, worked_funds).periods)
    with pytest.raises(ValueError, match="more than one value for 'liquid_gbp' on 2027-01-01"):
        normalize_market_data(pd.concat([long, long.iloc[[0]]]))


def test_market_data_sparse_fx_and_validation():
    wide = normalize_market_data(pd.DataFrame({"date": ["2027-01-01", "2027-02-01", "2027-03-01"],
                                               "liquid": [100, 101, 102], "fx": [0.8, None, 0.9]}))
    assert wide.index.name == "date" and list(wide.columns) == ["liquid", "fx"]
    assert wide["fx"].dropna().tolist() == [0.8, 0.9]
    with pytest.raises(ValueError, match="more than one row for 2027-01-01"):
        normalize_market_data(pd.DataFrame({"date": ["2027-01-01", "2027-01-01"], "liquid": [1, 2]}))
    with pytest.raises(ValueError, match="liquid in row 2 is 'n/a', not a number"):
        normalize_market_data(pd.DataFrame({"date": ["2027-01-01", "2027-02-01"], "liquid": [1, "n/a"]}))


def test_fx_quote_can_be_inverted():
    fund_spec, fund_market, market, rates = tables()
    market["usd_per_gbp"] = 1.0 / market["gbp_per_usd"]
    spec = SimulationSpec("GBP", liquid_series="liquid_gbp", fx_series="usd_per_gbp", fx_quote="usd_per_base",
                          weights={"A": 0.6, "B": 0.4})
    portfolio = Orchestrator(FrameRepository(fund_spec, fund_market, market, rates), spec).portfolio
    np.testing.assert_allclose(portfolio.usd_rate, [0.8, 0.8, 0.75])
    with pytest.raises(ValueError, match="fx_quote must be one of"):
        SimulationSpec("GBP", liquid_series="x", fx_series="y", fx_quote="sideways")


def test_series_and_currency_configuration_errors():
    fund_spec, fund_market, market, rates = tables()
    repository = FrameRepository(fund_spec, fund_market, market, rates)
    with pytest.raises(ValueError, match=r"no series 'nope' for liquid_series; series are \['liquid_gbp', 'gbp_per_usd'\]"):
        Orchestrator(repository, SimulationSpec("GBP", liquid_series="nope", fx_series="gbp_per_usd")).portfolio
    with pytest.raises(ValueError, match="fx_series is required: base_currency 'GBP' is not USD"):
        Orchestrator(repository, SimulationSpec("GBP", liquid_series="liquid_gbp")).portfolio
    with pytest.raises(ValueError, match="fx_series is set but base_currency is USD"):
        Orchestrator(repository, SimulationSpec("USD", liquid_series="liquid_gbp", fx_series="gbp_per_usd")).portfolio
    usd = Orchestrator(repository, SimulationSpec("USD", liquid_series="liquid_gbp", weights={"A": .6, "B": .4}))
    assert usd.portfolio.usd_rate is None and usd.run().periods["total_close"].iloc[-1] == pytest.approx(120_835_000)
    with pytest.raises(ValueError, match="liquid_series must name"):
        SimulationSpec("USD", liquid_series=" ")


# ------------------------------------------------------- commitment rates
def test_commitment_rates_long_wide_spec_and_missing():
    fund_spec, fund_market, market, wide = tables()
    long = pd.DataFrame({"year": [2027, 2027], "fund_type": ["BUYOUT", "VC"], "rate": [0.1, 0.05]})
    from_long = normalize_commitment_rates(long)
    assert from_long.loc[2027].to_dict() == {"BUYOUT": 0.1, "VC": 0.05}
    assert from_long.index.name == "year" and from_long.columns.name == "fund_type"
    pd.testing.assert_frame_equal(normalize_commitment_rates(pd.DataFrame({"Year": [2027.0], "BUYOUT": [0.1]})),
                                  normalize_commitment_rates(wide))
    with pytest.raises(ValueError, match=r"no rate for \[\(2028, 'VC'\)\]; use 0"):
        normalize_commitment_rates(pd.DataFrame({"year": [2027, 2027, 2028], "type": ["BUYOUT", "VC", "BUYOUT"], "rate": [.1, .05, .1]}))
    with pytest.raises(ValueError, match="more than one rate for 'BUYOUT' in 2027"):
        normalize_commitment_rates(pd.DataFrame({"year": [2027, 2027], "type": ["BUYOUT", "BUYOUT"], "rate": [.1, .2]}))
    with pytest.raises(ValueError, match="not a calendar year"):
        normalize_commitment_rates(pd.DataFrame({"year": ["FY27"], "BUYOUT": [0.1]}))
    # the spec overrides the sheet; with neither, a clear error
    override = Orchestrator(FrameRepository(fund_spec, fund_market, market, wide),
                            SimulationSpec("GBP", "liquid_gbp", "gbp_per_usd", commitment_rates={"BUYOUT": {2027: 0.2}},
                                           weights={"A": .6, "B": .4}))
    assert override.policy.rates.at[2027, "BUYOUT"] == 0.2  # the spec's table, not the repository's
    with pytest.raises(ValueError, match="commitment_rates are needed"):
        Orchestrator(FrameRepository(fund_spec, fund_market, market), SPEC).portfolio


# ------------------------------------------------------- expected returns
def test_expected_return_comes_from_the_spec_then_the_repository_then_nothing():
    fund_spec, fund_market, market, rates = tables()
    without_table = Orchestrator(FrameRepository(fund_spec, fund_market, market, rates), SPEC)
    assert without_table.expected_return is None and without_table.policy.expected_return is None  # rates are shares already

    returns = pd.DataFrame({"Liquid": ["Liquid GBP", "liquid_usd"], "ExRet": [0.05, 0.04]})
    with_table = Orchestrator(FrameRepository(fund_spec, fund_market, market, rates, returns), SPEC)
    assert with_table.expected_return == 0.05 and with_table.policy.expected_return == 0.05  # looked up by the liquid series' name

    override = SimulationSpec("GBP", "liquid_gbp", "gbp_per_usd", weights={"A": 0.6, "B": 0.4}, expected_return=0.08)
    assert Orchestrator(FrameRepository(fund_spec, fund_market, market, rates, returns), override).expected_return == 0.08

    other = pd.DataFrame({"Liquid": ["something else"], "ExRet": [0.05]})
    with pytest.raises(ValueError, match=r"expected_returns has no row for portfolio 'liquid_gbp'; portfolios are \['something else'\]"):
        Orchestrator(FrameRepository(fund_spec, fund_market, market, rates, other), SPEC).expected_return
    with pytest.raises(ValueError, match="write 5% as 0.05"):
        SimulationSpec("GBP", "liquid_gbp", "gbp_per_usd", expected_return=5)


def test_expected_return_changes_the_second_commitment_of_the_worked_example():
    fund_spec, fund_market, market, rates = tables()
    returns = pd.DataFrame({"Liquid": ["liquid_gbp"], "ExRet": [0.05]})
    c = Orchestrator(FrameRepository(fund_spec, fund_market, market, rates, returns), SPEC).run().commitments
    # A is the first commitment (31 Mar): expected value 1. B is 91 days into a year that runs to 31 Mar 2028 and holds a 29 Feb
    assert c["expected_value"].tolist() == pytest.approx([1.0, 1.05 ** (91 / 366)])
    assert c["commitment_usd"].tolist() == pytest.approx([8_250_000, 0.04 / 1.05 ** (91 / 366) * 121_000_000 / 0.75])


# -------------------------------------------------------------- repository
def test_frame_repository_normalizes_once_and_hands_out_copies():
    fund_spec, fund_market, market, rates = tables()
    repository = FrameRepository(fund_spec, fund_market, market, rates)
    specs = repository.fund_specs()
    assert list(specs.columns) == ["fund_name", "fund_type", "closing_date"]
    assert specs["closing_date"].tolist() == [date(2027, 2, 15), date(2027, 5, 10)]
    specs.loc[0, "fund_name"] = "changed"
    assert repository.fund_specs()["fund_name"].tolist() == ["A", "B"]
    assert list(repository.fund_market_data().columns) == ["fund_name", "kind", "date", "value", "scale", "unit"]
    assert repository.market_data().index.equals(pd.DatetimeIndex(["2027-01-01", "2027-03-31", "2027-06-30"], name="date"))
    assert FrameRepository(fund_spec, fund_market, market).commitment_rates() is None
    with pytest.raises(TypeError, match="expected a DataFrame"):
        FrameRepository("not a frame", fund_market, market)


def test_build_portfolio_directly():
    fund_spec, fund_market, market, rates = tables()
    portfolio = build_portfolio(normalize_market_data(market), SPEC, normalize_commitment_rates(rates))
    # the sheet quotes 1,000,000, 1,100,000, 1,210,000: read as an index, and started at 100,000,000
    assert portfolio.liquid_levels.tolist() == [1e8, 1.1e8, 1.21e8] and portfolio.usd_rate.tolist() == [0.8, 0.8, 0.75]


# ---------------------------------------------------------- draw plans
@pytest.mark.parametrize("cell, expected", [
    ("1-4", {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}),
    ("9-11", {9: 1.0, 10: 1.0, 11: 1.0}),
    ("12x3", {12: 3.0}),
    ("16X3", {16: 3.0}),
    ("16*3", {16: 3.0}),
    ("16×3", {16: 3.0}),
    ("1-4, 7", {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 7: 1.0}),
    ("12x3, 15", {12: 3.0, 15: 1.0}),
    (" 1 - 2 x 1.5 ", {1: 1.5, 2: 1.5}),
    ("7", {7: 1.0}),
    (7, {7: 1.0}),
    ("", None), ("-", None), ("none", None), ("N/A", None), (None, None), (np.nan, None),
])
def test_a_draw_plan_is_read_from_one_cell(cell, expected):
    assert parse_draw_plan(cell, fund_name="SEC_VI") == expected


@pytest.mark.parametrize("cell, message", [
    ("abc", "cannot read 'abc'"),
    ("1-4-7", "cannot read '1-4-7'"),
    ("12x", "cannot read '12x'"),
    ("4-1", "runs backwards"),
    ("1, 1", "year 1 appears twice"),
    ("1-4, 3", "year 3 appears twice"),
])
def test_an_unreadable_draw_plan_names_the_fund_and_the_term(cell, message):
    with pytest.raises(ValueError, match=message):
        parse_draw_plan(cell, fund_name="SEC_VI")


def test_relative_draw_years_are_mapped_onto_calendar_years():
    plans = relative_draw_plans_to_calendar({"SEC_VI": {1: 1.0, 4: 1.0}, "SEC_IX": {12: 3.0}}, inception_year=2009)
    assert plans == {"SEC_VI": {2010: 1.0, 2013: 1.0}, "SEC_IX": {2021: 3.0}}


def test_the_repository_and_the_spec_both_carry_draw_plans():
    fund_spec, fund_market, market, rates = tables()
    repository = FrameRepository(fund_spec, fund_market, market, rates, draw_plans={"A": {2027: 2.0}})
    assert repository.draw_plans() == {"A": {2027: 2.0}}
    assert Orchestrator(repository, SPEC).draw_plans == {"A": {2027: 2.0}}
    # the spec wins when both say something, and an empty mapping in the spec means "no plans at all"
    assert Orchestrator(repository, SimulationSpec(**{**vars(SPEC), "draws": {"B": {2027: 3.0}}})).draw_plans \
        == {"B": {2027: 3.0}}
    assert Orchestrator(repository, SimulationSpec(**{**vars(SPEC), "draws": {}})).draw_plans == {}
    assert Orchestrator(FrameRepository(fund_spec, fund_market, market, rates), SPEC).draw_plans == {}


def test_a_drawn_plan_reaches_the_policy_and_doubles_the_commitment():
    fund_spec, fund_market, market, rates = tables()
    plain = Orchestrator(FrameRepository(fund_spec, fund_market, market, rates), SPEC).run()
    doubled = Orchestrator(FrameRepository(fund_spec, fund_market, market, rates),
                           SimulationSpec(**{**vars(SPEC), "draws": {"A": {2027: 2.0}, "B": {2027: 1.0}}})).run()
    a = (pd.Timestamp("2027-03-31"), "A")
    assert doubled.commitments.loc[a, "commitment_usd"] == pytest.approx(2 * plain.commitments.loc[a, "commitment_usd"])
    assert doubled.commitments.loc[a, "drawn_years"] == "2027x2"
    b = (pd.Timestamp("2027-06-30"), "B")  # B's plan is its own year at 1: unchanged
    assert doubled.commitments.loc[b, "commitment_usd"] == pytest.approx(plain.commitments.loc[b, "commitment_usd"])


# ---------------------------------------------------------------- rounding
def test_the_spec_carries_the_rounding_unit_to_the_policy():
    fund_spec, fund_market, market, rates = tables()
    plain = Orchestrator(FrameRepository(fund_spec, fund_market, market, rates), SPEC)
    assert plain.spec.commitment_rounding_unit_usd is None and plain.policy.rounding_unit_usd is None  # off unless asked for

    rounded_spec = SimulationSpec(**{**vars(SPEC), "commitment_rounding_unit_usd": 1_000_000})
    rounded = Orchestrator(FrameRepository(fund_spec, fund_market, market, rates), rounded_spec)
    assert rounded.policy.rounding_unit_usd == 1_000_000.0
    commitments = rounded.run().commitments["commitment_usd"]
    # the two own-year budgets — 10% of $137,500,000 and of $161,333,333 — go to the nearest million, then split 60/40
    assert commitments.tolist() == pytest.approx([0.6 * 14_000_000, 0.4 * 16_000_000])
    assert commitments.tolist() != pytest.approx(plain.run().commitments["commitment_usd"].tolist())

    with pytest.raises(ValueError, match="commitment_rounding_unit_usd must be a positive number of dollars"):
        SimulationSpec(**{**vars(SPEC), "commitment_rounding_unit_usd": -1})


# ------------------------------------------------------- the starting value
@pytest.mark.parametrize("quoted_from", [1.0, 100.0, 1_000_000.0, 100_000_000.0, 2_500_000_000.0])
def test_a_level_series_is_an_index_and_the_run_always_starts_at_a_hundred_million(quoted_from):
    """Only the changes in a level series matter. Whatever scale it is quoted on, the run starts at 100,000,000."""
    fund_spec, fund_market, market, rates = tables()
    market["liquid_gbp"] = [quoted_from, quoted_from * 1.1, quoted_from * 1.21]
    result = Orchestrator(FrameRepository(fund_spec, fund_market, market, rates), SPEC).run()

    assert result.periods["liquid_open"].iloc[0] == pytest.approx(100_000_000.0, rel=1e-12)
    np.testing.assert_allclose(result.periods["liquid_only"], [1e8, 1.1e8, 1.21e8])
    np.testing.assert_allclose(result.periods["period_return"], [0.0, 0.1, 0.1])  # the returns are what the series said
    assert result.periods["total_close"].iloc[-1] == pytest.approx(120_731_875.0)  # and so the same run every time


def test_the_spec_has_no_starting_value_and_a_level_series_must_be_positive():
    assert "initial_value" not in SimulationSpec.__dataclass_fields__
    with pytest.raises(TypeError, match="initial_value"):
        SimulationSpec("GBP", liquid_series="liquid_gbp", fx_series="gbp_per_usd", initial_value=1_000_000)

    fund_spec, fund_market, market, rates = tables()
    market["liquid_gbp"] = [0.0, 1.1e6, 1.21e6]
    with pytest.raises(ValueError, match="liquid levels must be strictly positive; the first is 0.0"):
        Orchestrator(FrameRepository(fund_spec, fund_market, market, rates), SPEC).portfolio
