"""The liquid-only counterfactual and the public-market-equivalent measures read off a finished run."""
import math

import numpy as np
import pandas as pd
import pytest

from pmsim import AnnualRatePolicy, Fund, Portfolio, Simulator, annualised_irr
from pmsim.benchmark import COMPARISON_COLUMNS, PME_COLUMNS, growth_to_horizon
from tests.conftest import levels

D = pd.Timestamp
QUARTER = 91  # days from 31 Mar 2027 to 30 Jun 2027


def worked(portfolio, funds):
    policy = AnnualRatePolicy(portfolio.commitment_rates, funds, weights={"A": 0.6, "B": 0.4})
    return Simulator(portfolio, funds, policy).run()


# ------------------------------------------------------- the worked example
def test_liquid_only_comparison_on_the_worked_example(usd_portfolio, worked_funds):
    comparison = worked(usd_portfolio, worked_funds).compare_with_liquid_only()
    assert list(comparison.columns) == COMPARISON_COLUMNS and comparison.index.name == "date"
    np.testing.assert_allclose(comparison["liquid_only"], [1_000_000, 1_100_000, 1_210_000])
    np.testing.assert_allclose(comparison["with_programme"], [1_000_000, 1_100_000, 1_208_350])
    # 16,500 was called on 31 Mar into a fund carried at cost while the index rose 10%: 1,650 forgone
    np.testing.assert_allclose(comparison["value_added"], [0, 0, -1_650], atol=1e-6)
    assert comparison["value_added_share"].iloc[-1] == pytest.approx(-1_650 / 1_210_000)


def test_pme_on_the_worked_example(usd_portfolio, worked_funds):
    pme = worked(usd_portfolio, worked_funds).public_market_equivalent()
    assert list(pme.columns) == PME_COLUMNS and list(pme.index.names) == ["level", "name"]
    assert list(pme.index) == [("programme", "all"), ("fund_type", "BUYOUT"), ("fund", "A"), ("fund", "B")]

    programme = pme.loc[("programme", "all")]
    assert programme["calls"] == pytest.approx(16_500 + 12_100) and programme["distributions"] == pytest.approx(3_300)
    assert programme["nav"] == pytest.approx(25_300)
    assert programme["fv_calls"] == pytest.approx(16_500 * 1.1 + 12_100) and programme["fv_distributions"] == pytest.approx(3_300)
    assert programme["value_added"] == pytest.approx(-1_650)
    assert programme["ks_pme"] == pytest.approx(28_600 / 30_250)
    pd.testing.assert_series_equal(pme.loc[("fund_type", "BUYOUT")], programme, check_names=False)  # one type: same figures

    a = pme.loc[("fund", "A")]  # 18,150 of index-compounded calls against 16,500 back, a quarter later
    assert a["value_added"] == pytest.approx(-1_650) and a["ks_pme"] == pytest.approx(16_500 / 18_150)
    assert a["irr"] == pytest.approx(0.0, abs=1e-9)  # carried at cost: the money came back unchanged
    assert a["direct_alpha"] == pytest.approx((16_500 / 18_150) ** (365 / QUARTER) - 1)
    index_return = 1.1 ** (365 / QUARTER) - 1
    assert (1 + a["irr"]) == pytest.approx((1 + index_return) * (1 + a["direct_alpha"]))  # what direct alpha means

    b = pme.loc[("fund", "B")]  # called and valued on the same day: nothing to annualise
    assert b["value_added"] == pytest.approx(0.0, abs=1e-6) and b["ks_pme"] == pytest.approx(1.0)
    assert math.isnan(b["irr"]) and math.isnan(b["direct_alpha"])


def test_gbp_example_is_benchmarked_in_base_currency(gbp_portfolio, worked_funds):
    result = worked(gbp_portfolio, worked_funds)
    comparison, pme = result.compare_with_liquid_only(), result.public_market_equivalent()
    assert comparison["value_added"].iloc[-1] == pytest.approx(1_207_318.75 - 1_210_000)
    programme = pme.loc[("programme", "all")]
    assert programme["calls"] == pytest.approx(16_500 + 12_100) and programme["nav"] == pytest.approx(24_475)
    assert programme["value_added"] == pytest.approx(-2_681.25)  # 1,650 forgone return + 1,031.25 lost to the weaker dollar


# ------------------------------------------------------------- the identity
def shortfall_run(stop):
    portfolio = Portfolio("USD", levels(("2027-01-01", 100), ("2027-02-01", 110), ("2027-03-01", 99)), {"VC": {2027: 1.2}})
    fund = Fund("S", "VC", "2027-02-01", unit_calls=[("2027-02-01", 1.0)], unit_distributions=[("2027-02-15", 0.5)])
    return Simulator(portfolio, [fund], stop_on_shortfall=stop).run()


def two_types_with_marks():
    portfolio = Portfolio("EUR", levels(("2027-01-31", 1000), ("2027-02-28", 1030), ("2027-03-31", 990), ("2027-04-30", 1060)),
                          {"BUYOUT": {2027: 0.10}, "VC": {2027: 0.05}},
                          usd_rate=[("2027-01-31", 0.90), ("2027-03-31", 0.95), ("2027-04-30", 0.92)])
    funds = [
        Fund("B1", "BUYOUT", "2027-01-31", unit_calls=[("2027-02-10", 0.4), ("2027-04-02", 0.2)],
             unit_nav=[("2027-03-31", 0.5)], unit_distributions=[("2027-04-20", 0.1)]),
        Fund("V1", "VC", "2027-02-15", unit_calls=[("2027-03-05", 0.3)], unit_nav=[("2027-04-30", 0.45)]),
    ]
    return Simulator(portfolio, funds).run()


@pytest.mark.parametrize("make_result", [
    lambda: shortfall_run(stop=True), lambda: shortfall_run(stop=False), two_types_with_marks,
], ids=["stopped at a shortfall", "continued through a shortfall", "two types, marks and FX"])
def test_value_added_is_the_same_number_three_ways(make_result):
    result = make_result()
    comparison, pme = result.compare_with_liquid_only(), result.public_market_equivalent()
    programme = pme.loc[("programme", "all")]
    # total wealth less the liquid-only counterfactual == index-compounded net flows + closing NAV
    assert programme["value_added"] == pytest.approx(comparison["value_added"].iloc[-1], abs=1e-9)
    assert programme["value_added"] == pytest.approx(programme["fv_calls"] * (programme["ks_pme"] - 1), abs=1e-9)
    for level in ("fund_type", "fund"):  # and it adds up across fund types and across funds
        for column in ("calls", "distributions", "nav", "fv_calls", "fv_distributions", "value_added"):
            assert pme.loc[level, column].sum() == pytest.approx(programme[column], abs=1e-9)


def test_growth_to_horizon_compounds_the_return_factors():
    growth = growth_to_horizon(two_types_with_marks().periods)
    np.testing.assert_allclose(growth, [1060 / 1000, 1060 / 1030, 1060 / 990, 1.0])


def test_a_run_without_funds_adds_nothing():
    result = Simulator(Portfolio("USD", levels(("2027-01-01", 100), ("2027-02-01", 90)), {})).run()
    comparison, pme = result.compare_with_liquid_only(), result.public_market_equivalent()
    np.testing.assert_allclose(comparison["liquid_only"], [100, 90])
    assert (comparison["value_added"] == 0).all() and (comparison["value_added_share"] == 0).all()
    assert list(pme.index) == [("programme", "all")]
    assert pme.loc[("programme", "all"), "value_added"] == 0 and pme[["ks_pme", "irr", "direct_alpha"]].isna().all().all()


def test_empty_tables_are_rejected():
    empty = Simulator(Portfolio("USD", levels(("2027-01-01", 100)), {})).run().periods.iloc[0:0]
    with pytest.raises(ValueError, match="no run to benchmark"):
        growth_to_horizon(empty)


# -------------------------------------------------------------- annualised IRR
def test_annualised_irr_on_known_cases():
    assert annualised_irr(["2027-01-01", "2028-01-01"], [-100, 110]) == pytest.approx(0.10)  # 365 days
    assert annualised_irr(["2027-01-01", "2028-12-31"], [-100, 121]) == pytest.approx(0.10)  # 730 days
    assert annualised_irr(["2027-01-01", "2028-01-01"], [-100, 90]) == pytest.approx(-0.10)
    assert annualised_irr(["2027-01-01", "2027-04-02"], [-100, 110]) == pytest.approx(1.1 ** (365 / 91) - 1)
    assert annualised_irr(["2028-01-01", "2027-01-01"], [110, -100]) == pytest.approx(0.10)  # order does not matter
    dates = pd.to_datetime(["2027-01-01", "2027-07-01", "2028-03-01", "2029-01-01", "2030-06-30"])
    amounts = np.array([-100.0, -50.0, 30.0, 60.0, 120.0])
    rate = annualised_irr(dates, amounts)
    years = (dates - dates[0]).days.to_numpy() / 365
    assert float(np.sum(amounts / (1 + rate) ** years)) == pytest.approx(0.0, abs=1e-7)  # it is a root


@pytest.mark.parametrize("dates, amounts", [
    ([], []),
    (["2027-01-01", "2028-01-01"], [100, 110]),      # nothing paid out
    (["2027-01-01", "2028-01-01"], [-100, -10]),     # nothing received
    (["2027-01-01", "2027-01-01"], [-100, 110]),     # no time passes
    (["2027-01-01", "2028-01-01"], [-100, 0]),       # a zero is not a receipt
    (["2027-01-01", "2028-01-01"], [-100, 1e-9]),    # a loss beyond −99.99% a year
])
def test_annualised_irr_is_nan_when_undefined(dates, amounts):
    assert math.isnan(annualised_irr(dates, amounts))


def test_annualised_irr_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="same length"):
        annualised_irr(["2027-01-01"], [-100, 110])
