"""Simulator: accounting, timing, currency, failure and repeatability."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from pmsim import AnnualRatePolicy, Fund, Portfolio, Simulator
from tests.conftest import levels

D = pd.Timestamp


def usd(levels_, rates, **kwargs):
    return Portfolio("USD", levels_, rates, **kwargs)


# ------------------------------------------------------------ no funds
@pytest.mark.parametrize("currency, rate", [("USD", None), ("GBP", [("2027-01-01", 0.8)])])
def test_without_funds_the_liquid_balance_is_the_index(currency, rate, identities):
    portfolio = Portfolio(currency, levels(("2027-01-01", 100), ("2027-02-01", 90), ("2027-03-01", 99)), {}, usd_rate=rate)
    result = Simulator(portfolio).run()
    assert result.status == "completed" and result.shortfall is None and result.funds_beyond_horizon == ()
    np.testing.assert_allclose(result.periods["liquid_close"], [100, 90, 99])
    np.testing.assert_allclose(result.periods["return_factor"], [1.0, 0.9, 1.1])
    assert (result.periods[["private_close", "calls", "distributions", "commitments"]] == 0).all().all()
    assert result.funds.empty and result.commitments.empty and result.totals_by_fund_type().empty
    assert list(result.funds.index.names) == ["date", "fund"] and "nav_base" in result.funds.columns
    assert result.nav_by_fund().shape == (3, 0)
    identities(result)


# ------------------------------------------------------- worked examples
def test_worked_example_in_usd(usd_portfolio, worked_funds, identities):
    policy = AnnualRatePolicy(usd_portfolio.commitment_rates, worked_funds, weights={"A": 0.6, "B": 0.4})
    result = Simulator(usd_portfolio, worked_funds, policy).run()
    p = result.periods
    assert result.status == "completed"
    np.testing.assert_allclose(p["sizing_base"], [1_000_000, 1_100_000, 1_195_150])
    np.testing.assert_allclose(p["commitments"], [0, 66_000, 47_806])
    np.testing.assert_allclose(p["calls"], [0, 16_500, 11_951.5])
    np.testing.assert_allclose(p["distributions"], [0, 0, 3_300])
    np.testing.assert_allclose(p["liquid_close"], [1_000_000, 1_083_500, 1_183_198.5])
    np.testing.assert_allclose(p["private_close"], [0, 16_500, 25_151.5])
    np.testing.assert_allclose(p["total_close"], [1_000_000, 1_100_000, 1_208_350])
    assert (p["usd_rate"] == 1).all() and (p["fx_translation"] == 0).all()
    c = result.commitments
    assert list(c.index) == [(D("2027-03-31"), "A"), (D("2027-06-30"), "B")]
    assert c["commitment_usd"].tolist() == pytest.approx([66_000, 47_806])
    assert c["closing_date"].tolist() == [D("2027-02-15"), D("2027-05-10")]
    assert c["policy_year"].tolist() == [2027, 2027] and c["rate"].tolist() == pytest.approx([0.06, 0.04])
    assert c["weight"].tolist() == [0.6, 0.4] and c["pooled_rate"].tolist() == [0.1, 0.1]
    f = result.funds
    assert f.loc[(D("2027-06-30"), "A"), "nav_usd"] == pytest.approx(13_200)
    assert f.loc[(D("2027-06-30"), "B"), "calls_usd"] == pytest.approx(11_951.5)
    assert (D("2027-03-31"), "B") not in f.index  # B does not exist before its closing
    identities(result)


def test_worked_example_in_gbp_translates_at_the_observation_rate(gbp_portfolio, worked_funds, identities):
    policy = AnnualRatePolicy(gbp_portfolio.commitment_rates, worked_funds, weights={"A": 0.6, "B": 0.4})
    result = Simulator(gbp_portfolio, worked_funds, policy).run()
    p = result.periods
    assert result.base_currency == "GBP"
    np.testing.assert_allclose(p["usd_rate"], [0.80, 0.80, 0.75])
    np.testing.assert_allclose(p["sizing_base"], [1_000_000, 1_100_000, 1_194_943.75])
    np.testing.assert_allclose(p["commitments"], [0, 66_000, 47_797.75])
    np.testing.assert_allclose(p["commitments_usd"], [0, 82_500, 47_797.75 / 0.75])
    np.testing.assert_allclose(p["distributions"], [0, 0, 3_093.75])
    np.testing.assert_allclose(p["calls"], [0, 16_500, 11_949.4375])
    np.testing.assert_allclose(p["liquid_close"], [1_000_000, 1_083_500, 1_182_994.3125])
    np.testing.assert_allclose(p["private_close"], [0, 16_500, 24_324.4375])
    np.testing.assert_allclose(p["total_close"], [1_000_000, 1_100_000, 1_207_318.75])
    np.testing.assert_allclose(p["fx_translation"], [0, 0, -1_031.25])
    np.testing.assert_allclose(p["private_valuation_pnl"], [0, 0, -1_031.25])  # no marks: all of it is FX
    c = result.commitments
    assert c.loc[(D("2027-03-31"), "A"), "commitment_usd"] == pytest.approx(82_500)
    assert c.loc[(D("2027-06-30"), "B"), "commitment_usd"] == pytest.approx(63_730.3333333)
    assert c["usd_rate"].tolist() == [0.80, 0.75]
    a = result.funds.xs("A", level="fund")
    np.testing.assert_allclose(a["nav_usd"], [20_625, 16_500])
    np.testing.assert_allclose(a["nav_base"], [16_500, 12_375])
    identities(result)


def test_carry_forward_example(identities):
    portfolio = usd(levels(("2027-01-01", 1e6), ("2028-12-31", 1e6), ("2029-03-31", 1e6), ("2029-06-30", 1.2e6)),
                    {"BUYOUT": {2027: 0.10, 2028: 0.08, 2029: 0.12}})
    funds = [Fund("C", "BUYOUT", "2029-03-01"), Fund("D", "BUYOUT", "2029-06-01")]
    policy = AnnualRatePolicy(portfolio.commitment_rates, funds, {"C": 0.6, "D": 0.4}, carry_forward=True, years=portfolio.calendar_years)
    result = Simulator(portfolio, funds, policy).run()
    c = result.commitments
    assert c["commitment_usd"].tolist() == pytest.approx([180_000, 144_000])
    assert c["carried_rate"].tolist() == pytest.approx([0.18, 0.18])
    assert c["pooled_rate"].tolist() == pytest.approx([0.30, 0.30])
    assert c["current_year_rate"].tolist() == [0.12, 0.12]
    identities(result)


# ----------------------------------------------------------- timing rules
def test_closing_maps_to_next_observation_but_keeps_its_policy_year(identities):
    portfolio = usd(levels(("2027-01-01", 100), ("2027-12-01", 100), ("2028-01-31", 100)),
                    {"BUYOUT": {2027: 0.1, 2028: 0.2}})
    funds = [Fund("Dec", "BUYOUT", "2027-12-15"), Fund("Jan", "BUYOUT", "2028-01-10")]
    result = Simulator(portfolio, funds).run()
    c = result.commitments
    assert list(c.index) == [(D("2028-01-31"), "Dec"), (D("2028-01-31"), "Jan")]
    assert c["policy_year"].tolist() == [2027, 2028]
    assert c["commitment_usd"].tolist() == pytest.approx([10, 20])  # both sized from the same 100
    assert c["sizing_base"].tolist() == [100, 100]
    identities(result)


def test_first_date_closing_and_same_day_call_are_processed_once(identities):
    portfolio = usd(levels(("2027-01-01", 1000), ("2027-04-01", 1100)), {"BUYOUT": {2027: 0.2}})
    result = Simulator(portfolio, [Fund("A", "BUYOUT", "2027-01-01", unit_calls=[("2027-01-01", 0.5)])]).run()
    p = result.periods
    assert p["return_factor"].tolist() == [1.0, 1.1]
    np.testing.assert_allclose(p["commitments"], [200, 0])
    np.testing.assert_allclose(p["calls"], [100, 0])
    np.testing.assert_allclose(p["liquid_close"], [900, 990])
    np.testing.assert_allclose(p["private_close"], [100, 100])
    identities(result)


def test_new_cohort_distributions_are_banked_after_sizing(identities):
    # E closes on 10 Jan (observed 1 Mar), calls 0.30 on 20 Jan and distributes 0.10 on 10 Feb
    portfolio = usd(levels(("2027-01-01", 100), ("2027-03-01", 100)), {"VC": {2027: 0.5}})
    fund = Fund("E", "VC", "2027-01-10", unit_calls=[("2027-01-20", 0.3)], unit_distributions=[("2027-02-10", 0.1)])
    result = Simulator(portfolio, [fund]).run()
    row = result.periods.iloc[1]
    assert row["sizing_base"] == 100 and row["commitments"] == 50  # its own distribution did not inflate the base
    assert row["distributions"] == pytest.approx(5) and row["calls"] == pytest.approx(15)
    assert row["liquid_close"] == pytest.approx(90) and row["private_close"] == pytest.approx(10)
    identities(result)


def test_same_day_gross_flows_stay_visible(identities):
    portfolio = usd(levels(("2027-01-01", 100), ("2027-02-01", 100)), {"VC": {2027: 0.5}})
    fund = Fund("G", "VC", "2027-01-01", unit_calls=[("2027-01-15", 0.2)], unit_distributions=[("2027-01-15", 0.2)])
    result = Simulator(portfolio, [fund]).run()
    row = result.periods.iloc[1]
    assert row["calls"] == pytest.approx(10) and row["distributions"] == pytest.approx(10)
    assert row["liquid_close"] == pytest.approx(100) and row["private_close"] == 0
    identities(result)


def test_off_grid_nav_mark_drives_valuation_pnl(identities):
    portfolio = usd(levels(("2027-01-31", 100), ("2027-02-28", 100), ("2027-03-31", 100)), {"VC": {2027: 0.5}})
    fund = Fund("M", "VC", "2027-01-31", unit_calls=[("2027-01-31", 0.4)], unit_nav=[("2027-02-10", 0.6)],
                unit_distributions=[("2027-03-05", 0.1)])
    result = Simulator(portfolio, [fund]).run()
    p = result.periods
    np.testing.assert_allclose(p["private_close"], [20, 30, 25])
    np.testing.assert_allclose(p["private_valuation_pnl"], [0, 10, 0])
    identities(result)


def test_fund_order_does_not_change_any_result(gbp_portfolio, worked_funds):
    a, b = worked_funds
    forward = Simulator(gbp_portfolio, [a, b], AnnualRatePolicy(gbp_portfolio.commitment_rates, [a, b], {"A": .6, "B": .4})).run()
    reverse = Simulator(gbp_portfolio, [b, a], AnnualRatePolicy(gbp_portfolio.commitment_rates, [b, a], {"A": .6, "B": .4})).run()
    pd.testing.assert_frame_equal(forward.periods, reverse.periods, check_exact=True)
    pd.testing.assert_frame_equal(forward.funds.sort_index(), reverse.funds.sort_index(), check_exact=True)
    pd.testing.assert_frame_equal(forward.commitments.sort_index(), reverse.commitments.sort_index(), check_exact=True)


# --------------------------------------------------------------- failure
def shortfall_setup(*extra_dates):
    dates = [("2027-01-01", 100.0), ("2027-02-01", 100.0), *[(d, 100.0) for d in extra_dates]]
    portfolio = usd(dates, {"VC": {2027: 1.2}})
    fund = Fund("S", "VC", "2027-02-01", unit_calls=[("2027-02-01", 1.0)], unit_distributions=[("2027-02-15", 0.5)])
    return portfolio, fund


def test_shortfall_stops_at_the_failed_observation_and_reports_the_gap():
    portfolio, fund = shortfall_setup("2027-03-01")
    result = Simulator(portfolio, [fund]).run()
    assert result.status == "shortfall"
    s = result.shortfall
    assert (s.t, s.date, s.calls_due, s.cash_available, s.deficit) == (1, date(2027, 2, 1), 120, 100, pytest.approx(20))
    assert s.calls_by_fund.to_dict() == {"S": 120.0} and s.calls_by_fund.index.name == "fund"
    assert "shortfall of 20.00 on 2027-02-01" in str(s)
    assert len(result.periods) == 2  # the failed period is recorded, nothing after it
    assert result.periods["liquid_close"].iloc[-1] == pytest.approx(-20)  # the negative balance is visible
    assert result.commitments["commitment_usd"].tolist() == [120]


def test_shortfall_can_be_allowed_to_continue(identities):
    portfolio, fund = shortfall_setup("2027-03-01")
    result = Simulator(portfolio, [fund], stop_on_shortfall=False).run()
    assert result.status == "shortfall" and result.shortfall.t == 1
    assert len(result.periods) == 3
    np.testing.assert_allclose(result.periods["liquid_close"], [100, -20, 40])  # 60 distributed in March
    np.testing.assert_allclose(result.periods["private_close"], [0, 120, 60])
    identities(result)


def test_first_date_shortfall_leaves_one_recorded_period():
    portfolio = usd(levels(("2027-01-01", 100)), {"VC": {2027: 2.0}})
    result = Simulator(portfolio, [Fund("S", "VC", "2027-01-01", unit_calls=[("2027-01-01", 1.0)])]).run()
    assert result.status == "shortfall" and result.shortfall.deficit == pytest.approx(100)
    assert len(result.periods) == 1 and len(result.commitments) == 1


def test_full_cash_use_is_not_a_shortfall_and_zero_base_gives_zero_commitment(identities):
    portfolio = usd(levels(("2027-01-01", 100), ("2027-02-01", 100), ("2027-03-01", 100)), {"A": {2027: 1.0}, "B": {2027: 1.0}})
    funds = [Fund("F", "A", "2027-01-01", unit_calls=[("2027-01-15", 1.0)]), Fund("G", "B", "2027-02-15")]
    result = Simulator(portfolio, funds).run()
    assert result.status == "completed"
    np.testing.assert_array_equal(result.periods["liquid_close"], [100, 0, 0])
    assert result.commitments.loc[(D("2027-03-01"), "G"), "commitment_usd"] == 0
    assert result.commitments.loc[(D("2027-03-01"), "G"), "rate"] == 1.0  # the rate was used even though it bought nothing
    identities(result)


def test_cash_tolerance_absorbs_rounding_only():
    portfolio = usd(levels(("2027-01-01", 100), ("2027-02-01", 100)), {"VC": {2027: 1.0}})
    fund = Fund("S", "VC", "2027-01-01", unit_calls=[("2027-02-01", 1.0 + 1e-13)])
    assert Simulator(portfolio, [fund]).run().status == "completed"
    assert Simulator(portfolio, [fund], cash_tolerance=0.0).run().status == "shortfall"
    with pytest.raises(ValueError, match="cash_tolerance"):
        Simulator(portfolio, [fund], cash_tolerance=-1)


# ------------------------------------------------------------- validation
def test_fund_closing_before_inception_is_rejected():
    portfolio = usd(levels(("2027-01-01", 100)), {"VC": {2026: 0.1, 2027: 0.1}})
    with pytest.raises(ValueError, match="closes on 2026-12-31, before the first observation 2027-01-01"):
        Simulator(portfolio, [Fund("Old", "VC", "2026-12-31")])


def test_duplicate_names_and_wrong_types_are_rejected():
    portfolio = usd(levels(("2027-01-01", 100)), {"VC": {2027: 0.1}})
    with pytest.raises(ValueError, match=r"duplicated: \['A'\]"):
        Simulator(portfolio, [Fund("A", "VC", "2027-01-01"), Fund("A", "VC", "2027-02-01")])
    with pytest.raises(TypeError):
        Simulator(portfolio, ["not a fund"])
    with pytest.raises(TypeError):
        Simulator("not a portfolio")


def test_default_policy_requires_rates_for_every_horizon_year():
    portfolio = usd(levels(("2027-01-01", 100), ("2029-01-01", 100)), {"VC": {2028: 0.1, 2029: 0.1}})
    with pytest.raises(ValueError, match=r"no row for year\(s\) \[2027\]"):
        Simulator(portfolio, [Fund("A", "VC", "2028-06-01")])


def test_bad_policy_output_is_rejected():
    portfolio = usd(levels(("2027-01-01", 100)), {"VC": {2027: 0.1}})
    fund = Fund("A", "VC", "2027-01-01")

    class Negative:
        def size_commitments(self, cohort, base):
            return {"A": -1.0}

    class Stranger:
        def size_commitments(self, cohort, base):
            return {"A": 1.0, "Z": 1.0}

    with pytest.raises(ValueError, match="invalid commitment"):
        Simulator(portfolio, [fund], Negative()).run()
    with pytest.raises(ValueError, match=r"not closing now: \['Z'\]"):
        Simulator(portfolio, [fund], Stranger()).run()


def test_custom_policy_without_explain_still_reports_rate(identities):
    portfolio = usd(levels(("2027-01-01", 200), ("2027-02-01", 200)), {"VC": {2027: 0.0}})

    class FlatDollars:
        def size_commitments(self, cohort, base):
            return {f.name: 50.0 for f in cohort}

    result = Simulator(portfolio, [Fund("A", "VC", "2027-01-15")], FlatDollars()).run()
    row = result.commitments.iloc[0]
    assert row["commitment_usd"] == 50 and row["rate"] == pytest.approx(0.25) and np.isnan(row["weight"])
    identities(result)


# ---------------------------------------------------------- horizon edges
def test_funds_beyond_the_horizon_keep_their_weight_but_are_never_committed(identities):
    portfolio = usd(levels(("2027-01-01", 100), ("2027-06-30", 100)), {"BUYOUT": {2027: 0.1}})
    funds = [Fund("A", "BUYOUT", "2027-03-01"), Fund("B", "BUYOUT", "2027-09-01")]
    policy = AnnualRatePolicy(portfolio.commitment_rates, funds, {"A": 0.6, "B": 0.4})
    result = Simulator(portfolio, funds, policy).run()
    assert result.funds_beyond_horizon == ("B",)
    assert result.commitments["commitment_usd"].tolist() == pytest.approx([6.0])  # A keeps 60% of 10%, not 100%
    assert "B" not in result.funds.index.get_level_values("fund")
    identities(result)


def test_exposures_and_by_type(gbp_portfolio, worked_funds):
    policy = AnnualRatePolicy(gbp_portfolio.commitment_rates, worked_funds, {"A": 0.6, "B": 0.4})
    result = Simulator(gbp_portfolio, worked_funds, policy).run()
    exposures = result.nav_by_fund()
    assert list(exposures.columns) == ["A", "B"] and exposures.index.equals(result.periods.index)
    np.testing.assert_allclose(exposures["B"], [0, 0, 11_949.4375])
    by_type = result.totals_by_fund_type()
    assert list(by_type.index.names) == ["date", "fund_type"]
    assert by_type.loc[(D("2027-06-30"), "BUYOUT"), "nav_base"] == pytest.approx(24_324.4375)
    assert by_type.loc[(D("2027-06-30"), "BUYOUT"), "commitment_usd"] == pytest.approx(82_500 + 47_797.75 / 0.75)


# ----------------------------------------------------------- repeatability
def test_runs_repeat_exactly_and_leave_inputs_untouched(gbp_portfolio, worked_funds):
    before = [(f.unit_calls.copy(), f.unit_distributions.copy(), f.unit_nav.copy()) for f in worked_funds]
    levels_before = gbp_portfolio.liquid_levels.copy()
    simulator = Simulator(gbp_portfolio, worked_funds)
    first, second = simulator.run(), simulator.run()
    pd.testing.assert_frame_equal(first.periods, second.periods, check_exact=True)
    pd.testing.assert_frame_equal(first.funds, second.funds, check_exact=True)
    pd.testing.assert_frame_equal(first.commitments, second.commitments, check_exact=True)
    for fund, (calls, distributions, nav) in zip(worked_funds, before):
        pd.testing.assert_series_equal(fund.unit_calls, calls)
        pd.testing.assert_series_equal(fund.unit_distributions, distributions)
        pd.testing.assert_series_equal(fund.unit_nav, nav)
    pd.testing.assert_series_equal(gbp_portfolio.liquid_levels, levels_before)
    # a Series the caller mutates after construction does not reach the simulation
    calls = pd.Series([0.25], index=pd.to_datetime(["2027-03-01"]))
    fund = Fund("A", "BUYOUT", "2027-02-15", unit_calls=calls)
    calls.iloc[0] = 1.0
    result = Simulator(Portfolio("USD", levels(("2027-01-01", 100), ("2027-03-31", 100)), {"BUYOUT": {2027: 0.1}}), [fund]).run()
    assert result.periods["calls"].iloc[1] == pytest.approx(2.5)


def test_result_tables_have_stable_columns_and_dtypes(usd_portfolio, worked_funds):
    result = Simulator(usd_portfolio, worked_funds).run()
    from pmsim.simulator import COMMITMENT_COLUMNS, FUND_COLUMNS, PERIOD_COLUMNS
    assert list(result.periods.columns) == PERIOD_COLUMNS
    assert list(result.funds.columns) == FUND_COLUMNS
    assert list(result.commitments.columns) == COMMITMENT_COLUMNS
    assert result.periods.index.name == "date" and isinstance(result.periods.index, pd.DatetimeIndex)
    assert result.commitments["policy_year"].dtype == "int64"
    assert str(result.commitments["closing_date"].dtype).startswith("datetime64")
    assert result.periods.dtypes.eq(float).all()
