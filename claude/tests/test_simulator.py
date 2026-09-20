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
    np.testing.assert_allclose(result.periods["period_return"], [0.0, -0.1, 0.1])  # percent changes, 0 in the first period
    assert (result.periods[["private_close", "calls", "distributions", "commitments"]] == 0).all().all()
    assert result.funds.empty and result.commitments.empty and result.totals_by_fund_type().empty
    assert list(result.funds.index.names) == ["date", "fund"] and "nav_base" in result.funds.columns
    assert result.nav_by_fund().shape == (3, 0)
    identities(result)


def test_the_five_running_values(identities):
    """1 liquid alone · 2 liquid at the expected return · 3 liquid with private flows · 4 the private book · 5 the total."""
    portfolio = usd(levels(("2027-12-31", 1_000_000), ("2028-12-31", 1_100_000), ("2029-12-31", 1_100_000)),
                    {"BUYOUT": {2027: 0.0, 2028: 0.02, 2029: 0.0}})
    fund = Fund("P", "BUYOUT", "2028-12-31", unit_calls=[("2028-12-31", 1.0)], unit_nav=[("2029-12-31", 1.15)])
    policy = AnnualRatePolicy(portfolio.commitment_rates, [fund], expected_return=0.05, years=portfolio.calendar_years)
    result = Simulator(portfolio, [fund], policy).run()

    tracked = result.tracked_values()
    assert list(tracked.columns) == ["liquid_only", "expected_liquid", "liquid_close", "private_close", "total_close"]
    pd.testing.assert_frame_equal(tracked, result.periods[tracked.columns])  # a view of the period table, nothing new

    # 1 · the liquid returns alone: the index itself, whatever the fund calls
    np.testing.assert_allclose(tracked["liquid_only"], [1_000_000, 1_100_000, 1_100_000])
    # 2 · the same start growing at 5% a year, on the anniversaries of the first observation
    np.testing.assert_allclose(tracked["expected_liquid"], [1_000_000, 1_050_000, 1_102_500])
    # 3 · what the account really holds: 22,000 was called at the end of 2028 and never came back
    np.testing.assert_allclose(tracked["liquid_close"], [1_000_000, 1_078_000, 1_078_000])
    # 4 · the private book: at cost, then at its mark
    np.testing.assert_allclose(tracked["private_close"], [0, 22_000, 25_300])
    # 5 · the two together
    np.testing.assert_allclose(tracked["total_close"], tracked["liquid_close"] + tracked["private_close"])
    identities(result)


def test_expected_liquid_is_nan_without_an_expected_return(usd_portfolio, worked_funds):
    result = Simulator(usd_portfolio, worked_funds).run()
    assert result.tracked_values()["expected_liquid"].isna().all()  # no expected return: no expected path


# ------------------------------------------------------- worked examples
def test_worked_example_in_usd(usd_portfolio, worked_funds, identities):
    policy = AnnualRatePolicy(usd_portfolio.commitment_rates, worked_funds, weights={"A": 0.6, "B": 0.4})
    result = Simulator(usd_portfolio, worked_funds, policy).run()
    p = result.periods
    assert result.status == "completed"
    # sized on the liquid-only value: the index itself, not the account that has by then paid A's call
    np.testing.assert_allclose(p["liquid_only"], [1_000_000, 1_100_000, 1_210_000])
    np.testing.assert_allclose(p["commitments"], [0, 66_000, 48_400])
    np.testing.assert_allclose(p["calls"], [0, 16_500, 12_100])
    np.testing.assert_allclose(p["distributions"], [0, 0, 3_300])
    np.testing.assert_allclose(p["liquid_close"], [1_000_000, 1_083_500, 1_183_050])
    np.testing.assert_allclose(p["private_close"], [0, 16_500, 25_300])
    np.testing.assert_allclose(p["total_close"], [1_000_000, 1_100_000, 1_208_350])
    assert (p["usd_rate"] == 1).all() and (p["fx_translation"] == 0).all()
    np.testing.assert_array_equal(p["liquid_only_usd"], p["liquid_only"])  # a dollar portfolio: one and the same
    c = result.commitments
    assert list(c.index) == [(D("2027-03-31"), "A"), (D("2027-06-30"), "B")]
    assert c["commitment_usd"].tolist() == pytest.approx([66_000, 48_400])
    assert c["closing_date"].tolist() == [D("2027-02-15"), D("2027-05-10")]
    assert c["policy_year"].tolist() == [2027, 2027] and c["rate"].tolist() == pytest.approx([0.06, 0.04])
    assert c["weight"].tolist() == [0.6, 0.4] and c["own_year_rate"].tolist() == [0.1, 0.1]
    assert c["own_year_usd"].tolist() == pytest.approx([110_000, 121_000])  # 10% of the liquid-only value, before weight
    assert c["expected_value"].isna().all()  # no expected return given: the rates are shares of the liquid value already
    assert (c["other_years_usd"] == 0).all() and (c["drawn_years"] == "2027").all()  # each fund draws its own year alone
    f = result.funds
    assert f.loc[(D("2027-06-30"), "A"), "nav_usd"] == pytest.approx(13_200)
    assert f.loc[(D("2027-06-30"), "B"), "calls_usd"] == pytest.approx(12_100)
    assert (D("2027-03-31"), "B") not in f.index  # B does not exist before its closing
    identities(result)


def test_worked_example_in_gbp_translates_at_the_observation_rate(gbp_portfolio, worked_funds, identities):
    policy = AnnualRatePolicy(gbp_portfolio.commitment_rates, worked_funds, weights={"A": 0.6, "B": 0.4})
    result = Simulator(gbp_portfolio, worked_funds, policy).run()
    p = result.periods
    assert result.base_currency == "GBP"
    np.testing.assert_allclose(p["usd_rate"], [0.80, 0.80, 0.75])
    np.testing.assert_allclose(p["liquid_only"], [1_000_000, 1_100_000, 1_210_000])  # the liquid-only value, in sterling
    np.testing.assert_allclose(p["liquid_only_usd"], [1_250_000, 1_375_000, 1_210_000 / 0.75])  # what the rate is applied to
    np.testing.assert_allclose(p["commitments"], [0, 66_000, 48_400])
    np.testing.assert_allclose(p["commitments_usd"], [0, 82_500, 48_400 / 0.75])
    np.testing.assert_allclose(p["distributions"], [0, 0, 3_093.75])
    np.testing.assert_allclose(p["calls"], [0, 16_500, 12_100])
    np.testing.assert_allclose(p["liquid_close"], [1_000_000, 1_083_500, 1_182_843.75])
    np.testing.assert_allclose(p["private_close"], [0, 16_500, 24_475])
    np.testing.assert_allclose(p["total_close"], [1_000_000, 1_100_000, 1_207_318.75])
    np.testing.assert_allclose(p["fx_translation"], [0, 0, -1_031.25])
    np.testing.assert_allclose(p["private_valuation_pnl"], [0, 0, -1_031.25])  # no marks: all of it is FX
    c = result.commitments
    assert c.loc[(D("2027-03-31"), "A"), "commitment_usd"] == pytest.approx(82_500)
    assert c.loc[(D("2027-06-30"), "B"), "commitment_usd"] == pytest.approx(64_533.3333333)
    assert c["usd_rate"].tolist() == [0.80, 0.75]
    assert c.loc[(D("2027-03-31"), "A"), "sizing_base_usd"] == pytest.approx(1_375_000)  # 6% of it is the 82,500
    assert c["commitment_base"].tolist() == pytest.approx([66_000, 48_400])  # reported, never decided
    a = result.funds.xs("A", level="fund")
    np.testing.assert_allclose(a["nav_usd"], [20_625, 16_500])
    np.testing.assert_allclose(a["nav_base"], [16_500, 12_375])
    identities(result)


def carry_forward_setup(carry_forward=True):
    portfolio = usd(levels(("2027-12-31", 1_000_000), ("2028-12-31", 1_250_000), ("2029-03-31", 1_250_000), ("2029-06-30", 1_500_000)),
                    {"BUYOUT": {2027: 0.10, 2028: 0.08, 2029: 0.12}})
    funds = [Fund("C", "BUYOUT", "2029-03-01"), Fund("D", "BUYOUT", "2029-06-01")]
    policy = AnnualRatePolicy(portfolio.commitment_rates, funds, {"C": 0.6, "D": 0.4},
                              carry_forward=carry_forward, years=portfolio.calendar_years)
    return Simulator(portfolio, funds, policy).run()


def test_carry_forward_sizes_each_year_then_accumulates_the_dollars(identities):
    result = carry_forward_setup()
    c = result.commitments
    # 2027: 10% of 1,000,000 at its year end; 2028: 8% of 1,250,000 at its year end; both wait for 2029's funds
    assert c["other_years_usd"].tolist() == pytest.approx([200_000, 200_000]) and c["drawn_years"].tolist() == ["2027, 2028, 2029"] * 2
    assert c["own_year_usd"].tolist() == pytest.approx([150_000, 180_000])  # 12% where each fund closes
    assert c["commitment_usd"].tolist() == pytest.approx([210_000, 152_000])    # 60% and 40% of (carried + this year's)
    assert c["own_year_rate"].tolist() == [0.12, 0.12] and c["weight"].tolist() == [0.6, 0.4]
    np.testing.assert_allclose(c["commitment_usd"], c["weight"] * (c["own_year_usd"] + c["other_years_usd"]))
    np.testing.assert_allclose(c["rate"], c["commitment_usd"] / c["sizing_base_usd"])  # a share of the closing balance
    assert result.periods["commitments_usd"].tolist() == pytest.approx([0, 0, 210_000, 152_000])  # nothing moves before a fund exists
    identities(result)


def test_without_carry_forward_the_years_with_no_fund_are_lost(identities):
    c = carry_forward_setup(carry_forward=False).commitments
    assert c["commitment_usd"].tolist() == pytest.approx([0.6 * 150_000, 0.4 * 180_000])
    assert (c["other_years_usd"] == 0).all()


def test_carried_budgets_are_sized_in_dollars_at_each_year_ends_exchange_rate(identities):
    portfolio = Portfolio("GBP", levels(("2027-12-31", 800_000), ("2028-12-31", 900_000), ("2029-12-31", 1_000_000)),
                          {"VC": {2027: 0.10, 2028: 0.10, 2029: 0.10}},
                          usd_rate=[("2027-12-31", 0.80), ("2028-12-31", 0.75), ("2029-12-31", 0.50)])
    fund = Fund("V", "VC", "2029-12-31")
    policy = AnnualRatePolicy(portfolio.commitment_rates, [fund], carry_forward=True, years=portfolio.calendar_years)
    row = Simulator(portfolio, [fund], policy).run().commitments.iloc[0]
    # 10% of $1,000,000 (800,000 / 0.80), then 10% of $1,200,000 (900,000 / 0.75): dollars, fixed when sized
    assert row["other_years_usd"] == pytest.approx(100_000 + 120_000)
    assert row["own_year_usd"] == pytest.approx(0.10 * 1_000_000 / 0.50)
    assert row["commitment_usd"] == pytest.approx(420_000) and row["commitment_base"] == pytest.approx(210_000)


def test_pacing_schedule_is_normalised_by_the_expected_value_seeded_at_the_first_commitment(identities):
    portfolio = usd(levels(("2027-12-31", 1_000_000), ("2028-12-31", 1_100_000), ("2029-12-31", 1_100_000), ("2030-12-31", 1_331_000)),
                    {"BUYOUT": {2027: 0.0, 2028: 0.02, 2029: 0.021, 2030: 0.02205}})
    funds = [Fund("P1", "BUYOUT", "2028-12-31", unit_calls=[("2028-12-31", 1.0)]),  # called in full: the account falls behind the index
             Fund("P2", "BUYOUT", "2029-12-31"), Fund("P3", "BUYOUT", "2030-12-31")]
    policy = AnnualRatePolicy(portfolio.commitment_rates, funds, expected_return=0.05, years=portfolio.calendar_years)
    simulator = Simulator(portfolio, funds, policy)
    assert simulator.first_commitment_date == date(2028, 12, 31)  # the pacing model's value is 1 here
    result = simulator.run()
    c = result.commitments
    np.testing.assert_allclose(c["expected_value"], [1.0, 1.05, 1.1025])
    np.testing.assert_allclose(c["rate"], [0.02, 0.02, 0.02])  # a schedule growing at X is a constant share of the liquid value
    np.testing.assert_allclose(c["commitment_usd"], [22_000, 22_000, 26_620])
    np.testing.assert_allclose(c["sizing_base_usd"], [1_100_000, 1_100_000, 1_331_000])  # the index, not the account P1 drew on
    assert result.periods["liquid_close"].iloc[1] == pytest.approx(1_100_000 - 22_000)
    identities(result)


def test_every_commitment_follows_from_the_liquid_path_alone():
    """The same liquid returns, initial value and schedule give the same commitments whatever the funds call or distribute."""
    portfolio = usd(levels(("2027-12-31", 1_000_000), ("2028-12-31", 1_100_000), ("2029-12-31", 1_250_000)),
                    {"BUYOUT": {2027: 0.0, 2028: 0.05, 2029: 0.05}})
    quiet = [Fund("P1", "BUYOUT", "2028-12-31"), Fund("P2", "BUYOUT", "2029-12-31")]
    busy = [Fund("P1", "BUYOUT", "2028-12-31", unit_calls=[("2028-12-31", 1.0)], unit_distributions=[("2029-06-30", 2.5)],
                 unit_nav=[("2029-06-30", 0.0)]),
            Fund("P2", "BUYOUT", "2029-12-31", unit_calls=[("2029-12-31", 0.5)])]
    first, second = (Simulator(portfolio, funds).run().commitments["commitment_usd"].tolist() for funds in (quiet, busy))
    assert first == second == pytest.approx([55_000, 62_500])


def test_runs_with_carry_forward_repeat_exactly():
    portfolio = usd(levels(("2027-12-31", 1_000_000), ("2028-12-31", 1_250_000), ("2029-12-31", 1_500_000)),
                    {"BUYOUT": {2027: 0.10, 2028: 0.08, 2029: 0.12}})
    fund = Fund("C", "BUYOUT", "2029-12-31")
    simulator = Simulator(portfolio, [fund], AnnualRatePolicy(portfolio.commitment_rates, [fund], carry_forward=True))
    first, second = simulator.run(), simulator.run()  # the policy keeps no state between runs
    pd.testing.assert_frame_equal(first.commitments, second.commitments, check_exact=True)
    assert first.commitments["commitment_usd"].iloc[0] == pytest.approx(100_000 + 100_000 + 180_000)


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
    assert p["period_return"].tolist() == [0.0, pytest.approx(0.1)]
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
    assert row["liquid_only"] == 100 and row["commitments"] == 50  # its own distribution did not inflate the base
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


class FixedDollarsRecordingWhatItSaw:
    """Commits the same number of dollars to every fund, and keeps the balances it was shown."""

    def __init__(self, dollars):
        self.dollars, self.seen = dollars, []

    def size_commitments(self, cohort, balances):
        self.seen.append(balances)
        return {f.name: self.dollars for f in cohort}


def test_sizing_happens_in_dollars_whatever_the_base_currency(gbp_portfolio, worked_funds, identities):
    policy = FixedDollarsRecordingWhatItSaw(50_000.0)
    result = Simulator(gbp_portfolio, worked_funds, policy).run()
    at_a_closing, at_b_closing = policy.seen  # asked only when a fund closes
    # the sterling balance reaches the policy converted at that day's rate; private NAV is dollars as they are
    assert at_a_closing.liquid_only_usd == pytest.approx(1_100_000 / 0.80) and at_a_closing.private_nav_usd == 0
    assert at_b_closing.liquid_only_usd == pytest.approx(1_210_000 / 0.75)  # the index, untouched by A's call and distribution
    # the account has paid £10,000 of calls and banked £1,875: it is shown, in dollars, but it is not what gets sized on
    assert at_b_closing.liquid_account_usd == pytest.approx(((1_100_000 - 10_000) * 1.1 + 1_875) / 0.75)
    assert at_b_closing.private_nav_usd == pytest.approx(50_000 * 0.25)  # A's NAV, untouched by the 0.80 → 0.75 move
    assert at_b_closing.total_usd == pytest.approx(at_b_closing.liquid_account_usd + 12_500)
    assert at_b_closing.first_commitment_date == date(2027, 3, 31) and list(at_b_closing.year_ends) == []  # no year completed yet
    c, p = result.commitments, result.periods
    assert c["commitment_usd"].tolist() == [50_000.0, 50_000.0]  # exactly the dollars decided, whatever the rate
    assert c["commitment_base"].tolist() == pytest.approx([50_000 * 0.80, 50_000 * 0.75])  # the same dollars, in sterling
    np.testing.assert_allclose(c["sizing_base_usd"], c["sizing_base"] / c["usd_rate"])
    np.testing.assert_allclose(c["rate"], c["commitment_usd"] / c["sizing_base_usd"])
    np.testing.assert_allclose(p["liquid_only_usd"], p["liquid_only"] / p["usd_rate"])
    np.testing.assert_allclose(p["commitments"], p["commitments_usd"] * p["usd_rate"])
    assert p["calls"].iloc[1] == pytest.approx(50_000 * 0.25 * 0.80)  # dollars called, paid in sterling at that day's rate
    identities(result)


def test_fund_order_does_not_change_any_result(gbp_portfolio, worked_funds):
    a, b = worked_funds
    forward = Simulator(gbp_portfolio, [a, b], AnnualRatePolicy(gbp_portfolio.commitment_rates, [a, b], {"A": .6, "B": .4})).run()
    reverse = Simulator(gbp_portfolio, [b, a], AnnualRatePolicy(gbp_portfolio.commitment_rates, [b, a], {"A": .6, "B": .4})).run()
    # each fund's market-value columns sit in the order the funds were given in; no value may differ
    assert list(reverse.market_values().columns)[-4:] == ["B_nav_usd", "B_nav_base", "A_nav_usd", "A_nav_base"]
    pd.testing.assert_frame_equal(forward.periods, reverse.periods[forward.periods.columns], check_exact=True)
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


def test_full_cash_use_is_not_a_shortfall_and_an_empty_account_does_not_shrink_commitments(identities):
    portfolio = usd(levels(("2027-01-01", 100), ("2027-02-01", 100), ("2027-03-01", 100)), {"A": {2027: 1.0}, "B": {2027: 1.0}})
    funds = [Fund("F", "A", "2027-01-01", unit_calls=[("2027-01-15", 1.0)]), Fund("G", "B", "2027-02-15")]
    result = Simulator(portfolio, funds).run()
    assert result.status == "completed"  # exactly zero cash is not a shortfall
    np.testing.assert_array_equal(result.periods["liquid_close"], [100, 0, 0])
    g = result.commitments.loc[(D("2027-03-01"), "G")]
    # F's call emptied the account, but commitments are sized on the liquid-only value, which no call touches
    assert g["sizing_base_usd"] == 100 and g["commitment_usd"] == 100 and g["rate"] == 1.0
    identities(result)


def test_cash_tolerance_absorbs_rounding_only():
    portfolio = usd(levels(("2027-01-01", 100), ("2027-02-01", 100)), {"VC": {2027: 1.0}})
    fund = Fund("S", "VC", "2027-01-01", unit_calls=[("2027-02-01", 1.0 + 1e-13)])
    assert Simulator(portfolio, [fund]).run().status == "completed"
    assert Simulator(portfolio, [fund], cash_tolerance=0.0).run().status == "shortfall"
    assert Simulator(portfolio, [fund], cash_tolerance=np.int64(0)).cash_tolerance == 0.0  # numpy ints are numbers
    for bad in (-1, float("nan"), True, "0"):  # a bool is not a tolerance
        with pytest.raises(ValueError, match="cash_tolerance"):
            Simulator(portfolio, [fund], cash_tolerance=bad)


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
        def size_commitments(self, cohort, balances):
            return {"A": -1.0}

    class Stranger:
        def size_commitments(self, cohort, balances):
            return {"A": 1.0, "Z": 1.0}

    with pytest.raises(ValueError, match="invalid commitment"):
        Simulator(portfolio, [fund], Negative()).run()
    with pytest.raises(ValueError, match=r"not closing now: \['Z'\]"):
        Simulator(portfolio, [fund], Stranger()).run()


def test_custom_policy_without_explain_still_reports_rate(identities):
    portfolio = usd(levels(("2027-01-01", 200), ("2027-02-01", 200)), {"VC": {2027: 0.0}})

    class FlatDollars:
        def size_commitments(self, cohort, balances):
            return {f.name: 50.0 for f in cohort}

    result = Simulator(portfolio, [Fund("A", "VC", "2027-01-15")], FlatDollars()).run()
    row = result.commitments.iloc[0]
    assert row["commitment_usd"] == 50 and row["rate"] == pytest.approx(0.25)
    assert np.isnan(row["weight"]) and np.isnan(row["other_years_usd"]) and row["drawn_years"] == ""  # nothing to explain
    assert result.draws.empty  # and nothing to break down either
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
    np.testing.assert_allclose(exposures["B"], [0, 0, 12_100])
    by_type = result.totals_by_fund_type()
    assert list(by_type.index.names) == ["date", "fund_type"]
    assert by_type.loc[(D("2027-06-30"), "BUYOUT"), "nav_base"] == pytest.approx(24_475)
    assert by_type.loc[(D("2027-06-30"), "BUYOUT"), "commitment_usd"] == pytest.approx(82_500 + 48_400 / 0.75)


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
    # the fixed columns, then the market values: the one fund type's total, then funds A and B, each in USD and base
    assert list(result.periods.columns) == PERIOD_COLUMNS + [
        "BUYOUT_total_nav_usd", "BUYOUT_total_nav_base", "A_nav_usd", "A_nav_base", "B_nav_usd", "B_nav_base"]
    assert list(result.market_values().columns) == list(result.periods.columns)[len(PERIOD_COLUMNS):]
    assert list(result.funds.columns) == FUND_COLUMNS
    assert list(result.commitments.columns) == COMMITMENT_COLUMNS
    assert result.periods.index.name == "date" and isinstance(result.periods.index, pd.DatetimeIndex)
    assert result.commitments["policy_year"].dtype == "int64"
    assert str(result.commitments["closing_date"].dtype).startswith("datetime64")
    assert result.periods.dtypes.eq(float).all()


# ------------------------------------------------------------- the draws table
def test_a_drawn_year_after_the_closing_is_priced_on_its_own_year_end():
    """The run looks ahead: a year drawn before it has happened is priced on what the portfolio will be worth then."""
    portfolio = usd(levels(("2027-12-31", 1_000_000), ("2028-12-31", 2_000_000), ("2029-12-31", 4_000_000)),
                    {"VC": {2027: 0.10, 2028: 0.10, 2029: 0.10}})
    fund = Fund("V", "VC", "2027-12-31")
    # V closes at the end of 2027 and draws 2027, 2028 and 2029 — two of them years the run has not reached
    policy = AnnualRatePolicy(portfolio.commitment_rates, [fund], years=portfolio.calendar_years,
                              draws={"V": {2027: 1.0, 2028: 1.0, 2029: 1.0}})
    result = Simulator(portfolio, [fund], policy).run()

    row = result.commitments.iloc[0]
    # 10% of each year's own year-end value: 1,000,000 then 2,000,000 then 4,000,000 — the last two
    # seen with hindsight from a commitment made on 31 Dec 2027
    assert row["commitment_usd"] == pytest.approx(700_000.0) and row["drawn_years"] == "2027, 2028, 2029"
    assert row["own_year_usd"] == pytest.approx(100_000.0) and row["other_years_usd"] == pytest.approx(600_000.0)
    assert result.commitments.index[0][0] == pd.Timestamp("2027-12-31")  # and it is all committed on the closing day

    draws = result.draws.reset_index()
    assert draws["year"].tolist() == [2027, 2028, 2029]
    assert draws["sizing_date"].tolist() == [pd.Timestamp("2027-12-31"), pd.Timestamp("2028-12-31"), pd.Timestamp("2029-12-31")]
    assert draws["liquid_only_usd"].tolist() == [1_000_000, 2_000_000, 4_000_000]
    assert draws["looks_ahead"].tolist() == [False, True, True] and draws["looks_ahead"].dtype == bool
    assert draws["commitment_usd"].tolist() == pytest.approx([100_000, 200_000, 400_000])
    assert draws["commitment_usd"].sum() == pytest.approx(row["commitment_usd"])


def test_a_plan_naming_a_year_beyond_the_liquid_series_stops_the_run_with_a_clear_error():
    portfolio = usd(levels(("2027-12-31", 1_000_000), ("2028-12-31", 2_000_000)),
                    {"VC": {2027: 0.10, 2028: 0.10, 2029: 0.10}})
    fund = Fund("V", "VC", "2027-12-31")
    policy = AnnualRatePolicy(portfolio.commitment_rates, [fund], years=portfolio.calendar_years,
                              draws={"V": {2027: 1.0, 2029: 1.0}})  # the rates reach 2029; the liquid series stops in 2028
    with pytest.raises(ValueError, match="'V' draws 2029, but the liquid series has no observation in 2029"):
        Simulator(portfolio, [fund], policy).run()


def test_a_multiplier_shows_as_one_row_carrying_it():
    portfolio = usd(levels(("2027-12-31", 1_000_000), ("2028-12-31", 1_000_000)), {"VC": {2027: 0.10, 2028: 0.10}})
    fund = Fund("V", "VC", "2028-12-31")
    policy = AnnualRatePolicy(portfolio.commitment_rates, [fund], years=portfolio.calendar_years,
                              draws={"V": {2028: 3.0}})
    result = Simulator(portfolio, [fund], policy).run()
    assert result.commitments.iloc[0]["commitment_usd"] == pytest.approx(300_000.0)
    assert result.commitments.iloc[0]["drawn_years"] == "2028x3"
    draw = result.draws.iloc[0]
    assert draw["multiplier"] == 3.0 and draw["rate"] == 0.10 and draw["commitment_usd"] == pytest.approx(300_000.0)


# ------------------------------------------- market values on every period row
def test_every_period_row_carries_each_funds_and_each_fund_types_market_value_in_both_currencies(identities):
    portfolio = Portfolio("GBP", levels(("2027-01-31", 1_000_000), ("2027-02-28", 1_000_000), ("2027-03-31", 1_000_000)),
                          {"BUYOUT": {2027: 0.10}, "SECONDARIES": {2027: 0.05}},
                          usd_rate=[("2027-01-31", 0.80), ("2027-02-28", 0.75), ("2027-03-31", 0.50)])
    funds = [
        Fund("B1", "BUYOUT", "2027-01-31", unit_calls=[("2027-01-31", 0.5)], unit_nav=[("2027-03-31", 0.8)]),
        Fund("S1", "SECONDARIES", "2027-02-28", unit_calls=[("2027-02-28", 1.0)]),
        Fund("B2", "BUYOUT", "2027-02-28", unit_calls=[("2027-02-28", 0.25)]),
    ]
    policy = AnnualRatePolicy(portfolio.commitment_rates, funds, weights={"B1": 0.5, "B2": 0.5},
                              years=portfolio.calendar_years)
    result = Simulator(portfolio, funds, policy).run()
    p = result.periods

    # fund types in the order they first appear, then the funds in the order given
    assert list(result.market_values().columns) == [
        "BUYOUT_total_nav_usd", "BUYOUT_total_nav_base", "SECONDARIES_total_nav_usd", "SECONDARIES_total_nav_base",
        "B1_nav_usd", "B1_nav_base", "S1_nav_usd", "S1_nav_base", "B2_nav_usd", "B2_nav_base"]

    # the two buyout funds split 2027's 10% equally, each sized where it closes:
    # B1: half of 10% of $1,250,000 committed, half of that called, then marked at 0.8
    # B2: half of 10% of $1,333,333, a quarter called · S1: 5% of $1,333,333, all called
    # A fund is worth nothing until it is committed.
    b1 = 0.5 * 0.10 * 1_000_000 / 0.80
    b2 = 0.5 * 0.10 * 1_000_000 / 0.75
    s1 = 0.05 * 1_000_000 / 0.75
    assert p["B1_nav_usd"].tolist() == pytest.approx([0.5 * b1, 0.5 * b1, 0.8 * b1])
    assert p["B2_nav_usd"].tolist() == pytest.approx([0.0, 0.25 * b2, 0.25 * b2])
    assert p["S1_nav_usd"].tolist() == pytest.approx([0.0, s1, s1])

    # the base-currency value is the dollar value at that observation's rate: it moves with the rate alone
    assert p["S1_nav_base"].tolist() == pytest.approx([0.0, s1 * 0.75, s1 * 0.50])

    # a fund type's total is the sum of its funds, in each currency
    assert p["BUYOUT_total_nav_usd"].tolist() == pytest.approx((p["B1_nav_usd"] + p["B2_nav_usd"]).tolist())
    assert p["BUYOUT_total_nav_base"].tolist() == pytest.approx((p["B1_nav_base"] + p["B2_nav_base"]).tolist())
    assert p["SECONDARIES_total_nav_usd"].tolist() == pytest.approx(p["S1_nav_usd"].tolist())

    # and the types together are the private book
    totals = p["BUYOUT_total_nav_base"] + p["SECONDARIES_total_nav_base"]
    assert totals.tolist() == pytest.approx(p["private_close"].tolist())
    by_type = result.totals_by_fund_type()["nav_usd"].unstack("fund_type")
    assert p.loc[by_type.index, "BUYOUT_total_nav_usd"].tolist() == pytest.approx(by_type["BUYOUT"].tolist())
    identities(result)


def test_a_fund_beyond_the_horizon_keeps_its_columns_at_zero_and_a_run_without_funds_has_none():
    portfolio = usd(levels(("2027-01-31", 100.0), ("2027-02-28", 100.0)), {"VC": {2027: 0.1, 2028: 0.1}})
    never_committed = Fund("LATE", "VC", "2028-06-30")
    result = Simulator(portfolio, [never_committed]).run()
    assert result.funds_beyond_horizon == ("LATE",)
    assert (result.market_values() == 0).all().all()
    assert list(result.market_values().columns) == ["VC_total_nav_usd", "VC_total_nav_base", "LATE_nav_usd", "LATE_nav_base"]

    from pmsim.simulator import PERIOD_COLUMNS
    no_funds = Simulator(portfolio).run()
    assert list(no_funds.periods.columns) == PERIOD_COLUMNS and no_funds.market_values().shape == (2, 0)


def test_a_fund_name_that_collides_with_a_market_value_column_is_refused():
    portfolio = usd(levels(("2027-01-31", 100.0)), {"VC": {2027: 0.1}})
    with pytest.raises(ValueError, match=r"the same column twice: \['VC_total_nav_base', 'VC_total_nav_usd'\]; rename the fund"):
        Simulator(portfolio, [Fund("VC_total", "VC", "2027-01-31")])
