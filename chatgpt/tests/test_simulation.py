from copy import deepcopy
from datetime import date

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from chatgpt.simulation import Simulation, SimulationConfig, SimulationValidationError, ValuationError
from vintage import FundVintage, HistoryEntry, EntryType


def fund(name="A", closing="2027-01-01", strategy="BUYOUT", flows=(), navs=(), commitment=0):
    return FundVintage(name, strategy, commitment_size=commitment, commitment_date=closing,
                       normalized_realized_net_cash_flow=list(flows), normalized_realized_nav=list(navs))


def config(dates, levels, funds=(), rates=None, weights=None, tolerance=1e-8):
    if rates is None:
        years = range(pd.Timestamp(dates[0]).year, pd.Timestamp(dates[-1]).year + 1)
        rates = pd.DataFrame({s: 0.1 for s in {f.strategy for f in funds}}, index=list(years))
    return SimulationConfig(pd.Series(levels, index=pd.to_datetime(dates), dtype=float), funds,
                            rates, weights or {}, tolerance)


def assert_reconciles(result):
    p = result.portfolio
    np.testing.assert_allclose(p.liquid_close, p.liquid_open + p.liquid_investment_pnl
                               + p.distributions - p.capital_calls + p.cash_rounding_adjustment,
                               rtol=1e-12, atol=1e-8)
    np.testing.assert_allclose(p.total_close, p.liquid_close + p.private_nav_close, atol=1e-8)
    np.testing.assert_allclose(p.total_close - p.total_open,
                               p.liquid_investment_pnl + p.private_valuation_pnl + p.cash_rounding_adjustment,
                               rtol=1e-12, atol=1e-8)
    b = result.commitment_budget
    np.testing.assert_allclose(b.cumulative_target_percentage,
                               b.cumulative_used_percentage + b.unallocated_carried_percentage + b.reserved_percentage,
                               rtol=1e-12, atol=1e-12)
    if not result.fund_detail.empty:
        f = result.fund_detail.groupby(level="date").sum(numeric_only=True)
        np.testing.assert_allclose(f.nav_close, p.private_nav_close)
        np.testing.assert_allclose(f.capital_calls, p.capital_calls)
        np.testing.assert_allclose(f.distributions, p.distributions)
    if not result.strategy_detail.empty:
        s = result.strategy_detail.groupby(level="date").sum(numeric_only=True)
        np.testing.assert_allclose(s.nav_close, p.private_nav_close)
        np.testing.assert_allclose(s.new_commitments, p.new_commitments)


def test_no_funds_follows_return_index():
    cfg = config(["2027-01-01", "2027-03-20", "2028-04-17"], [100, 85, 140])
    r = Simulation(cfg).run()
    assert r.status == "completed"
    np.testing.assert_allclose(r.portfolio.liquid_close, cfg.liquid_total_return_index)
    assert r.fund_detail.empty and r.commitment_events.empty
    assert r.portfolio.private_nav_close.eq(0).all()
    assert_reconciles(r)


def test_plan_worked_example():
    a = fund("A", "2027-02-15", flows=[("2027-03-01", -.25), ("2027-06-01", .05)])
    b = fund("B", "2027-05-10", flows=[("2027-05-20", -.25)])
    cfg = config(["2027-01-01", "2027-03-31", "2027-06-30"], [1e6, 1.1e6, 1.21e6],
                 [a, b], weights={"A": .6, "B": .4})
    r = Simulation(cfg).run()
    np.testing.assert_allclose(r.commitment_events.commitment, [66000, 47806])
    np.testing.assert_allclose(r.portfolio.capital_calls, [0, 16500, 11951.5])
    np.testing.assert_allclose(r.portfolio.liquid_close, [1e6, 1083500, 1183198.5])
    np.testing.assert_allclose(r.portfolio.total_close, [1e6, 1100000, 1208350])
    assert r.fund_detail.loc[(pd.Timestamp("2027-06-30"), "A"), "nav_close"] == pytest.approx(13200)
    assert r.commitment_events.iloc[0].actual_closing_date == pd.Timestamp("2027-02-15")
    assert_reconciles(r)


def test_carryforward_weighted_closings_use_different_navs():
    c, d, e = fund("C", "2029-03-01"), fund("D", "2029-06-01"), fund("E", "2030-01-01")
    rates = pd.DataFrame({"BUYOUT": [.10, .08, .12, .05]}, index=[2027, 2028, 2029, 2030])
    cfg = config(["2027-01-01", "2028-12-31", "2029-03-31", "2029-06-30", "2030-01-01"],
                 [1e6, 1e6, 1e6, 1.2e6, 1.2e6], [c, d, e], rates, {"C": .6, "D": .4})
    r = Simulation(cfg).run()
    np.testing.assert_allclose(r.commitment_events.effective_rate, [.18, .12, .05])
    np.testing.assert_allclose(r.commitment_events.commitment, [180000, 144000, 60000])
    budget = r.commitment_budget.xs("BUYOUT", level="strategy")
    np.testing.assert_allclose(budget.unallocated_carried_percentage, [.10, .18, 0, 0, 0])
    np.testing.assert_allclose(budget.reserved_percentage, [0, 0, .12, 0, 0], atol=1e-15)
    assert r.commitment_events.iloc[0].source_year_rates == pytest.approx({2027: .06, 2028: .048, 2029: .072})
    assert_reconciles(r)


def test_types_are_independent_and_policy_only_types_accumulate():
    rates = pd.DataFrame({"BUYOUT": [.1, .1, .1], "SECONDARIES": [.2, .2, .2],
                          "CREDIT": [.03, .03, .03]}, index=[2027, 2028, 2029])
    cfg = config(["2027-01-01", "2029-12-31"], [100, 100],
                 [fund("B", "2029-05-01"), fund("S", "2028-06-01", "SECONDARIES")], rates)
    r = Simulation(cfg).run()
    e = r.commitment_events.reset_index().set_index("fund_id")
    assert e.loc["B", "effective_rate"] == pytest.approx(.3)
    assert e.loc["S", "effective_rate"] == pytest.approx(.4)
    last = r.commitment_budget.loc[pd.Timestamp("2029-12-31")]
    assert last.loc["SECONDARIES", "unallocated_carried_percentage"] == pytest.approx(.2)
    assert last.loc["CREDIT", "unallocated_carried_percentage"] == pytest.approx(.09)
    assert_reconciles(r)


def test_zero_current_rate_uses_carry_and_target_accrues_only_once():
    dates = ["2027-01-01", "2027-02-01", "2027-03-01", "2028-04-01"]
    rates = pd.DataFrame({"BUYOUT": [.1, 0]}, index=[2027, 2028])
    r = Simulation(config(dates, [100] * 4, [fund(closing="2028-03-01")], rates)).run()
    assert r.commitment_events.iloc[0].commitment == pytest.approx(10)
    np.testing.assert_allclose(r.commitment_budget.cumulative_target_percentage, [.1] * 4)
    assert_reconciles(r)


def test_first_date_closing_flows_and_nav_mark():
    a = fund(flows=[("2027-01-01", -.5)], navs=[("2027-01-01", .55)])
    r = Simulation(config(["2027-01-01", "2027-02-01"], [1000, 1100], [a])).run()
    np.testing.assert_allclose(r.portfolio.capital_calls, [50, 0])
    np.testing.assert_allclose(r.portfolio.liquid_close, [950, 1045])
    np.testing.assert_allclose(r.portfolio.private_nav_close, [55, 55])
    assert_reconciles(r)


def test_actual_year_preserved_across_new_year_and_same_observation():
    a, b = fund("A", "2027-12-20"), fund("B", "2028-01-05")
    rates = pd.DataFrame({"BUYOUT": [.1, .2]}, index=[2027, 2028])
    r = Simulation(config(["2027-01-01", "2028-01-31"], [100, 200], [b, a], rates)).run()
    np.testing.assert_allclose(r.commitment_events.commitment, [20, 40])
    np.testing.assert_allclose(r.commitment_events.pooled_rate, [.1, .2])
    assert r.commitment_events.policy_year.tolist() == [2027, 2028]
    assert_reconciles(r)


def test_fund_order_does_not_change_same_period_sizing_or_flows():
    a = fund("A", "2027-02-01", flows=[("2027-02-01", -.4)])
    b = fund("B", "2027-02-15", flows=[("2027-02-15", -.8)])
    kwargs = {"weights": {"A": .4, "B": .6}}
    one = Simulation(config(["2027-01-01", "2027-03-31"], [1000, 1100], [a, b], **kwargs)).run()
    two = Simulation(config(["2027-01-01", "2027-03-31"], [1000, 1100], [b, a], **kwargs)).run()
    assert_frame_equal(one.portfolio, two.portfolio)
    assert_frame_equal(one.commitment_events, two.commitment_events)
    assert_frame_equal(one.commitment_budget, two.commitment_budget)
    np.testing.assert_allclose(one.commitment_events.sizing_liquid_nav, [1100, 1100])


def test_new_fund_distributions_do_not_inflate_commitment_sizing():
    a = fund(flows=[("2027-01-01", -.5), ("2027-01-01", .2)])
    r = Simulation(config(["2027-01-01"], [1000], [a])).run()
    p = r.portfolio.iloc[0]
    assert p.sizing_liquid_nav == 1000
    assert p.new_commitments == 100
    assert p.distributions_new == 20
    assert p.liquid_close == 970
    assert p.private_nav_close == 30
    assert_reconciles(r)


def test_same_day_gross_flows_preserved_even_when_net_is_zero():
    a = fund(flows=[("2027-01-01", -.1), ("2027-01-01", .1)])
    r = Simulation(config(["2027-01-01"], [1000], [a])).run()
    assert r.portfolio.iloc[0].capital_calls == 10
    assert r.portfolio.iloc[0].distributions == 10
    assert r.portfolio.iloc[0].net_private_cash_flow == 0
    assert r.portfolio.iloc[0].private_nav_close == 0


def test_nav_marks_between_observations_and_post_mark_flow_adjustments():
    a = fund(flows=[("2027-01-10", -.2), ("2027-03-25", .03)],
             navs=[("2027-03-20", .23), ("2027-04-10", 0)])
    r = Simulation(config(["2027-01-01", "2027-03-31", "2027-05-01"], [1000] * 3, [a])).run()
    np.testing.assert_allclose(r.portfolio.private_nav_close, [0, 20, 0])
    assert r.fund_detail.loc[(pd.Timestamp("2027-03-31"), "A"), "latest_nav_mark_date"] == pd.Timestamp("2027-03-20")
    assert_reconciles(r)


def test_same_day_nav_mark_overrides_flow_even_if_intermediate_nav_negative():
    a = fund(flows=[("2027-01-01", .2)], navs=[("2027-01-01", .3)])
    r = Simulation(config(["2027-01-01"], [1000], [a])).run()
    assert r.portfolio.iloc[0].private_nav_close == pytest.approx(30)
    assert_reconciles(r)


def test_quarter_end_month_without_mark_still_adjusts_cash_flows():
    a = fund(flows=[("2027-03-10", -.5), ("2027-06-12", .1)])
    r = Simulation(config(["2027-01-01", "2027-03-31", "2027-06-30"], [1000] * 3, [a])).run()
    np.testing.assert_allclose(r.portfolio.private_nav_close, [0, 50, 40])


def test_shortfall_returns_completed_periods_and_unconsumed_candidate_budget():
    a = fund("A", flows=[("2027-02-01", -6)])
    b = fund("B", "2027-02-01", flows=[("2027-02-01", -20)])
    rates = pd.DataFrame({"BUYOUT": [.5]}, index=[2027])
    r = Simulation(config(["2027-01-01", "2027-02-01", "2027-03-01"], [100] * 3,
                          [a, b], rates, {"A": .5, "B": .5})).run()
    assert r.status == "liquidity_shortfall"
    assert len(r.portfolio) == 1 and len(r.commitment_events) == 1
    assert r.shortfall.deficit == pytest.approx(550)
    assert r.shortfall.calls_required == pytest.approx(650)
    assert r.shortfall.calls_by_fund.to_dict() == {"A": 150, "B": 500}
    budget = r.shortfall.candidate_budget.iloc[0]
    assert budget.cumulative_used_percentage == .25
    assert budget.reserved_percentage == .25
    assert budget.attempted_used_percentage == .25
    assert r.shortfall.candidate_commitments.iloc[0].commitment == 25
    assert_reconciles(r)


def test_first_date_shortfall_returns_empty_tables_with_defined_columns():
    a = fund(flows=[("2027-01-01", -11)])
    r = Simulation(config(["2027-01-01"], [100], [a])).run()
    assert r.status == "liquidity_shortfall"
    assert r.portfolio.empty and "liquid_close" in r.portfolio
    assert r.commitment_budget.empty and "reserved_percentage" in r.commitment_budget
    assert r.shortfall.deficit == pytest.approx(10)
    assert r.shortfall.candidate_budget.iloc[0].cumulative_used_percentage == 0


def test_early_shortfall_not_hidden_by_future_negative_nav():
    a = fund(flows=[("2027-01-01", -11), ("2027-03-01", 100)])
    r = Simulation(config(["2027-01-01", "2027-03-31"], [100, 100], [a])).run()
    assert r.status == "liquidity_shortfall"


def test_material_negative_nav_reports_fund_and_actual_event_date():
    a = fund(flows=[("2027-02-15", .1)])
    with pytest.raises(ValuationError) as caught:
        Simulation(config(["2027-01-01", "2027-03-31"], [100, 100], [a])).run()
    assert caught.value.fund_id == "A"
    assert caught.value.event_date == date(2027, 2, 15)
    assert caught.value.nav == pytest.approx(-1)


def test_negative_nav_between_marks_cannot_be_hidden_by_later_mark():
    a = fund(flows=[("2027-02-15", .1)], navs=[("2027-03-01", .2)])
    with pytest.raises(ValuationError):
        Simulation(config(["2027-01-01", "2027-03-31"], [100, 100], [a])).run()


def test_full_cash_use_and_zero_liquid_sizing_consumes_percentage():
    a = fund("A", flows=[("2027-01-01", -1)])
    b = fund("B", "2028-01-01")
    rates = pd.DataFrame({"BUYOUT": [1, .2]}, index=[2027, 2028])
    r = Simulation(config(["2027-01-01", "2028-01-01"], [100, 110], [a, b], rates)).run()
    assert r.status == "completed"
    np.testing.assert_allclose(r.portfolio.liquid_close, [0, 0])
    assert r.commitment_events.iloc[1].commitment == 0
    assert r.commitment_budget.iloc[-1].cumulative_used_percentage == pytest.approx(1.2)
    assert_reconciles(r)


def test_rounding_tolerance_is_reported():
    a = fund(flows=[("2027-01-01", -1.00000000001)])
    cfg = config(["2027-01-01"], [100], [a], pd.DataFrame({"BUYOUT": [1.]}, index=[2027]))
    r = Simulation(cfg).run()
    assert r.status == "completed"
    assert r.portfolio.iloc[0].liquid_close == 0
    assert 0 < r.portfolio.iloc[0].cash_rounding_adjustment < cfg.cash_tolerance
    assert_reconciles(r)


def test_partial_year_does_not_reallocate_outside_horizon_weight():
    a, b = fund("A", "2027-03-01"), fund("B", "2027-12-01")
    r = Simulation(config(["2027-01-01", "2027-06-30"], [100, 100], [a, b],
                          weights={"A": .4, "B": .6})).run()
    assert r.commitment_events.iloc[0].commitment == pytest.approx(4)
    assert r.commitment_budget.iloc[-1].reserved_percentage == pytest.approx(.06)
    assert r.commitment_budget.iloc[-1].unallocated_carried_percentage == 0
    assert any(d.code == "fund_outside_horizon" and d.fund_id == "B" for d in r.diagnostics)
    assert_reconciles(r)


def test_long_sparse_horizon_counts_all_years_and_does_not_cap_pool():
    rates = pd.DataFrame({"BUYOUT": [.1] * 26}, index=range(2027, 2053))
    a = fund(closing="2052-01-01")
    r = Simulation(config(["2027-01-01", "2052-12-31"], [100, 100], [a], rates)).run()
    assert r.commitment_events.iloc[0].commitment == pytest.approx(260)
    assert_reconciles(r)


def test_outside_policy_years_excluded_and_partial_inception_year_not_prorated():
    rates = pd.DataFrame({"BUYOUT": [.9, .1, .8]}, index=[2026, 2027, 2028])
    a = fund(closing="2027-10-01")
    r = Simulation(config(["2027-07-01", "2027-12-31"], [100, 100], [a], rates)).run()
    assert r.commitment_events.iloc[0].commitment == 10
    assert any(d.code == "policy_outside_horizon" for d in r.diagnostics)


def test_config_snapshot_repeatability_and_result_independence():
    a = fund(flows=[("2027-01-01", -.2)], commitment=999)
    cfg = config(["2027-01-01", "2027-02-01"], [100, 110], [a])
    before = deepcopy(a.to_dict())
    sim = Simulation(cfg)
    first = sim.run()
    assert a.to_dict() == before
    cfg.liquid_total_return_index.iloc[1] = 900
    cfg.annual_commitment_rates.iloc[0, 0] = .9
    a.add_call("2027-02-01", 100)
    second = sim.simulate()
    for name in ["portfolio", "fund_detail", "strategy_detail", "commitment_events", "commitment_budget"]:
        assert_frame_equal(getattr(first, name), getattr(second, name))
    first.commitment_budget.iloc[0].used_by_year[2027] = 99
    assert sim.run().commitment_budget.iloc[0].used_by_year[2027] == pytest.approx(.1)


@pytest.mark.parametrize("dates,levels", [
    ([], []), (["2027-01-01", "2027-01-01"], [1, 2]),
    (["2027-02-01", "2027-01-01"], [1, 2]), (["2027-01-01"], [0]),
    (["2027-01-01"], [-1]), (["2027-01-01"], [float("nan")]),
    (["2027-01-01"], [float("inf")]),
    (["2027-01-01 12:00:00"], [100]), (["2027-01-01T00:00:00Z"], [100]),
])
def test_invalid_liquid_inputs(dates, levels):
    cfg = SimulationConfig(pd.Series(levels, index=pd.to_datetime(dates), dtype=float), [], pd.DataFrame())
    with pytest.raises(SimulationValidationError):
        Simulation(cfg)


@pytest.mark.parametrize("rates", [
    pd.DataFrame({"BUYOUT": [.1]}, index=[2027]),
    pd.DataFrame({"BUYOUT": [.1, np.nan]}, index=[2027, 2028]),
    pd.DataFrame({"BUYOUT": [.1, -.1]}, index=[2027, 2028]),
    pd.DataFrame({"BUYOUT": [.1, .1]}, index=["2027", "2028"]),
    pd.DataFrame({"BUYOUT": [.1, .1]}, index=[2027, 2027]),
    pd.DataFrame({"BUYOUT": [.1, np.inf]}, index=[2027, 2028]),
])
def test_invalid_rates_including_no_fund_years(rates):
    with pytest.raises(SimulationValidationError):
        Simulation(config(["2027-01-01", "2028-12-31"], [100, 100], [fund()], rates))


@pytest.mark.parametrize("weights", [{}, {"A": .6}, {"A": .6, "B": .6},
                                      {"A": -1, "B": 2}, {"A": .5, "B": .5, "C": 0}])
def test_invalid_weights(weights):
    with pytest.raises(SimulationValidationError):
        Simulation(config(["2027-01-01"], [100], [fund("A"), fund("B")], weights=weights))


def test_duplicate_names_missing_closing_and_historical_funds_rejected():
    for fs in ([fund(), fund()], [FundVintage("A", "BUYOUT", vintage_year=2027)],
               [fund(closing="2026-01-01")]):
        with pytest.raises(SimulationValidationError):
            Simulation(config(["2027-01-01"], [100], fs))


def test_preclosing_events_and_mutated_duplicate_marks_rejected():
    a = fund(closing="2027-02-01", flows=[("2027-01-15", -.1)])
    with pytest.raises(SimulationValidationError, match="before its closing"):
        Simulation(config(["2027-01-01", "2027-03-01"], [100, 100], [a]))
    b = fund(navs=[("2027-01-01", .1)])
    b.normalized_realized_nav.append(HistoryEntry.create("2027-01-01", .2, EntryType.NAV))
    with pytest.raises(SimulationValidationError, match="Duplicate NAV"):
        Simulation(config(["2027-01-01"], [100], [b]))


def test_simulation_accepts_calendar_date_index():
    cfg = SimulationConfig(pd.Series([100, 120], index=[date(2027, 1, 1), date(2027, 2, 1)]),
                           [], pd.DataFrame())
    np.testing.assert_allclose(Simulation(cfg).run().portfolio.liquid_close, [100, 120])
