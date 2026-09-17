"""Observation frequency.

The liquid index sets the observation grid — month ends, quarter ends, business days,
anything irregular. Fund events are dated on whatever day they happened. One rule covers
every mismatch: an event on any day pools onto the first observation on or after it.
"""
import numpy as np
import pandas as pd
import pytest

from pmsim import Fund, Portfolio, Simulator, Timeline

MONTH_ENDS = pd.date_range("2027-01-31", "2027-06-30", freq="ME")
QUARTER_ENDS = pd.date_range("2027-03-31", "2027-06-30", freq="QE")
HISTORY = dict(
    unit_calls=[("2027-02-03", 0.1), ("2027-02-15", 0.2), ("2027-02-28", 0.3), ("2027-03-01", 0.05), ("2027-05-20", 0.1)],
    unit_distributions=[("2027-04-02", 0.04), ("2027-06-30", 0.06)],
    unit_nav=[("2027-03-20", 0.7)],
)


def fund(**history):
    return Fund("F", "VC", "2027-01-31", **history)


def month_end_portfolio():
    return Portfolio("USD", [(d, 100.0) for d in MONTH_ENDS], {"VC": {2027: 0.5}})


def test_intramonth_flows_pool_onto_that_month_end():
    path = fund(**HISTORY).align_history(Timeline(MONTH_ENDS))
    np.testing.assert_allclose(path.unit_calls, [0, 0.6, 0.05, 0, 0.1, 0])  # 3, 15 and 28 Feb → 28 Feb; 1 Mar → 31 Mar
    np.testing.assert_allclose(path.unit_distributions, [0, 0, 0, 0.04, 0, 0.06])  # a flow on a month end stays there
    np.testing.assert_allclose(path.unit_nav, [0, 0.6, 0.7, 0.66, 0.76, 0.7])  # the 20 Mar mark is what 31 Mar sees


def test_pooling_is_consistent_across_frequencies():
    monthly = fund(**HISTORY).align_history(Timeline(MONTH_ENDS))
    quarterly = fund(**HISTORY).align_history(Timeline(QUARTER_ENDS))
    np.testing.assert_allclose(quarterly.unit_calls, [monthly.unit_calls[:3].sum(), monthly.unit_calls[3:].sum()])
    np.testing.assert_allclose(quarterly.unit_distributions, [monthly.unit_distributions[:3].sum(), monthly.unit_distributions[3:].sum()])
    np.testing.assert_allclose(quarterly.unit_nav, monthly.unit_nav[[2, 5]])
    single = fund(**HISTORY).align_history(Timeline([MONTH_ENDS[-1]]))
    np.testing.assert_allclose(single.unit_calls, [monthly.unit_calls.sum()])
    np.testing.assert_allclose(single.unit_nav, [monthly.unit_nav[-1]])


def test_daily_grid_moves_weekend_events_to_the_next_pricing_day():
    business_days = Timeline(pd.bdate_range("2027-01-01", "2027-01-31"))
    path = Fund("W", "VC", "2027-01-01", unit_calls=[("2027-01-09", 0.5)]).align_history(business_days)  # a Saturday
    assert list(business_days.dates[np.flatnonzero(path.unit_calls)]) == [pd.Timestamp("2027-01-11")]  # the Monday


def test_irregular_grid_uses_the_same_rule():
    grid = Timeline(["2027-01-05", "2027-01-06", "2027-02-17", "2027-05-30"])
    history = [("2027-01-06", 0.1), ("2027-01-07", 0.2), ("2027-02-17", 0.3), ("2027-02-18", 0.4)]
    path = Fund("I", "VC", "2027-01-05", unit_calls=history).align_history(grid)
    np.testing.assert_allclose(path.unit_calls, [0, 0.1, 0.5, 0.4])


def test_intramonth_closing_is_sized_at_the_month_end_and_pays_that_months_calls(identities):
    result = Simulator(month_end_portfolio(), [Fund("F", "VC", "2027-02-10", unit_calls=[("2027-02-20", 0.4)])]).run()
    c = result.commitments
    assert list(c.index) == [(pd.Timestamp("2027-02-28"), "F")]
    assert c["closing_date"].iloc[0] == pd.Timestamp("2027-02-10") and c["commitment_usd"].iloc[0] == 50
    feb = result.periods.loc["2027-02-28"]
    assert feb["calls"] == pytest.approx(20) and feb["liquid_close"] == pytest.approx(80)
    assert feb["private_close"] == pytest.approx(20)
    identities(result)


def test_quarterly_grid_with_monthly_marks_uses_the_last_mark_then_adjusts_for_flows():
    f = Fund("Q", "VC", "2027-01-01", unit_calls=[("2027-01-10", 0.5)],
             unit_nav=[("2027-01-31", 0.55), ("2027-02-28", 0.6), ("2027-03-15", 0.65)],
             unit_distributions=[("2027-03-25", 0.05), ("2027-05-05", 0.1)])
    path = f.align_history(Timeline(QUARTER_ENDS))
    np.testing.assert_allclose(path.unit_nav, [0.60, 0.50])  # 0.65 mark then −0.05; then −0.10 with no newer mark


def test_event_map_shows_where_each_event_landed():
    f = Fund("F", "VC", "2027-01-31", unit_calls=[("2027-02-03", 0.1), ("2027-07-15", 0.2)], unit_nav=[("2027-03-20", 0.7)])
    events = Simulator(month_end_portfolio(), [f]).map_events_to_observations()
    assert list(events.index.names) == ["fund", "event_date"]
    feb = events.loc[("F", pd.Timestamp("2027-02-03"))]
    assert feb["observation_date"] == pd.Timestamp("2027-02-28") and feb["period"] == 1 and feb["unit_call"] == 0.1
    mark = events.loc[("F", pd.Timestamp("2027-03-20"))]
    assert mark["observation_date"] == pd.Timestamp("2027-03-31") and mark["unit_nav_mark"] == 0.7 and mark["unit_call"] == 0
    late = events.loc[("F", pd.Timestamp("2027-07-15"))]
    assert pd.isna(late["observation_date"]) and late["period"] == 6  # beyond the horizon: the run ignores it
    assert Simulator(month_end_portfolio()).map_events_to_observations().empty


def test_timeline_assign_matches_index_of():
    timeline = Timeline(MONTH_ENDS)
    days = ["2027-01-31", "2027-02-01", "2027-02-28", "2027-07-01"]
    assert timeline.first_observations_on_or_after(days).tolist() == [timeline.first_observation_on_or_after(d) for d in days] == [0, 1, 1, 6]


def test_exchange_rates_carry_forward_while_flows_roll_forward():
    # a rate is a state (use the last one known); a flow is an event (settle at the next observation)
    timeline = Timeline(MONTH_ENDS)
    rates = pd.Series([0.8, 0.9], index=pd.to_datetime(["2027-01-15", "2027-03-10"]))
    np.testing.assert_allclose(timeline.last_value_on_or_before(rates), [0.8, 0.8, 0.9, 0.9, 0.9, 0.9])
    assert timeline.first_observations_on_or_after(["2027-03-10"]).tolist() == [2]
