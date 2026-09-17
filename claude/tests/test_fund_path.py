"""Fund.on(timeline): dated history → per-period unit arrays."""
import numpy as np
import pandas as pd
import pytest

from pmsim import Fund, Timeline

GRID = Timeline(pd.to_datetime(["2027-01-31", "2027-02-28", "2027-03-31"]))


def fund(**history):
    return Fund("A", "BUYOUT", "2027-01-01", **history)


def test_flows_bucket_into_prev_exclusive_current_inclusive():
    path = fund(unit_calls=[("2027-01-31", 0.1), ("2027-02-01", 0.2), ("2027-02-28", 0.3), ("2027-03-01", 0.4)]).on(GRID)
    np.testing.assert_allclose(path.calls, [0.1, 0.5, 0.4])
    assert path.closing_index == 0


def test_first_period_takes_everything_on_or_before_the_first_date():
    path = fund(unit_calls=[("2027-01-01", 0.1), ("2027-01-15", 0.2)]).on(GRID)
    np.testing.assert_allclose(path.calls, [0.3, 0.0, 0.0])
    np.testing.assert_allclose(path.nav, [0.3, 0.3, 0.3])


def test_same_day_call_and_distribution_both_survive():
    path = fund(unit_calls=[("2027-02-01", 0.1)], unit_distributions=[("2027-02-01", 0.1)]).on(GRID)
    np.testing.assert_allclose(path.calls, [0, 0.1, 0])
    np.testing.assert_allclose(path.distributions, [0, 0.1, 0])
    np.testing.assert_allclose(path.nav, [0, 0, 0])


def test_nav_walks_events_in_order_and_marks_replace():
    # the design note's example: a 0.20 call, then a 0.23 mark off-grid, then a 0.03 distribution
    path = fund(unit_calls=[("2027-01-15", 0.20)], unit_nav=[("2027-02-10", 0.23)],
                unit_distributions=[("2027-03-05", 0.03)]).on(GRID)
    np.testing.assert_allclose(path.nav, [0.20, 0.23, 0.20])


def test_missing_mark_keeps_cash_adjusted_estimate_and_zero_mark_resets():
    carried = fund(unit_calls=[("2027-01-15", 0.2), ("2027-03-15", 0.1)]).on(GRID)
    np.testing.assert_allclose(carried.nav, [0.2, 0.2, 0.3])
    reset = fund(unit_calls=[("2027-01-15", 0.2)], unit_nav=[("2027-02-10", 0.0)]).on(GRID)
    np.testing.assert_allclose(reset.nav, [0.2, 0.0, 0.0])


def test_same_day_mark_includes_that_days_flows():
    # call then mark on one day: the mark wins, the call is still bucketed
    path = fund(unit_calls=[("2027-02-10", 0.2)], unit_nav=[("2027-02-10", 0.5)]).on(GRID)
    np.testing.assert_allclose(path.calls, [0, 0.2, 0])
    np.testing.assert_allclose(path.nav, [0, 0.5, 0.5])


def test_negative_running_nav_is_a_data_error_naming_fund_and_day():
    with pytest.raises(ValueError, match=r"Fund 'A': unit NAV would be -0.1 on 2027-01-15"):
        fund(unit_distributions=[("2027-01-15", 0.1)]).on(GRID)
    # a mark on the same day rescues it: marks are taken to include that day's flows
    rescued = fund(unit_distributions=[("2027-01-15", 0.1)], unit_nav=[("2027-01-15", 0.5)]).on(GRID)
    np.testing.assert_allclose(rescued.nav, [0.5, 0.5, 0.5])


def test_events_after_the_last_observation_are_ignored():
    path = fund(unit_calls=[("2027-02-01", 0.2), ("2027-04-15", 0.5)], unit_nav=[("2027-05-01", 9.0)]).on(GRID)
    np.testing.assert_allclose(path.calls, [0, 0.2, 0])
    np.testing.assert_allclose(path.nav, [0, 0.2, 0.2])


def test_closing_maps_to_next_observation_or_beyond_horizon():
    assert Fund("A", "BUYOUT", "2027-02-01").on(GRID).closing_index == 1
    assert Fund("A", "BUYOUT", "2027-02-28").on(GRID).closing_index == 1
    late = Fund("A", "BUYOUT", "2027-05-01").on(GRID)
    assert late.closing_index == 3 and late.beyond_horizon


def test_empty_history_gives_zero_arrays():
    path = fund().on(GRID)
    for array in (path.calls, path.distributions, path.nav):
        np.testing.assert_array_equal(array, [0, 0, 0])
