from datetime import date

import numpy as np
import pandas as pd
import pytest

from pmsim import AlignedFundHistory, Timeline

GRID = Timeline(pd.to_datetime(["2027-01-31", "2027-02-28", "2027-03-31"]))


def test_timeline_basics():
    assert GRID.n_observations == 3
    assert GRID.observation_date(0) == date(2027, 1, 31) and GRID.observation_date(-1) == date(2027, 3, 31)
    assert list(GRID.calendar_years) == [2027]
    assert GRID.dates.name == "date"


def test_timeline_accepts_strings_and_dates():
    same = Timeline(["2027-01-31", date(2027, 2, 28), pd.Timestamp("2027-03-31")])
    assert same.dates.equals(GRID.dates)


@pytest.mark.parametrize("dates, message", [
    ([], "at least one date"),
    (["2027-02-28", "2027-01-31"], "unique and increasing"),
    (["2027-01-31", "2027-01-31"], "unique and increasing"),
    (["2027-01-31 12:00"], "calendar date"),
])
def test_timeline_validation(dates, message):
    with pytest.raises(ValueError, match=message):
        Timeline(dates)


def test_index_of_is_first_observation_on_or_after():
    assert GRID.first_observation_on_or_after("2026-12-01") == 0
    assert GRID.first_observation_on_or_after("2027-01-31") == 0
    assert GRID.first_observation_on_or_after("2027-02-01") == 1
    assert GRID.first_observation_on_or_after("2027-02-28") == 1
    assert GRID.first_observation_on_or_after("2027-03-31") == 2
    assert GRID.first_observation_on_or_after("2027-04-01") == 3 == GRID.n_observations


def test_asof_carries_the_last_known_value_forward():
    sparse = pd.Series([0.9, 0.8], index=pd.to_datetime(["2026-12-15", "2027-03-31"]))
    np.testing.assert_array_equal(GRID.last_value_on_or_before(sparse), [0.9, 0.9, 0.8])
    daily = pd.Series(np.linspace(1, 2, 120), index=pd.date_range("2027-01-01", periods=120))
    np.testing.assert_allclose(GRID.last_value_on_or_before(daily), daily.loc[GRID.dates].to_numpy())
    with pytest.raises(ValueError, match="usd_rate has no value on or before the first observation 2027-01-31"):
        GRID.last_value_on_or_before(pd.Series([0.9], index=pd.to_datetime(["2027-02-01"])), name="usd_rate")


def test_fund_path_arrays_are_read_only_copies_of_one_length():
    calls = np.array([0.1, 0.0, 0.0])
    path = AlignedFundHistory(calls, [0, 0, 0.05], [0.1, 0.1, 0.05], closing_period=0)
    calls[0] = 9.0
    assert path.unit_calls[0] == 0.1 and path.n_observations == 3 and not path.closes_beyond_horizon
    with pytest.raises(ValueError):
        path.unit_calls[0] = 1.0
    assert AlignedFundHistory([0], [0], [0], closing_period=1).closes_beyond_horizon
    with pytest.raises(ValueError, match="one length"):
        AlignedFundHistory([0, 0], [0], [0], closing_period=0)
    with pytest.raises(ValueError, match="outside"):
        AlignedFundHistory([0], [0], [0], closing_period=2)
