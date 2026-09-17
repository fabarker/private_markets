from datetime import date

import numpy as np
import pandas as pd
import pytest

from pmsim import FundPath, Timeline

GRID = Timeline(pd.to_datetime(["2027-01-31", "2027-02-28", "2027-03-31"]))


def test_timeline_basics():
    assert GRID.n == 3
    assert GRID.date_at(0) == date(2027, 1, 31) and GRID.date_at(-1) == date(2027, 3, 31)
    assert list(GRID.years) == [2027]
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
    assert GRID.index_of("2026-12-01") == 0
    assert GRID.index_of("2027-01-31") == 0
    assert GRID.index_of("2027-02-01") == 1
    assert GRID.index_of("2027-02-28") == 1
    assert GRID.index_of("2027-03-31") == 2
    assert GRID.index_of("2027-04-01") == 3 == GRID.n


def test_asof_carries_the_last_known_value_forward():
    sparse = pd.Series([0.9, 0.8], index=pd.to_datetime(["2026-12-15", "2027-03-31"]))
    np.testing.assert_array_equal(GRID.asof(sparse), [0.9, 0.9, 0.8])
    daily = pd.Series(np.linspace(1, 2, 120), index=pd.date_range("2027-01-01", periods=120))
    np.testing.assert_allclose(GRID.asof(daily), daily.loc[GRID.dates].to_numpy())
    with pytest.raises(ValueError, match="usd_rate has no value on or before the first observation 2027-01-31"):
        GRID.asof(pd.Series([0.9], index=pd.to_datetime(["2027-02-01"])), name="usd_rate")


def test_fund_path_arrays_are_read_only_copies_of_one_length():
    calls = np.array([0.1, 0.0, 0.0])
    path = FundPath(calls, [0, 0, 0.05], [0.1, 0.1, 0.05], closing_index=0)
    calls[0] = 9.0
    assert path.calls[0] == 0.1 and path.n == 3 and not path.beyond_horizon
    with pytest.raises(ValueError):
        path.calls[0] = 1.0
    assert FundPath([0], [0], [0], closing_index=1).beyond_horizon
    with pytest.raises(ValueError, match="one length"):
        FundPath([0, 0], [0], [0], closing_index=0)
    with pytest.raises(ValueError, match="outside"):
        FundPath([0], [0], [0], closing_index=2)
