from datetime import date, datetime, timezone

import pandas as pd
import pytest

from vintage import EntryType, FundVintage, HistoryEntry


def test_future_vintages_and_midnight_datetime_are_supported():
    f = FundVintage("Future", "BUYOUT", commitment_date=datetime(2052, 1, 1))
    assert f.vintage_year == 2052 and type(f.commitment_date) is date


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), "bad"])
def test_invalid_history_is_not_silently_discarded(value):
    with pytest.raises(ValueError, match="Invalid FLOW entry"):
        FundVintage("A", "BUYOUT", vintage_year=2027,
                    normalized_realized_net_cash_flow=[("2027-01-01", value)])


@pytest.mark.parametrize("method", ["add_flow", "add_call", "add_distribution", "add_nav"])
def test_mutation_helpers_reject_nonfinite_values_atomically(method):
    f = FundVintage("A", "BUYOUT", vintage_year=2027)
    with pytest.raises(ValueError):
        getattr(f, method)("2027-01-01", float("nan"))
    assert not f.normalized_realized_nav and not f.normalized_realized_net_cash_flow


def test_navs_nonnegative_and_unique_and_zero_is_preserved():
    f = FundVintage("A", "BUYOUT", vintage_year=2027)
    f.add_nav("2027-01-01", 0)
    with pytest.raises(ValueError, match="Duplicate"):
        f.add_nav("2027-01-01", 1)
    with pytest.raises(ValueError, match="non-negative"):
        f.add_nav("2027-02-01", -.1)
    assert len(f.normalized_realized_nav) == 1
    assert f.normalized_realized_nav[0].value == 0


def test_direct_history_entries_are_normalized_and_validated():
    e = HistoryEntry("2027-01-01", "1.2", "NAV")
    assert e.date == date(2027, 1, 1) and e.type is EntryType.NAV and e.value == 1.2
    with pytest.raises(ValueError):
        HistoryEntry("2027-01-01", float("inf"), EntryType.FLOW)


@pytest.mark.parametrize("d", [datetime(2027, 1, 1, 12), datetime(2027, 1, 1, tzinfo=timezone.utc)])
def test_ambiguous_times_rejected(d):
    with pytest.raises(ValueError, match="calendar dates"):
        FundVintage("A", "BUYOUT", commitment_date=d)


def test_asof_helpers_exclude_future_history_and_roll_stale_nav():
    f = FundVintage("A", "BUYOUT", commitment_size=100, commitment_date="2027-01-01",
                    normalized_realized_net_cash_flow=[("2027-01-02", -.2), ("2027-02-05", .03),
                                                       ("2028-01-01", -.9)],
                    normalized_realized_nav=[("2027-01-31", .23), ("2028-03-01", 1.1)])
    assert f.nav_on("2027-02-28") == pytest.approx(20)
    assert f.total_called_as_of("2027-02-28") == 20
    assert f.total_distributed_as_of("2027-02-28") == 3
    assert f.nav_series([]).empty
    assert f.nav_on("2026-01-01") == 0


def test_gross_flows_and_serialization_roundtrip():
    f = FundVintage("A", "BUYOUT", commitment_size=100, commitment_date="2050-01-01")
    f.add_call("2050-03-01", .5)
    f.add_distribution("2050-03-01", .2)
    f.add_nav("2050-03-31", .4)
    assert f.total_called == 50 and f.total_distributed == 20
    assert FundVintage.from_dict(f.to_dict()).to_dict() == f.to_dict()


def test_irr_asof_adjusts_for_post_mark_distributions():
    f = FundVintage("A", "BUYOUT", commitment_size=100, commitment_date="2025-01-01",
                    normalized_realized_net_cash_flow=[("2025-01-01", -1), ("2026-01-01", .2)],
                    normalized_realized_nav=[("2025-06-01", 1.1)])
    # Terminal value is .9, not stale 1.1; .2 distribution + .9 terminal = 1.1.
    assert f.irr(as_of="2026-01-01") == pytest.approx(.1, abs=1e-6)


def test_legacy_monthly_helpers_remain_available():
    f = FundVintage("A", "BUYOUT", commitment_size=100, vintage_year=2027,
                    normalized_realized_nav=[("2027-01-01", .3)])
    dates = pd.to_datetime(["2027-01-01", "2027-02-01"])
    assert f.monthly_flows(dates).tolist() == [0, 0]
    assert f.monthly_nav(dates).tolist() == [30, 30]
