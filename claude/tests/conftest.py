import numpy as np
import pandas as pd
import pytest

from pmsim import Fund, Portfolio, SimulationResult


def levels(*pairs):
    return [(day, float(value)) for day, value in pairs]


def check_identities(result: SimulationResult, atol: float = 1e-6) -> None:
    """Every completed period reconciles, and the fund table adds up to the period table."""
    p = result.periods
    close = lambda a, b: np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float), atol=atol, rtol=0)
    close(p["total_open"], p["liquid_open"] + p["private_open"])
    close(p["total_close"], p["liquid_close"] + p["private_close"])
    close(p["liquid_close"], p["liquid_open"] + p["liquid_pnl"] + p["distributions"] - p["calls"])
    close(p["private_valuation_pnl"], p["private_close"] - p["private_open"] - p["calls"] + p["distributions"])
    close(p["total_close"] - p["total_open"], p["liquid_pnl"] + p["private_valuation_pnl"])
    close(p["liquid_open"].to_numpy()[1:], p["liquid_close"].to_numpy()[:-1])
    close(p["private_open"].to_numpy()[1:], p["private_close"].to_numpy()[:-1])
    if result.funds.empty:
        assert (p[["calls", "distributions", "private_close", "commitments"]] == 0).all().all()
        return
    f = result.funds.groupby(level="date")[["calls_base", "distributions_base", "nav_base"]].sum()
    close(f["calls_base"], p["calls"].reindex(f.index))
    close(f["distributions_base"], p["distributions"].reindex(f.index))
    close(f["nav_base"], p["private_close"].reindex(f.index))
    close(result.nav_by_fund().sum(axis=1), p["private_close"])
    c = result.commitments.groupby(level="date")["commitment_base"].sum()
    close(c, p["commitments"].reindex(c.index))


@pytest.fixture
def identities():
    return check_identities


@pytest.fixture
def worked_funds():
    return [
        Fund("A", "BUYOUT", "2027-02-15", unit_calls=[("2027-03-01", 0.25)], unit_distributions=[("2027-06-01", 0.05)]),
        Fund("B", "BUYOUT", "2027-05-10", unit_calls=[("2027-05-20", 0.25)]),
    ]


@pytest.fixture
def worked_levels():
    return levels(("2027-01-01", 1_000_000), ("2027-03-31", 1_100_000), ("2027-06-30", 1_210_000))


@pytest.fixture
def usd_portfolio(worked_levels):
    return Portfolio("USD", worked_levels, {"BUYOUT": {2027: 0.10}})


@pytest.fixture
def gbp_portfolio(worked_levels):
    return Portfolio("GBP", worked_levels, {"BUYOUT": {2027: 0.10}},
                     usd_rate=[("2027-01-01", 0.80), ("2027-06-30", 0.75)])
