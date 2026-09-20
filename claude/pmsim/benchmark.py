"""Benchmarks read off a finished run: the liquid-only counterfactual and public-market-equivalent measures.

Every call is paid by selling the liquid portfolio and every distribution buys it back, so a
run already is a public-market-equivalent calculation against the investor's own portfolio.
With ``I`` the liquid index and ``T`` the last observation, the relationship is exact:

    with programme − liquid only  =  Σ (distributions(t) − calls(t)) × I(T)/I(t)  +  private NAV(T)
                                  =  FV(calls) × (KS-PME − 1)

Everything here is computed from a result's ``periods`` and ``funds`` tables, in base
currency; nothing in the simulation loop is involved. The index is compounded from
``periods.period_return``, so no extra data is needed.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

COMPARISON_COLUMNS = ["liquid_only", "with_programme", "value_added", "value_added_share"]
PME_COLUMNS = ["calls", "distributions", "nav", "fv_calls", "fv_distributions",
               "value_added", "ks_pme", "irr", "direct_alpha"]
DAYS_PER_YEAR = 365.0
LOWEST_RATE = -0.9999  # a return below −99.99% a year is reported as undefined
HIGHEST_RATE_TRIED = 1e6  # the search for a rate stops widening here
BISECTION_STEPS = 200
BISECTION_WIDTH = 1e-12  # the search stops when the bracket is this narrow


def _require_periods(periods: pd.DataFrame) -> None:
    if periods.empty:
        raise ValueError("periods is empty: there is no run to benchmark")


def growth_to_horizon(periods: pd.DataFrame) -> pd.Series:
    """``I(T)/I(t)`` per observation.

    What 1 put into the liquid portfolio at ``t`` is worth at the last observation.
    """
    _require_periods(periods)

    liquid_index = (1.0 + periods["period_return"]).cumprod()
    index_at_horizon = liquid_index.iloc[-1]

    return index_at_horizon / liquid_index


def compare_with_liquid_only(periods: pd.DataFrame) -> pd.DataFrame:
    """The same liquid portfolio with no private programme, beside the run.

    ``liquid_only`` compounds the opening balance by the liquid returns alone;
    ``with_programme`` is the run's ``total_close``; ``value_added`` is their difference and
    ``value_added_share`` that difference as a fraction of ``liquid_only``. Private NAV is
    counted at its carrying value, so part of the value added is unrealised.
    """
    _require_periods(periods)

    opening_balance = periods["liquid_open"].iloc[0]
    compounded_returns = (1.0 + periods["period_return"]).cumprod()

    table = pd.DataFrame({
        "liquid_only": opening_balance * compounded_returns,
        "with_programme": periods["total_close"],
    })
    table["value_added"] = table["with_programme"] - table["liquid_only"]
    table["value_added_share"] = table["value_added"] / table["liquid_only"]

    return table[COMPARISON_COLUMNS]


def annualised_irr(dates: Any, amounts: Any) -> float:
    """Annualised internal rate of return of dated amounts (ACT/365); NaN when it is undefined.

    Amounts follow the investor's view: money paid out is negative, money received positive.
    The rate is undefined when nothing was paid out or nothing received, when every amount
    falls on one date, or when no rate above −99.99% a year sets the net present value to
    zero. Found by bisection; flows that change sign more than once can have several roots,
    and this returns the one bracketed first.
    """
    dates = pd.DatetimeIndex(dates)
    amounts = np.asarray(amounts, dtype=float)

    if len(dates) != len(amounts):
        raise ValueError("dates and amounts must have the same length")

    # Zero amounts carry no information: drop them.
    is_not_zero = amounts != 0
    dates = dates[is_not_zero]
    amounts = amounts[is_not_zero]

    # A rate needs money going both ways, on more than one date.
    nothing_left = len(amounts) == 0
    all_received = (amounts > 0).all()
    all_paid_out = (amounts < 0).all()
    if nothing_left or all_received or all_paid_out or dates.min() == dates.max():
        return float("nan")

    years_from_first_flow = (dates - dates.min()).days.to_numpy() / DAYS_PER_YEAR

    def net_present_value(rate: float) -> float:
        discount_factors = (1.0 + rate) ** -years_from_first_flow
        return float(np.sum(amounts * discount_factors))

    # Bracket a root: widen the upper rate until the net present value changes sign.
    low = LOWEST_RATE
    high = 10.0
    sign_at_low = np.sign(net_present_value(low))

    while np.sign(net_present_value(high)) == sign_at_low and high < HIGHEST_RATE_TRIED:
        high *= 10.0

    if np.sign(net_present_value(high)) == sign_at_low:
        return float("nan")

    # Bisect: keep the half of the bracket whose ends still disagree in sign.
    for _ in range(BISECTION_STEPS):
        middle = (low + high) / 2.0

        if np.sign(net_present_value(middle)) == sign_at_low:
            low = middle
        else:
            high = middle

        if high - low < BISECTION_WIDTH:
            break

    return (low + high) / 2.0


def _pme_row(calls: pd.Series, distributions: pd.Series, nav: float, growth: pd.Series) -> dict[str, float]:
    """One line of the PME table.

    ``calls`` and ``distributions`` are dated base-currency amounts, ``nav`` the NAV at the
    horizon and ``growth`` the liquid index's growth from each date to the horizon.
    """
    nav = float(nav)
    growth_from_each_flow = growth.reindex(calls.index)
    horizon = growth.index[-1]

    # The investor's view: calls are money out, distributions money in, and the NAV is
    # received at the horizon.
    net_flows = distributions - calls
    nav_at_horizon = pd.Series([nav], index=[horizon])

    nominal_flows = pd.concat([net_flows, nav_at_horizon])
    index_compounded_flows = pd.concat([net_flows * growth_from_each_flow, nav_at_horizon])

    # Each flow carried to the horizon at the liquid index's return.
    fv_calls = float((calls * growth_from_each_flow).sum())
    fv_distributions = float((distributions * growth_from_each_flow).sum())

    if fv_calls > 0:
        ks_pme = (fv_distributions + nav) / fv_calls
    else:
        ks_pme = float("nan")

    return {
        "calls": float(calls.sum()),
        "distributions": float(distributions.sum()),
        "nav": nav,
        "fv_calls": fv_calls,
        "fv_distributions": fv_distributions,
        "value_added": fv_distributions + nav - fv_calls,
        "ks_pme": ks_pme,
        "irr": annualised_irr(nominal_flows.index, nominal_flows.to_numpy()),
        "direct_alpha": annualised_irr(index_compounded_flows.index, index_compounded_flows.to_numpy()),
    }


def public_market_equivalent(periods: pd.DataFrame, funds: pd.DataFrame) -> pd.DataFrame:
    """The private programme measured against the liquid portfolio that funded it.

    One row for the whole programme, then one per fund type and one per fund (index:
    ``level``, ``name``), all in base currency with the liquid index as the benchmark:

    ``calls``, ``distributions``   nominal sums
    ``nav``                        private NAV at the last observation
    ``fv_calls``, ``fv_distributions``   each flow compounded to the last observation by the index
    ``value_added``                ``fv_distributions + nav − fv_calls``; the programme's equals
                                   ``compare_with_liquid_only`` at the last observation, and the
                                   funds' (and the fund types') add up to it
    ``ks_pme``                     Kaplan–Schoar PME, ``(fv_distributions + nav) / fv_calls``; above 1
                                   means the programme beat the liquid portfolio
    ``irr``                        annualised IRR of the flows and the closing NAV
    ``direct_alpha``               annualised IRR of the index-compounded flows and the closing NAV:
                                   the yearly rate by which the programme out- or under-performed
                                   (``log1p`` of it is the continuously compounded form)

    Ratios are NaN where they are undefined: nothing called yet, or every flow on one date.
    """
    _require_periods(periods)

    growth = growth_to_horizon(periods)
    horizon = periods.index[-1]

    # The whole programme, from the period totals.
    programme_nav = periods["private_close"].iloc[-1]
    rows = {
        ("programme", "all"): _pme_row(periods["calls"], periods["distributions"], programme_nav, growth),
    }

    # Then each fund type, and each fund, from the fund-level table.
    if not funds.empty:
        fund_flows = funds.reset_index()

        for level in ("fund_type", "fund"):
            for name, group in fund_flows.groupby(level, sort=False):
                by_date = group.groupby("date")[["calls_base", "distributions_base", "nav_base"]].sum()
                nav_at_horizon = float(by_date["nav_base"].get(horizon, 0.0))

                rows[(level, name)] = _pme_row(
                    by_date["calls_base"],
                    by_date["distributions_base"],
                    nav_at_horizon,
                    growth,
                )

    index = pd.MultiIndex.from_tuples(list(rows), names=["level", "name"])
    table = pd.DataFrame(list(rows.values()), index=index, columns=PME_COLUMNS)
    return table.astype(float)
