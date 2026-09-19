"""Benchmarks read off a finished run: the liquid-only counterfactual and public-market-equivalent measures.

Every call is paid by selling the liquid portfolio and every distribution buys it back, so a
run already is a public-market-equivalent calculation against the investor's own portfolio.
With ``I`` the liquid index and ``T`` the last observation, the relationship is exact:

    with programme − liquid only  =  Σ (distributions(t) − calls(t)) × I(T)/I(t)  +  private NAV(T)
                                  =  FV(calls) × (KS-PME − 1)

Everything here is computed from a result's ``periods`` and ``funds`` tables, in base
currency; nothing in the simulation loop is involved. The index comes from
``periods.return_factor``, so no extra data is needed.
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


def _require_periods(periods: pd.DataFrame) -> None:
    if periods.empty:
        raise ValueError("periods is empty: there is no run to benchmark")


def growth_to_horizon(periods: pd.DataFrame) -> pd.Series:
    """``I(T)/I(t)`` per observation: what 1 put into the liquid portfolio at ``t`` is worth at the last observation."""
    _require_periods(periods)
    index = periods["return_factor"].cumprod()
    return index.iloc[-1] / index


def compare_with_liquid_only(periods: pd.DataFrame) -> pd.DataFrame:
    """The same liquid portfolio with no private programme, beside the run.

    ``liquid_only`` compounds the opening balance by the liquid returns alone;
    ``with_programme`` is the run's ``total_close``; ``value_added`` is their difference and
    ``value_added_share`` that difference as a fraction of ``liquid_only``. Private NAV is
    counted at its carrying value, so part of the value added is unrealised.
    """
    _require_periods(periods)
    table = pd.DataFrame({
        "liquid_only": periods["liquid_open"].iloc[0] * periods["return_factor"].cumprod(),
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
    dates, amounts = dates[amounts != 0], amounts[amounts != 0]
    if len(amounts) == 0 or (amounts > 0).all() or (amounts < 0).all() or dates.min() == dates.max():
        return float("nan")
    years = (dates - dates.min()).days.to_numpy() / DAYS_PER_YEAR

    def net_present_value(rate: float) -> float:
        return float(np.sum(amounts * (1.0 + rate) ** -years))

    low, high = LOWEST_RATE, 10.0
    sign_at_low = np.sign(net_present_value(low))
    while np.sign(net_present_value(high)) == sign_at_low and high < 1e6:
        high *= 10.0
    if np.sign(net_present_value(high)) == sign_at_low:
        return float("nan")
    for _ in range(200):
        middle = (low + high) / 2.0
        if np.sign(net_present_value(middle)) == sign_at_low:
            low = middle
        else:
            high = middle
        if high - low < 1e-12:
            break
    return (low + high) / 2.0


def _pme_row(calls: pd.Series, distributions: pd.Series, nav: float, growth: pd.Series) -> dict[str, float]:
    """One line of the PME table from dated base-currency calls and distributions and the NAV at the horizon."""
    factor = growth.reindex(calls.index)
    net = distributions - calls  # the investor's view: calls are money out
    terminal = pd.Series([float(nav)], index=[growth.index[-1]])
    nominal = pd.concat([net, terminal])
    compounded = pd.concat([net * factor, terminal])
    fv_calls = float((calls * factor).sum())
    fv_distributions = float((distributions * factor).sum())
    return {
        "calls": float(calls.sum()), "distributions": float(distributions.sum()), "nav": float(nav),
        "fv_calls": fv_calls, "fv_distributions": fv_distributions,
        "value_added": fv_distributions + float(nav) - fv_calls,
        "ks_pme": (fv_distributions + float(nav)) / fv_calls if fv_calls > 0 else float("nan"),
        "irr": annualised_irr(nominal.index, nominal.to_numpy()),
        "direct_alpha": annualised_irr(compounded.index, compounded.to_numpy()),
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
    rows = {("programme", "all"): _pme_row(periods["calls"], periods["distributions"],
                                           periods["private_close"].iloc[-1], growth)}
    if not funds.empty:
        flows = funds.reset_index()
        for level in ("fund_type", "fund"):
            for name, group in flows.groupby(level, sort=False):
                by_date = group.groupby("date")[["calls_base", "distributions_base", "nav_base"]].sum()
                nav = float(by_date["nav_base"].get(horizon, 0.0))
                rows[(level, name)] = _pme_row(by_date["calls_base"], by_date["distributions_base"], nav, growth)
    index = pd.MultiIndex.from_tuples(list(rows), names=["level", "name"])
    return pd.DataFrame(list(rows.values()), index=index, columns=PME_COLUMNS).astype(float)
