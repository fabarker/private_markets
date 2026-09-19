"""The portfolio workbook (Liquid, Liquid Spec, FX, Flows, Commitments, Spec), run for one profile.

    python -m examples.profile_workbook                                   # sample workbook → temp folder; USD Conservative from 1,000,000
    python -m examples.profile_workbook book.xlsx USD Conservative 1e6    # a real workbook: path, currency, risk, starting balance
    python -m examples.profile_workbook book.xlsx EUR Conservative 5e6

If the path does not exist, a sample in the same layout as the real workbook is written
there first. The sample's shape is the point: compare a real file against it sheet by sheet.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pmsim.data import load_profile_workbook

# The Commitments sheet is an annual budget. A year in which no fund of a type closes is still sized — that year's
# rate on that year's balance — and its dollars wait for the next fund of the type. False: such a year is not used.
CARRY_FORWARD = True


# The fund universe, exactly as the Spec sheet lists it: name, closing date (dd/mm/yyyy), type.
FUNDS = [
    ("PEM2011", "31/12/2010", "BUYOUT"),
    ("SEC_VI", "31/12/2011", "SECONDARIES"),
    ("PEM2012", "31/12/2011", "BUYOUT"),
    ("PEM2013", "31/12/2012", "BUYOUT"),
    ("PEM2014", "31/12/2013", "BUYOUT"),
    ("PEM2015", "31/12/2014", "BUYOUT"),
    ("SEC_VII", "31/12/2015", "SECONDARIES"),
    ("PEM2016", "31/12/2015", "BUYOUT"),
    ("PEM2017", "30/12/2016", "BUYOUT"),
    ("PEM2018", "29/12/2017", "BUYOUT"),
    ("SEC_VIII", "31/12/2018", "SECONDARIES"),
    ("PEM2019", "31/12/2018", "BUYOUT"),
    ("PEM2020", "31/12/2019", "BUYOUT"),
    ("PEM2021", "31/12/2020", "BUYOUT"),
    ("PEM2022", "31/12/2021", "BUYOUT"),
    ("SEC_IX", "31/12/2021", "SECONDARIES"),
    ("PEM2023", "31/12/2022", "BUYOUT"),
    ("PEM2024", "31/12/2023", "BUYOUT"),
    ("SEC_X", "31/12/2025", "SECONDARIES"),
]

# Each portfolio's ExRet: the yearly return its pacing schedule was built on.
LIQUID_SPEC = {
    "USD Conservative": 0.054, "USD Moderate": 0.064, "USD Aggressive": 0.074,
    "EUR Conservative": 0.047, "EUR Moderate": 0.059, "EUR Aggressive": 0.070,
    "GBP Conservative": 0.054, "GBP Moderate": 0.064, "GBP Aggressive": 0.075,
}
VOLATILITY = {"Conservative": 0.05, "Moderate": 0.09, "Aggressive": 0.13}   # yearly, for the sample's return series
FIRST_YEAR_RATES = {   # the pacing schedule's year-1 commitment, per 1 of liquid value on the first commitment date
    "Conservative": {"BUYOUT": 0.022, "SECONDARIES": 0.008},
    "Moderate": {"BUYOUT": 0.030, "SECONDARIES": 0.011},
    "Aggressive": {"BUYOUT": 0.038, "SECONDARIES": 0.014},
}

# How $1 committed behaves: what is called in each year of the fund's life, how the NAV grows
# (the last rate repeats), and the share of it distributed once the harvest starts.
FUND_SHAPES = {
    "BUYOUT": {"life": 13, "calls": (0.25, 0.28, 0.22, 0.13, 0.07),
               "growth": (-0.06, 0.01, 0.07, 0.12, 0.13), "payout": 0.30, "harvest_from": 4},
    "SECONDARIES": {"life": 9, "calls": (0.45, 0.35, 0.15),
                    "growth": (0.02, 0.08, 0.11), "payout": 0.35, "harvest_from": 2},
}
SCALE = 1_000_000.0   # the Flows sheet is in dollars; unit = Value / Scale


def fund_events(name: str, closing: str, fund_type: str) -> list[tuple]:
    """A plausible J-curve for one fund: calls, a NAV mark, then distributions, year by year.

    Within each year the order is calls, then the mark, then that year's distributions, so
    the NAV the engine rolls forward between marks never goes negative. The fund winds up in
    its final year: everything left is distributed and the NAV reaches zero.
    """
    shape = FUND_SHAPES[fund_type]
    start = pd.to_datetime(closing, dayfirst=True)
    rows, nav = [], 0.0

    for year in range(1, shape["life"] + 1):
        on = lambda months, days: start + pd.DateOffset(years=year - 1, months=months, days=days)

        called = shape["calls"][year - 1] if year <= len(shape["calls"]) else 0.0
        for months, share in ((2, 0.6), (8, 0.4)):          # drawn down in two goes, mid-quarter
            if called:
                rows.append((name, on(months, 14), -called * share * SCALE, "Flow", SCALE))

        nav = (nav + called) * (1 + shape["growth"][min(year, len(shape["growth"])) - 1])
        rows.append((name, on(10, 29), nav * SCALE, "NAV", SCALE))      # the mark, before the year's distribution

        winding_up = year == shape["life"]
        distributed = nav if winding_up else (nav * shape["payout"] if year >= shape["harvest_from"] else 0.0)
        if distributed:
            rows.append((name, on(11, 14), distributed * SCALE, "Flow", SCALE))
        nav -= distributed

    return rows


def sample_tables() -> dict[str, pd.DataFrame]:
    """The six sheets, spanning the whole fund universe: Apr 2009 to Dec 2026, monthly."""
    month_ends = pd.date_range("2009-04-30", "2026-12-31", freq="ME")
    rng = np.random.default_rng(20260919)   # a fixed seed, so the sample is the same every time

    liquid = pd.DataFrame({
        profile: (expected / 12 + rng.normal(0.0, VOLATILITY[profile.split()[1]] / np.sqrt(12), len(month_ends)))
        for profile, expected in LIQUID_SPEC.items()
    }, index=month_ends).round(9)

    fx = pd.DataFrame({
        "EURUSD": (1.32 * np.exp(np.cumsum(rng.normal(0.0, 0.025, len(month_ends))))).round(4),
        "GBPUSD": (1.48 * np.exp(np.cumsum(rng.normal(0.0, 0.022, len(month_ends))))).round(4),
    }, index=month_ends)

    flows = pd.DataFrame(
        [row for name, closing, fund_type in FUNDS for row in fund_events(name, closing, fund_type)],
        columns=["Vintage", "Date", "Value", "Type", "Scale"],
    ).round({"Value": 2})

    # A pacing schedule: year 1's rate, grown at the profile's own expected return, so the share
    # of the liquid value it stands for is the same every year.
    commitments = pd.DataFrame([
        (fund_type, year, currency, risk,
         0.0 if year == 0 else round(rate * (1 + LIQUID_SPEC[f"{currency} {risk}"]) ** (year - 1), 6) * 100,
         0.0 if year == 0 else round(rate * (1 + LIQUID_SPEC[f"{currency} {risk}"]) ** (year - 1), 6))
        for currency in ("USD", "EUR", "GBP")
        for risk in ("Conservative", "Moderate", "Aggressive")
        for fund_type, rate in FIRST_YEAR_RATES[risk].items()
        for year in range(0, 21)
    ], columns=["Type", "Year", "Currency", "Risk", "Commitment", "Rate"])

    spec = pd.DataFrame(FUNDS, columns=["Name", "Year", "Type"])
    liquid_spec = pd.DataFrame({"Liquid": list(LIQUID_SPEC), "ExRet": list(LIQUID_SPEC.values())})

    return {"Liquid": liquid, "Liquid Spec": liquid_spec, "FX": fx, "Flows": flows,
            "Commitments": commitments, "Spec": spec}


def write_sample_workbook(path) -> Path:
    path = Path(path)
    tables = sample_tables()
    with pd.ExcelWriter(path) as writer:
        tables["Liquid"].to_excel(writer, sheet_name="Liquid")          # date as the index: a blank first header, like the real sheet
        tables["Liquid Spec"].to_excel(writer, sheet_name="Liquid Spec", index=False)
        tables["FX"].to_excel(writer, sheet_name="FX")
        for sheet in ("Flows", "Commitments", "Spec"):
            tables[sheet].to_excel(writer, sheet_name=sheet, index=False)
    return path


if __name__ == "__main__":
    pd.options.display.float_format = "{:,.2f}".format
    pd.options.display.width = 200

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp()) / "portfolio_workbook.xlsx"
    currency = sys.argv[2] if len(sys.argv) > 2 else "USD"
    risk = sys.argv[3] if len(sys.argv) > 3 else "Conservative"
    initial_value = float(sys.argv[4]) if len(sys.argv) > 4 else 1_000_000.0
    if not path.exists():
        write_sample_workbook(path)
        print(f"Wrote sample workbook to {path}")

    orchestrator = load_profile_workbook(path, currency, risk, initial_value, carry_forward=CARRY_FORWARD)
    repository = orchestrator.repository
    print(f"\n{repository}")
    print(f"liquid column: {repository.liquid_column!r} · fx column: {repository.fx_column!r} ({repository.fx_quote})"
          f" · inception year: {repository.inception_year}")
    print(f"expected return X: {repository.expected_return:.2%} · the pacing model's liquid value is 1 on the first commitment date, "
          f"{orchestrator.simulator.first_commitment_date}")
    print("\nPacing schedule (calendar year × type) for this profile, per 1 of liquid value on that date:")
    print(repository.commitment_rates().T)
    print("\nFunds loaded:")
    print(orchestrator.fund_summary())

    result = orchestrator.run()
    print(f"\nRun ({result.base_currency} base, {len(result.periods)} observations) — {result.status}")
    if result.shortfall is not None:
        print(result.shortfall)
    print(f"\nCommitments (sized in USD; carry-forward {'on' if CARRY_FORWARD else 'off'}; commitment = weight × (current_year_usd + carried_usd)):")
    print(result.commitments[["policy_year", "sizing_base_usd", "current_year_rate", "expected_value", "current_year_usd",
                              "carried_years", "carried_usd", "weight", "commitment_usd", "usd_rate", "commitment_base"]])
    print("\nThe five running values, last observations:")
    print(result.tracked_values().tail(6))
    print(f"\nFunds beyond the horizon (never committed): {result.funds_beyond_horizon}")
