"""The five-sheet portfolio workbook (Liquid, FX, Flows, Commitments, Spec), run for one profile.

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


def sample_tables() -> dict[str, pd.DataFrame]:
    month_ends = pd.date_range("2009-04-30", "2012-12-31", freq="ME")
    k = np.arange(len(month_ends))
    liquid = pd.DataFrame({
        "USD Conservative": 0.004 + 0.010 * np.sin(k / 3.0),
        "USD Moderate": 0.006 + 0.016 * np.sin(k / 3.0),
        "USD Aggressive": 0.008 + 0.024 * np.sin(k / 3.0),
        "EUR Conservative": 0.003 + 0.009 * np.cos(k / 4.0),
        "EUR Moderate": 0.005 + 0.014 * np.cos(k / 4.0),
    }, index=month_ends).round(9)
    fx = pd.DataFrame({
        "EURUSD": (1.35 + 0.10 * np.sin(k / 5.0)).round(4),
        "GBPUSD": (1.58 + 0.08 * np.cos(k / 7.0)).round(4),
    }, index=month_ends)
    flows = pd.DataFrame([
        ("PEM2011", "2011-06-07", -25000.00, "Flow", 1_000_000),
        ("PEM2011", "2011-09-02", -8024.00, "Flow", 1_000_000),
        ("PEM2011", "2011-09-21", -37621.12, "Flow", 1_000_000),
        ("PEM2011", "2011-11-01", 38193.84, "Flow", 1_000_000),
        ("PEM2011", "2011-12-31", 45000.00, "NAV", 1_000_000),
        ("PEM2011", "2012-01-13", -9742.94, "Flow", 1_000_000),
        ("PEM2011", "2012-06-19", -31388.64, "Flow", 1_000_000),
        ("PEM2011", "2012-12-31", 95000.00, "NAV", 1_000_000),
        ("SEC_VI", "2012-03-15", -120000.00, "Flow", 1_000_000),
        ("SEC_VI", "2012-09-30", -50000.00, "Flow", 1_000_000),
        ("SEC_VI", "2012-12-31", 180000.00, "NAV", 1_000_000),
    ], columns=["Vintage", "Date", "Value", "Type", "Scale"])
    schedule = {
        ("USD", "Conservative"): {"SECONDARIES": 0.008, "BUYOUT": 0.022},
        ("USD", "Moderate"): {"SECONDARIES": 0.010, "BUYOUT": 0.028},
        ("EUR", "Conservative"): {"SECONDARIES": 0.007, "BUYOUT": 0.020},
        ("EUR", "Moderate"): {"SECONDARIES": 0.009, "BUYOUT": 0.026},
    }
    commitments = pd.DataFrame([
        (fund_type, year, currency, risk, 0.0 if year == 0 else rate * 100, 0.0 if year == 0 else rate)
        for (currency, risk), rates in schedule.items()
        for fund_type, rate in rates.items()
        for year in range(0, 11)
    ], columns=["Type", "Year", "Currency", "Risk", "Commitment", "Rate"])
    spec = pd.DataFrame({
        "Name": ["PEM2011", "SEC_VI", "PEM2012", "PEM2013", "SEC_VII"],
        "Year": ["31/12/2010", "31/12/2011", "31/12/2011", "31/12/2012", "31/12/2015"],
        "Type": ["BUYOUT", "SECONDARIES", "BUYOUT", "BUYOUT", "SECONDARIES"],
    })
    return {"Liquid": liquid, "FX": fx, "Flows": flows, "Commitments": commitments, "Spec": spec}


def write_sample_workbook(path) -> Path:
    path = Path(path)
    tables = sample_tables()
    with pd.ExcelWriter(path) as writer:
        tables["Liquid"].to_excel(writer, sheet_name="Liquid")          # date as the index: a blank first header, like the real sheet
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
    print("\nCommitment rates (calendar year × type) for this profile:")
    print(repository.commitment_rates().T)
    print("\nFunds loaded:")
    print(orchestrator.fund_summary())

    result = orchestrator.run()
    print(f"\nRun ({result.base_currency} base, {len(result.periods)} observations) — {result.status}")
    if result.shortfall is not None:
        print(result.shortfall)
    print(f"\nCommitments (sized in USD; carry-forward {'on' if CARRY_FORWARD else 'off'}; commitment = weight × (current_year_usd + carried_usd)):")
    print(result.commitments[["policy_year", "sizing_base_usd", "current_year_rate", "current_year_usd",
                              "carried_years", "carried_usd", "weight", "commitment_usd", "usd_rate", "commitment_base"]])
    print("\nLast observations:")
    print(result.periods[["liquid_open", "liquid_pnl", "distributions", "commitments", "calls",
                          "liquid_close", "private_close", "total_close"]].tail(6))
    print(f"\nFunds beyond the horizon (never committed): {result.funds_beyond_horizon}")
