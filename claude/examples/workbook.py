"""Load a workbook and run it. From the claude directory:

    python -m examples.workbook              # writes a sample workbook to a temp folder, then runs it
    python -m examples.workbook book.xlsx    # runs book.xlsx if it exists, else writes the sample there first

The sample is the design note's worked example laid out the way the loader expects:

    fund_spec           fund_name | type    | closing_date
    fund_market_data    fund_name | type (Flow/NAV) | value | date | scale     (unit = value / scale)
    market_data         date | liquid_gbp | gbp_per_usd                       (wide; long also accepted)
    commitment_rates    year | BUYOUT                                         (optional; can come from the spec)
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd

from pmsim.data import SimulationSpec, load_tables_workbook

SPEC = SimulationSpec(
    base_currency="GBP",
    liquid_series="liquid_gbp",
    fx_series="gbp_per_usd",        # quoted as GBP per 1 USD; use fx_quote="usd_per_base" for the other way round
    weights={"A": 0.6, "B": 0.4},
)


def sample_tables() -> dict[str, pd.DataFrame]:
    return {
        "fund_spec": pd.DataFrame({
            "fund_name": ["A", "B"],
            "type": ["BUYOUT", "BUYOUT"],
            "closing_date": pd.to_datetime(["2027-02-15", "2027-05-10"]),
        }),
        "fund_market_data": pd.DataFrame(
            [
                ("A", "Flow", -250_000.0, "2027-03-01", 1_000_000),   # a call: negative flow
                ("A", "NAV", 250_000.0, "2027-03-31", 1_000_000),     # a mark
                ("A", "Flow", 50_000.0, "2027-06-01", 1_000_000),     # a distribution: positive flow
                ("B", "Flow", -250_000.0, "2027-05-20", 1_000_000),
            ],
            columns=["fund_name", "type", "value", "date", "scale"],
        ).assign(date=lambda frame: pd.to_datetime(frame["date"])),
        "market_data": pd.DataFrame({
            "date": pd.to_datetime(["2027-01-01", "2027-03-31", "2027-06-30"]),
            "liquid_gbp": [1_000_000.0, 1_100_000.0, 1_210_000.0],
            "gbp_per_usd": [0.80, 0.80, 0.75],
        }),
        "commitment_rates": pd.DataFrame({"year": [2027], "BUYOUT": [0.10]}),
    }


def write_sample_workbook(path) -> Path:
    path = Path(path)
    with pd.ExcelWriter(path) as writer:
        for sheet, frame in sample_tables().items():
            frame.to_excel(writer, sheet_name=sheet, index=False)
    return path


if __name__ == "__main__":
    pd.options.display.float_format = "{:,.2f}".format
    pd.options.display.width = 200

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp()) / "sample_workbook.xlsx"
    if not path.exists():
        write_sample_workbook(path)
        print(f"Wrote sample workbook to {path}")

    orchestrator = load_tables_workbook(path, SPEC)
    print(f"\nSheets: {orchestrator.repository.sheet_names}")
    print("\nFunds loaded:")
    print(orchestrator.fund_summary())
    print("\nWhere each fund event pooled:")
    print(orchestrator.map_events_to_observations())

    result = orchestrator.run()
    print(f"\nRun ({result.base_currency} base) — {result.status}")
    print(result.periods[["liquid_open", "liquid_pnl", "distributions", "sizing_base", "commitments",
                          "calls", "liquid_close", "private_close", "total_close", "fx_translation"]].T)
    print("\nCommitments:")
    print(result.commitments[["closing_date", "rate", "sizing_base", "commitment_base", "usd_rate", "commitment_usd"]])
