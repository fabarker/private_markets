"""Load every sheet of the portfolio workbook and run the EUR Moderate profile from 100 US dollars.

    python -m examples.eur_moderate /path/to/portfolio.xlsx            # print the run
    python -m examples.eur_moderate /path/to/portfolio.xlsx out/       # ... and write the result tables as CSV

The EUR portfolio's balance is kept in euros, so the 100 dollars are converted at the first
available EURUSD rate on or before the first Liquid date; the script prints that conversion.
Set START_IN_BASE_CURRENCY = True to start from 100 euros instead.
"""
import sys
from pathlib import Path

import pandas as pd

from pmsim.data import Orchestrator, WorkbookRepository

CURRENCY, RISK = "EUR", "Moderate"
START_USD = 100.0
START_IN_BASE_CURRENCY = False   # True: START_USD is read as euros, no conversion

PERIOD_COLUMNS = ["liquid_open", "liquid_pnl", "distributions", "commitments", "calls",
                  "liquid_close", "private_close", "total_close", "fx_translation"]


def starting_balance(repository: WorkbookRepository, start_usd: float) -> tuple[float, float, pd.Timestamp]:
    """The starting balance in the profile's currency, the rate used, and the date it applies to."""
    market = repository.market_data()
    first_date = market[repository.liquid_column].dropna().index[0]
    if repository.fx_column is None:  # a USD profile: nothing to convert
        return start_usd, 1.0, first_date
    rate = market[repository.fx_column].dropna().asof(first_date)
    if pd.isna(rate):
        raise ValueError(f"no {repository.fx_column} rate on or before {first_date.date()} to convert the starting balance")
    # usd_per_base (EURUSD): euros = dollars / rate; base_per_usd (USDEUR): euros = dollars × rate
    base = start_usd / rate if repository.fx_quote == "usd_per_base" else start_usd * rate
    return float(base), float(rate), first_date


def run(path, start_usd: float = START_USD, out_dir=None, *, start_in_base_currency: bool = START_IN_BASE_CURRENCY):
    repository = WorkbookRepository(path, CURRENCY, RISK)
    if start_in_base_currency:
        initial_value, rate, first_date = start_usd, float("nan"), repository.market_data().index[0]
    else:
        initial_value, rate, first_date = starting_balance(repository, start_usd)
    orchestrator = Orchestrator(repository, repository.spec(initial_value))
    result = orchestrator.run()

    print(f"Workbook: {repository.path}")
    print(f"Sheets:   {repository.sheet_names}")
    print(f"Profile:  {repository.profile} · liquid column {repository.liquid_column!r} · "
          f"fx column {repository.fx_column!r} ({repository.fx_quote}) · inception year {repository.inception_year}")
    if start_in_base_currency:
        print(f"\nStarting balance: {CURRENCY} {initial_value:,.2f} on {first_date.date()}")
    else:
        print(f"\nStarting balance: USD {start_usd:,.2f} = {CURRENCY} {initial_value:,.4f} "
              f"at {repository.fx_column} {rate:.4f} on {first_date.date()}")

    print(f"\nCommitment rates for {repository.profile} (calendar year × type):")
    print(repository.commitment_rates().T)
    print("\nFunds loaded:")
    print(orchestrator.fund_summary())
    print("\nWhere fund events pooled (first 12 rows; the full table is orchestrator.event_map()):")
    print(orchestrator.event_map().head(12))

    print(f"\nRun: {result.base_currency} base · {len(result.periods)} observations "
          f"{result.periods.index[0].date()} → {result.periods.index[-1].date()} · {result.status}")
    if result.shortfall is not None:
        print(f"  {result.shortfall}")
    print("\nCommitments:")
    print(result.commitments[["closing_date", "policy_year", "rate", "sizing_base", "commitment_base", "usd_rate", "commitment_usd"]])
    print(f"\nFirst observations ({result.base_currency}):")
    print(result.periods[PERIOD_COLUMNS].head(3))
    print(f"\nLast observations ({result.base_currency}):")
    print(result.periods[PERIOD_COLUMNS].tail(6))
    by_type = result.by_type()
    if not by_type.empty:
        print(f"\nBy fund type at {result.periods.index[-1].date()}:")
        print(by_type.xs(result.periods.index[-1], level="date")[["commitment_usd", "calls_base", "distributions_base", "nav_base"]])
    print(f"\nFunds beyond the horizon (never committed): {result.beyond_horizon or 'none'}")

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        result.periods.to_csv(out / "periods.csv")
        result.funds.to_csv(out / "funds.csv")
        result.commitments.to_csv(out / "commitments.csv")
        orchestrator.event_map().to_csv(out / "event_map.csv")
        orchestrator.fund_summary().to_csv(out / "fund_summary.csv")
        print(f"\nWrote periods.csv, funds.csv, commitments.csv, event_map.csv, fund_summary.csv to {out}")
    return orchestrator, result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m examples.eur_moderate /path/to/portfolio.xlsx [output_dir]")
    pd.options.display.float_format = "{:,.4f}".format
    pd.options.display.width = 200
    pd.options.display.max_columns = 20
    run(sys.argv[1], out_dir=sys.argv[2] if len(sys.argv) > 2 else None)
