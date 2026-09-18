"""Run the EUR Moderate profile of the portfolio workbook from 100 US dollars.

Made to be run and debugged straight from PyCharm: open this file, set WORKBOOK below to
your workbook, put a breakpoint anywhere in run(), and press Debug. It also works from a
terminal, from the claude directory:

    python -m examples.eur_moderate                                    # uses the settings below
    python -m examples.eur_moderate /path/to/portfolio.xlsx [out_dir]  # overrides them

If WORKBOOK does not exist, a sample workbook in the same five-sheet layout is generated
next to it and used instead, so the script runs on a fresh checkout — the notice printed
at the top says which file was used.

The EUR portfolio's balance is kept in euros, so the 100 dollars are converted at the first
available EURUSD rate — the rate the engine also carries at the inception date, one month
before the first Liquid return; the script prints that conversion.
"""
import sys
from pathlib import Path

# Make `import pmsim` work when PyCharm runs this file directly (not as `python -m ...`).
ROOT = Path(__file__).resolve().parents[1]  # the claude/ directory
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from pmsim.data import Orchestrator, WorkbookRepository  # noqa: E402

# ---- settings: edit these, then Run / Debug this file -------------------------------------------
WORKBOOK = ROOT / "data" / "portfolio.xlsx"   # your workbook; a sample is generated here if it is missing
OUTPUT_DIR = None                             # e.g. ROOT / "data" / "out" to also write the result tables as CSV
CURRENCY, RISK = "EUR", "Moderate"            # the profile: a Liquid column "<CURRENCY> <RISK>" and Commitments rows
START_USD = 100.0                             # starting balance, in dollars ...
START_IN_BASE_CURRENCY = False                # ... or True to read START_USD as euros and skip the conversion
# --------------------------------------------------------------------------------------------------

PERIOD_COLUMNS = ["liquid_open", "liquid_pnl", "distributions", "commitments", "calls",
                  "liquid_close", "private_close", "total_close", "fx_translation"]


def starting_balance(repository: WorkbookRepository, start_usd: float) -> tuple[float, float, pd.Timestamp]:
    """The starting balance in the profile's currency, the rate used, and the date of that rate."""
    market = repository.market_data()
    first_liquid_date = market[repository.liquid_column].dropna().index[0]
    if repository.fx_column is None:  # a USD profile: nothing to convert
        return start_usd, 1.0, first_liquid_date
    rates = market[repository.fx_column].dropna()
    rate = rates.asof(first_liquid_date)  # the first known rate; the engine carries it at inception too
    if pd.isna(rate):
        raise ValueError(f"no {repository.fx_column} rate on or before {first_liquid_date.date()} to convert the starting balance")
    rate_date = rates.index[rates.index <= first_liquid_date][-1]
    # usd_per_base (EURUSD): euros = dollars / rate; base_per_usd (USDEUR): euros = dollars × rate
    base = start_usd / rate if repository.fx_quote == "usd_per_base" else start_usd * rate
    return float(base), float(rate), rate_date


def run(path, start_usd: float = START_USD, out_dir=None, *, start_in_base_currency: bool = START_IN_BASE_CURRENCY):
    """Load the workbook, build the engine's inputs step by step, run, and report.

    Each step is its own local variable so a breakpoint shows one thing at a time:
    repository (the sheets), initial_value (the conversion), spec (the settings), funds,
    portfolio, policy (the inputs the engine sees), then result. Step into
    orchestrator.run() to follow the period loop in Simulator.run().
    """
    repository = WorkbookRepository(path, CURRENCY, RISK)                       # 1. read every sheet, pick the profile

    if start_in_base_currency:                                                  # 2. the starting balance in euros
        initial_value, rate, rate_date = float(start_usd), float("nan"), None
    else:
        initial_value, rate, rate_date = starting_balance(repository, start_usd)

    spec = repository.simulation_spec(initial_value)                                       # 3. base currency, series, fx quote, returns→levels
    orchestrator = Orchestrator(repository, spec)

    funds = orchestrator.funds                                                  # 4. one Fund per Spec row, unit histories from Flows
    portfolio = orchestrator.portfolio                                          # 5. levels compounded from returns; USD rate inverted
    policy = orchestrator.policy                                                # 6. the rate each fund will get at its closing

    result = orchestrator.run()                                                 # 7. the period loop

    report(repository, orchestrator, result, start_usd, initial_value, rate, rate_date, start_in_base_currency)
    if out_dir is not None:
        write_csvs(orchestrator, result, Path(out_dir))
    return orchestrator, result


def report(repository, orchestrator, result, start_usd, initial_value, rate, rate_date, start_in_base_currency):
    inception = orchestrator.portfolio.first_date  # one period before the first Liquid return
    print(f"Workbook: {repository.path}")
    print(f"Sheets:   {repository.sheet_names}")
    print(f"Profile:  {repository.profile} · liquid column {repository.liquid_column!r} · "
          f"fx column {repository.fx_column!r} ({repository.fx_quote}) · inception year {repository.inception_year}")
    if start_in_base_currency:
        print(f"\nStarting balance: {CURRENCY} {initial_value:,.2f} at inception {inception}")
    else:
        print(f"\nStarting balance: USD {start_usd:,.2f} = {CURRENCY} {initial_value:,.4f} "
              f"at {repository.fx_column} {rate:.4f} (first available rate, {rate_date.date()}), held at inception {inception}")

    print(f"\nCommitment rates for {repository.profile} (calendar year × type):")
    print(repository.commitment_rates().T)
    print("\nFunds loaded:")
    print(orchestrator.fund_summary())
    print("\nWhere fund events pooled (first 12 rows; the full table is orchestrator.map_events_to_observations()):")
    print(orchestrator.map_events_to_observations().head(12))

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
    by_type = result.totals_by_fund_type()
    if not by_type.empty:
        print(f"\nBy fund type at {result.periods.index[-1].date()}:")
        print(by_type.xs(result.periods.index[-1], level="date")[["commitment_usd", "calls_base", "distributions_base", "nav_base"]])
    print(f"\nFunds beyond the horizon (never committed): {result.funds_beyond_horizon or 'none'}")


def write_csvs(orchestrator, result, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    result.periods.to_csv(out / "periods.csv")
    result.funds.to_csv(out / "funds.csv")
    result.commitments.to_csv(out / "commitments.csv")
    orchestrator.map_events_to_observations().to_csv(out / "map_events_to_observations.csv")
    orchestrator.fund_summary().to_csv(out / "fund_summary.csv")
    print(f"\nWrote periods.csv, funds.csv, commitments.csv, map_events_to_observations.csv, fund_summary.csv to {out}")


def resolve_workbook(path: Path) -> Path:
    """The workbook to run: the configured file, or a generated sample beside it when that file is missing."""
    if path.exists():
        return path
    from examples.profile_workbook import write_sample_workbook
    sample = path.with_name("sample_portfolio.xlsx")
    if not sample.exists():
        sample.parent.mkdir(parents=True, exist_ok=True)
        write_sample_workbook(sample)
    print(f"NOTE: {path} not found — running the generated sample {sample}.\n"
          f"      Set WORKBOOK at the top of {Path(__file__).name} to your workbook.\n")
    return sample


if __name__ == "__main__":
    pd.options.display.float_format = "{:,.4f}".format
    pd.options.display.width = 200
    pd.options.display.max_columns = 20
    workbook = resolve_workbook(Path(sys.argv[1]) if len(sys.argv) > 1 else WORKBOOK)
    output_dir = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR
    run(workbook, out_dir=output_dir)
