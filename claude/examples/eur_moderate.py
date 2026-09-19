"""Run the EUR Moderate profile of the portfolio workbook from 100 US dollars.

Made to be run and debugged straight from PyCharm: open this file, set WORKBOOK below to
your workbook, put a breakpoint anywhere in run(), and press Debug. It also works from a
terminal, from the claude directory:

    python -m examples.eur_moderate                                    # uses the settings below
    python -m examples.eur_moderate /path/to/portfolio.xlsx [out_dir]  # overrides them

If WORKBOOK does not exist, a sample workbook in the same layout is generated
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
CARRY_FORWARD = True                          # a year with no fund of a type is still sized (its rate × that year-end's
                                              # balance) and the dollars wait for the next fund of the type; False: not used
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


def run(path, start_usd: float = START_USD, out_dir=None, *, start_in_base_currency: bool = START_IN_BASE_CURRENCY,
        carry_forward: bool = CARRY_FORWARD, draws=None):
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

    # draws=None leaves the Spec sheet's Draws column in charge; pass {} to ignore it and
    # let carry_forward decide the years instead.
    settings = {"carry_forward": carry_forward} | ({} if draws is None else {"draws": draws})
    spec = repository.simulation_spec(initial_value, **settings)                # 3. currency, series, fx quote, expected return
    orchestrator = Orchestrator(repository, spec)

    funds = orchestrator.funds                                                  # 4. one Fund per Spec row, unit histories from Flows
    portfolio = orchestrator.portfolio                                          # 5. levels compounded from returns; USD rate inverted
    policy = orchestrator.policy                                                # 6. the schedule, X, and per fund: weight and the years it draws

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

    print(f"\nExpected return X for {repository.profile}: {orchestrator.expected_return:.2%} a year (from the workbook's expected-returns sheet). "
          f"The pacing model's liquid value is 1 on the first commitment date, {orchestrator.simulator.first_commitment_date}.")
    print(f"Pacing schedule for {repository.profile} (calendar year × type), per 1 of liquid value on that date:")
    print(repository.commitment_rates().T)
    print("\nFunds loaded:")
    print(orchestrator.fund_summary())
    print("\nWhere fund events pooled (first 12 rows; the full table is orchestrator.map_events_to_observations()):")
    print(orchestrator.map_events_to_observations().head(12))

    print(f"\nRun: {result.base_currency} base · {len(result.periods)} observations "
          f"{result.periods.index[0].date()} → {result.periods.index[-1].date()} · {result.status}")
    if result.shortfall is not None:
        print(f"  {result.shortfall}")
    carry = "on" if orchestrator.spec.carry_forward else "off"
    print(f"\nCommitments, sized in USD on the liquid-only value (sizing_base_usd); carry-forward {carry}.")
    print("  own_year_usd = own_year_rate / expected_value × sizing_base_usd · commitment_usd = weight × (own_year_usd + other_years_usd)")
    print(result.commitments[["policy_year", "sizing_base_usd", "own_year_rate", "expected_value", "own_year_usd",
                              "drawn_years", "other_years_usd", "weight", "commitment_usd", "usd_rate", "commitment_base"]])

    # Which schedule years each fund drew. A fund with a Draws cell on the Spec sheet collects the
    # years it names; every other fund takes its own closing year, plus any carried to it.
    planned = orchestrator.draw_plans
    print(f"\nDraw plans from the Spec sheet's Draws column: {len(planned)} of {len(orchestrator.funds)} funds name their years.")
    if not result.draws.empty:
        forward = result.draws[result.draws["plan_date"] > result.draws["funding_date"]]
        print(f"  {len(result.draws)} drawn years behind {len(result.commitments)} commitments; "
              f"{len(forward)} of them a year the run had not reached, funded at the closing instead.")
        print(result.draws[["multiplier", "rate", "plan_date", "expected_value", "funding_date",
                            "liquid_only_usd", "commitment_usd", "commitment_base"]])
    unclaimed = {t: years for t, years in orchestrator.policy.unclaimed_schedule_years().items() if years}
    for fund_type, years in unclaimed.items():
        print(f"  {fund_type}: no fund draws {years} — that budget goes unspent.")

    print(f"\nThe five running values ({result.base_currency}): liquid alone · liquid at the expected return · "
          f"liquid with the private flows · the private book · the total")
    tracked = result.tracked_values()
    print(pd.concat([tracked.head(3), tracked.tail(3)]))
    print(f"\nWhat moved them, last observations ({result.base_currency}):")
    print(result.periods[PERIOD_COLUMNS].tail(6))
    by_type = result.totals_by_fund_type()
    if not by_type.empty:
        print(f"\nBy fund type at {result.periods.index[-1].date()}:")
        print(by_type.xs(result.periods.index[-1], level="date")[["commitment_usd", "calls_base", "distributions_base", "nav_base"]])
    print(f"\nFunds beyond the horizon (never committed): {result.funds_beyond_horizon or 'none'}")

    comparison = result.compare_with_liquid_only()
    last = comparison.iloc[-1]
    print(f"\nBeside the same liquid portfolio with no private programme, at {comparison.index[-1].date()} ({result.base_currency}):")
    print(f"  liquid only {last['liquid_only']:,.4f} · with programme {last['with_programme']:,.4f} · "
          f"value added {last['value_added']:,.4f} ({last['value_added_share']:.2%} of liquid only, NAV at carrying value)")
    print("\nPublic market equivalent against that liquid portfolio (ks_pme above 1 = the programme beat it):")
    print(result.public_market_equivalent())


def write_csvs(orchestrator, result, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    result.periods.to_csv(out / "periods.csv")
    result.funds.to_csv(out / "funds.csv")
    result.commitments.to_csv(out / "commitments.csv")
    result.draws.to_csv(out / "draws.csv")
    orchestrator.map_events_to_observations().to_csv(out / "map_events_to_observations.csv")
    orchestrator.fund_summary().to_csv(out / "fund_summary.csv")
    result.tracked_values().to_csv(out / "tracked_values.csv")
    result.compare_with_liquid_only().to_csv(out / "liquid_only_comparison.csv")
    result.public_market_equivalent().to_csv(out / "public_market_equivalent.csv")
    print(f"\nWrote periods.csv, funds.csv, commitments.csv, draws.csv, map_events_to_observations.csv, "
          f"fund_summary.csv, tracked_values.csv, liquid_only_comparison.csv, public_market_equivalent.csv to {out}")


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
