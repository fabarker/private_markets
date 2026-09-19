# pmsim — private markets simulator

A liquid portfolio in its base currency funds commitments to US-dollar private funds.
`Simulator(portfolio, funds).run()` walks the liquid index's own dates and reports the liquid
balance, the private NAV, every flow between the two pots, and whether the pot ever ran dry.

This is the implementation of `simulator-design.html` (in this folder). Package modules map to that note:

| Module | Contents |
| --- | --- |
| `pmsim/inputs.py` | `Fund` (fund-held data, USD per $1 committed), `Portfolio` (portfolio-held data, base currency), `PRIVATE_CURRENCY` |
| `pmsim/timeline.py` | `Timeline` (the observation grid), `AlignedFundHistory` (a fund's history on it) |
| `pmsim/state.py` | `LiquidAccount`, `Commitment`, `CommitmentBook` — mutable state during a run |
| `pmsim/policy.py` | `SizingBalances`, `CommitmentPolicy` protocol, `AnnualRatePolicy` |
| `pmsim/simulator.py` | `Simulator`, `SimulationResult`, `Shortfall` — the one loop |
| `pmsim/benchmark.py` | `compare_with_liquid_only`, `public_market_equivalent`, `annualised_irr` — a finished run against the liquid portfolio alone |
| `pmsim/dates.py` | date coercion shared by the above |
| `pmsim/data/tables.py` | the normalized tables a data source must deliver, and the column aliases accepted |
| `pmsim/data/repository.py` | `DataRepository` protocol, `ExcelRepository`, `FrameRepository`, `SheetNames` |
| `pmsim/data/orchestrator.py` | `SimulationSpec`, `Orchestrator`, `run_tables_workbook` — tables in, result out |
| `pmsim/data/workbook.py` | `WorkbookRepository`, `load_profile_workbook` — the five-sheet portfolio workbook, one profile at a time |

## Run

From this directory, with the project's virtual environment:

```bash
../.venv/bin/python -m pytest -q           # tests
../.venv/bin/python -m examples.basic      # worked example, carry-forward, shortfall
../.venv/bin/python -m examples.workbook   # writes a sample workbook, loads it, runs it
../.venv/bin/python -m examples.profile_workbook [book.xlsx USD Conservative 1e6]   # the five-sheet portfolio workbook
../.venv/bin/python -m examples.eur_moderate [book.xlsx [out/]]   # EUR Moderate from $100, converted at the first EURUSD
```

Nothing needs installing: `pyproject.toml` puts `.` on the test path, and `examples` is a
package. Dependencies are NumPy, pandas and openpyxl (for `.xlsx`); pytest for the tests.

**In PyCharm:** select the project's `.venv/bin/python` as the interpreter, open
`examples/eur_moderate.py`, set `WORKBOOK` at the top to your file (the default is
`claude/data/portfolio.xlsx`; `claude/data/` is git-ignored), put a breakpoint in `run()`
and press Debug. The script puts `claude/` on `sys.path` itself, so it runs as a plain file
with any working directory; if the workbook is missing it generates a sample beside it and
says so. `run()` is written as numbered steps — repository, starting balance, spec, funds,
portfolio, policy, result — so each breakpoint shows one object; step into
`orchestrator.run()` to follow the period loop in `Simulator.run()`.

## Usage

```python
from pmsim import AnnualRatePolicy, Fund, Portfolio, Simulator

funds = [
    Fund("A", "BUYOUT", "2027-02-15",
         unit_calls=[("2027-03-01", 0.25)],
         unit_distributions=[("2027-06-01", 0.05)]),
    Fund("B", "BUYOUT", "2027-05-10",
         unit_calls=[("2027-05-20", 0.25)]),
]
portfolio = Portfolio(
    base_currency="GBP",
    liquid_levels=[("2027-01-01", 1_000_000), ("2027-03-31", 1_100_000), ("2027-06-30", 1_210_000)],
    commitment_rates={"BUYOUT": {2027: 0.10}},
    usd_rate=[("2027-01-01", 0.80), ("2027-06-30", 0.75)],   # GBP per 1 USD
)
policy = AnnualRatePolicy(portfolio.commitment_rates, funds, weights={"A": 0.6, "B": 0.4})
result = Simulator(portfolio, funds, policy).run()

result.status          # "completed" or "shortfall"
result.periods         # one row per observation, base currency
result.funds           # one row per live commitment per observation, USD and base
result.commitments     # one row per closing: rate, sizing base, exchange rate, dollars
result.shortfall       # None, or the first failed observation
```

Leave `policy` out to get `AnnualRatePolicy` with equal weights and no carry-forward.

## Inputs, by who holds them

**`Fund`** — what the fund knows. `name` (unique), `fund_type` (a column of the rate table),
`closing_date`, and its realized history per $1 committed, in USD: `unit_calls`,
`unit_distributions` (gross, non-negative; keep them separate even on the same day) and
`unit_nav` (dated marks, at most one per day). Each history may be a Series indexed by
dates, a `{date: value}` mapping, an iterable of `(date, value)` pairs, or omitted. Nothing
may be dated before the closing.

**`Portfolio`** — what the portfolio knows.

- `base_currency` — the currency of the liquid index and of every report. This setting
  decides whether conversion happens: `"USD"` means none and `usd_rate` must be omitted;
  anything else requires `usd_rate`.
- `liquid_levels` — dated total-return levels in base currency. **Their dates are the
  simulation grid and the first level is the starting balance.** Scale the index before
  input; later levels only supply returns and never overwrite the simulated balance.
- `commitment_rates` — calendar year × fund type; `0.10` means commit 10% of the sizing
  base. List every year from the first to the last observation (0 for no target) and every
  fund type in the fund list.
- `usd_rate` — dated price of 1 USD in base currency. Sparse is fine: the last rate on or
  before each observation is used, and one is required on or before the first.

The dollar commitment to each fund is **not** an input. It is the engine's decision at
the closing and lives in `result.commitments`. Input objects are frozen, copy their data,
and are never changed by a run.

## Conventions

**Dates become periods once.** A `Timeline` is built from the liquid dates. A fund is
committed at the first observation on or after its `closing_date`; its rate comes from the
calendar year of the actual closing (a December closing observed in January uses
December's rate). Flows dated in `(previous, current]` belong to `current`. A fund closing
after the last observation is listed in `result.funds_beyond_horizon` and never committed; it
still counts in its year's weight split.

**Unit NAV** is rebuilt from a fund's events in date order, starting at zero: a call adds,
a distribution subtracts, a NAV mark replaces the running value (marks include that day's
flows). The value at an observation is the running value after the last event on or
before it, so marks between observations count and a missing mark leaves the
cash-adjusted estimate. A negative running value is a data error naming the fund and day.

**Each period, in this order:**

1. snapshot opening balances;
2. apply the liquid return `level[t] / level[t-1]` (1 at the first date);
3. bank distributions from existing commitments;
4. size all funds closing at this observation from that same balance, fix each commitment
   in USD at today's rate;
5. bank the new cohort's own distributions (kept out of the sizing base), then pay every
   commitment's calls;
6. value the book, record the period, stop if the calls exceeded the cash.

**Currency.** Private figures are USD until they touch the liquid pot or a report.
Commitments are sized in base currency and converted to USD at the closing observation's
rate, then never change. Calls, distributions and NAV are converted at each observation's
rate. `fx_translation` isolates `nav_usd(t-1) × (fx[t] − fx[t-1])`, the part of private
valuation P&L that is purely the currency moving.

**Shortfall.** If calls exceed the cash available (beyond `cash_tolerance`, default 1e-9),
the failed period is recorded with its negative balance visible, `result.shortfall` names
it, and the run stops. With `stop_on_shortfall=False` the balance goes negative and the run
continues — the "how much would I need to borrow" view. Exactly zero cash is valid; a zero
sizing base gives a zero commitment and still uses the rate.

## Observation frequency

The liquid index sets the observation frequency — monthly month ends, quarter ends,
business days, or any irregular set of dates. Fund events are dated on whatever day they
happened and need not line up with it. One rule covers every mismatch:

**An event on any day pools onto the first observation on or after that day.**

- Calls and distributions dated inside a month land on that month's end on a monthly
  grid; inside a quarter, on that quarter end; on a Saturday, on the next business day of
  a daily grid. Totals are consistent across frequencies: a quarter's calls on a quarterly
  grid equal the sum of its three months' calls on a monthly grid.
- A NAV mark between observations is applied in event order and is what the next
  observation sees, after any flows between the mark and the observation.
- A fund closing intramonth is committed at that month's end, sized from that month
  end's liquid balance, and its calls in the same month are paid in that period.
- Exchange rates go the other way, because a rate is a state rather than an event: each
  observation uses the last rate on or before it.

`Simulator.map_events_to_observations()` lists every fund event with the observation it pooled onto, for
audit. Because flows settle at observations, a call dated the 3rd and one dated the 28th
are treated alike within the month: neither loses nor earns that month's return, and a
liquidity shortfall is detected at the month end, not on the day. A finer grid makes the
simulation finer; the rule does not change.

## Results

`periods` (index `date`, base currency unless noted): `liquid_open`, `private_open`,
`total_open`, `return_factor`, `usd_rate`, `liquid_pnl`, `distributions`, `sizing_base`,
`commitments`, `commitments_usd`, `calls`, `liquid_close`, `private_close`, `total_close`,
`private_valuation_pnl`, `fx_translation`.

`funds` (index `date`, `fund`): `fund_type`, `commitment_usd`, `calls_usd`,
`distributions_usd`, `nav_usd`, `calls_base`, `distributions_base`, `nav_base`.

`commitments` (index `date`, `fund`): `fund_type`, `closing_date`, `policy_year`,
`sizing_base`, `rate`, `commitment_base`, `usd_rate`, `commitment_usd`, and from
`AnnualRatePolicy` the `current_year_rate`, `carried_rate`, `pooled_rate` and `weight`
behind the rate.

`result.totals_by_fund_type()` sums the fund table by date and fund type; `result.nav_by_fund()` is
private NAV in base currency by date × fund. Every completed period satisfies

```text
liquid_close  = liquid_open + liquid_pnl + distributions − calls
private_valuation_pnl = private_close − private_open − calls + distributions
total_close − total_open = liquid_pnl + private_valuation_pnl
```

and the tests assert these on every run.

## Benchmarking a run

Every call is paid by selling the liquid portfolio and every distribution buys it back, so a
run already is a public-market-equivalent calculation against the investor's own portfolio.
Two methods read it off the result; nothing in the loop is involved.

```python
result.compare_with_liquid_only()    # by date: liquid_only, with_programme, value_added, value_added_share
result.public_market_equivalent()    # programme, then each fund type, then each fund
```

`compare_with_liquid_only()` sets the run's `total_close` beside the same liquid portfolio
with no private programme. `public_market_equivalent()` (index `level`, `name`; base
currency; the liquid index as benchmark) gives `calls`, `distributions`, `nav`, the flows
compounded to the last observation (`fv_calls`, `fv_distributions`), `value_added`,
`ks_pme` (above 1 = the programme beat the liquid portfolio), `irr`, and `direct_alpha` —
the annualised rate of out- or under-performance, the IRR of the index-compounded flows.
With `I` the liquid index and `T` the last observation the link between the two is exact:

```text
with_programme(T) − liquid_only(T) = Σ (distributions(t) − calls(t)) × I(T)/I(t) + private_close(T)
                                   = fv_calls × (ks_pme − 1)
```

so the programme's `value_added` equals the comparison's last row, and the funds' (and the
fund types') add up to it; the tests assert all three. Private NAV is counted at its
carrying value, so part of the value added is unrealised. Ratios are NaN where undefined:
nothing called yet, or every flow on one date. `annualised_irr(dates, amounts)` (ACT/365,
money out negative) is exported for use on its own.

## Policy

`AnnualRatePolicy(rates, funds, weights=None, carry_forward=False, years=None)` gives each
fund `rate[closing year, type] × weight`. Weights split a year's rate among the funds of
one type closing that year; give them for all funds of such a group or none (equal split),
summing to 1. With `carry_forward=True` a year in which no fund of a type closes adds its
rate to the next year of that type that has one: 10% + 8% + 12% with 60/40 weights gives
18% and 12%. Fund types are independent.

Any object with `size_commitments(cohort, balances) -> {fund name: base-currency amount}` is a policy; the
`SizingBalances` it receives offer `liquid`, `private_nav` and `total`. If it also has
`explain_rate(fund_name)`, those figures land in the commitments table.

## The portfolio workbook

The five-sheet workbook — `Liquid`, `FX`, `Flows`, `Commitments`, `Spec` — is loaded one
*profile* at a time. A profile is a currency and a risk level; it selects the liquid return
column, the commitment-schedule rows, and (for a non-USD currency) the FX column.

```python
from pmsim.data import load_profile_workbook

o = load_profile_workbook("portfolio.xlsx", "USD", "Conservative", initial_value=1_000_000)
o.repository.commitment_rates()   # the schedule for this profile on calendar years
o.fund_summary()
result = o.run()
```

`python -m examples.profile_workbook book.xlsx EUR Conservative 5e6` does the same from
the command line, and with no arguments writes a sample workbook in this layout to
compare a real file against.

| Sheet | Layout | How it is read |
| --- | --- | --- |
| `Liquid` | blank header, then one column per profile: `USD Conservative`, `EUR Moderate`, … | **monthly returns**. The frequency is inferred, the simulation starts one period before the first return (rolled back to a business day; the profile's `initial_value` is the balance there), and every return is applied. Pass `inception_date=` to `load_profile_workbook` when the dates are too irregular to infer. The FX sheet's first rate is taken to apply at inception |
| `FX` | blank header, then `EURUSD`, `GBPUSD`, … (USD per 1 unit of the currency) | the profile's `<CCY>USD` column, inverted to base-per-USD; `USD<CCY>` is also recognised and used as is |
| `Flows` | `Vintage`, `Date`, `Value`, `Type` (`Flow`/`NAV`), `Scale` | `Vintage` is the fund name; unit = `Value ÷ Scale`; negative flows are calls |
| `Commitments` | `Type`, `Year`, `Currency`, `Risk`, `Commitment`, `Rate` | rows of the profile; `Year` is **years since inception** (0 = the year of the first Liquid date) and is mapped onto calendar years; `Rate` is the decimal used |
| `Spec` | `Name`, `Year`, `Type` | `Year` holds the closing date, read day-first (`31/12/2010`) |

Sheet names can be overridden with `SheetLayout(...)`; any keyword accepted by
`SimulationSpec` (`weights`, `carry_forward`, `stop_on_shortfall`, …) can be passed to
`load_profile_workbook` as an override.

## Loading from a one-table-per-sheet workbook

`pmsim.data` also reads a workbook laid out as the normalized tables (a database later):

```python
from pmsim.data import SimulationSpec, load_tables_workbook

spec = SimulationSpec(
    base_currency="GBP",
    liquid_series="liquid_gbp",      # market_data column holding the liquid total-return level
    fx_series="gbp_per_usd",         # market_data column holding the USD rate; omit when base is USD
    fx_quote="base_per_usd",         # or "usd_per_base" if the sheet quotes USD per 1 GBP
    weights={"A": 0.6, "B": 0.4},    # optional; carry_forward=True also available
    # commitment_rates=...           # optional: overrides the workbook's commitment_rates sheet
)
orchestrator = load_tables_workbook("portfolio.xlsx", spec)
orchestrator.fund_summary()          # what was loaded per fund, and whether it closes in the horizon
orchestrator.map_events_to_observations()             # where every fund event pools on the liquid grid
result = orchestrator.run()
```

The workbook's sheets (names matched case-, space- and hyphen-insensitively; override with
`SheetNames`):

| Sheet | Columns | Notes |
| --- | --- | --- |
| `fund_spec` | `fund_name`, `type`, `closing_date` | one row per fund; `type` is the fund type |
| `fund_market_data` | `fund_name`, `type`, `value`, `date`, `scale` | long form; `type` is `Flow` or `NAV` (also `Call`, `Distribution`); **unit = value ÷ scale** |
| `market_data` | `date` + one column per series | wide, or long with `date`, `series`, `value`; blanks are fine (sparse FX) |
| `commitment_rates` | `year` + one column per fund type | optional; or long with `year`, `type`, `rate`; every year, 0 for none |

Column names are matched by alias (`Fund Name`, `fund`, `name` → `fund_name`; `Strategy`
→ fund type; `Amount` → value; `Divisor`/`Commitment` → scale, and so on). `Flow` rows
follow the LP's sign convention — negative is a call, positive a distribution — flip it
with `calls_are_negative=False`; `Call`/`Distribution` rows are read as magnitudes. A fund
in `fund_spec` with no market rows is a future closing with an empty history; market
rows for a fund that is not in `fund_spec` are an error.

`FrameRepository(fund_specs, fund_market_data, market_data, commitment_rates)` takes the
same tables as DataFrames — the shape a database adapter will take: implement the four
`DataRepository` methods and hand the object to `Orchestrator`.

## Out of scope

Pre-existing commitments at inception, non-USD funds, external cash flows, fees, taxes,
recycling, secondary sales and stochastic returns. Monte Carlo is a loop outside the
engine: one `Portfolio` per liquid or FX path, funds shared.
