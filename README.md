# Private markets simulation

`Simulation` combines a liquid total-return index with specific `FundVintage`
objects. It calculates commitments, gross calls/distributions, private NAV and
liquid balances on the index's calendar-date grid. The implementation follows
`simulation-implementation-plan.html`, including percentage carryforward.

## Run locally

From this project directory, using its Python 3.11+ environment:

```bash
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m examples.basic
.venv/bin/python -m pytest -q
```

In PyCharm, select this project's `.venv/bin/python` as the interpreter. The
example also works as the module `examples.basic` with this directory as its
working directory. Dependencies are declared in `pyproject.toml`.

## Usage

```python
import pandas as pd
from simulation import Simulation, SimulationConfig
from vintage import FundVintage

liquid_index = pd.Series(
    [1_000_000, 1_100_000, 1_210_000],
    index=pd.to_datetime(["2027-01-01", "2027-03-31", "2027-06-30"]),
)

funds = [
    FundVintage(
        name="Buyout A",
        strategy="BUYOUT",
        commitment_date="2027-02-15",
        normalized_realized_net_cash_flow=[
            ("2027-03-01", -0.25),
            ("2027-06-01", 0.05),
        ],
    ),
    FundVintage(
        name="Buyout B",
        strategy="BUYOUT",
        commitment_date="2027-05-10",
        normalized_realized_net_cash_flow=[("2027-05-20", -0.25)],
    ),
]

config = SimulationConfig(
    liquid_total_return_index=liquid_index,
    funds=funds,
    annual_commitment_rates=pd.DataFrame({"BUYOUT": [0.10]}, index=[2027]),
    fund_weights={"Buyout A": 0.60, "Buyout B": 0.40},
)
simulation = Simulation(config)
result = simulation.run()  # simulate() is an equivalent alias.

print(result.portfolio)
print(result.commitment_events)
if result.shortfall is not None:
    print(result.shortfall.date, result.shortfall.deficit)
```

The example's commitments are $66,000 and $47,806. Final liquid assets are
$1,183,198.50 and final total portfolio value is $1,208,350.

## Input contract

- The first liquid index level **is the starting dollar value**. Rebase or scale
  the input before simulation if necessary. Later ratios are applied to the
  simulated liquid balance; later raw index levels never overwrite that balance.
- Index observations are unique, increasing, timezone-naive calendar dates.
  Daily, monthly, quarterly and irregular grids work. Intraday timestamps are
  rejected, not truncated. A single-date simulation is valid.
- Each fund is a distinct investment with a unique `name`, a string `strategy`
  matching a policy column, and a `commitment_date` on/after inception. String-valued
  strategy enums are also accepted. Closing date and commitment date are synonyms.
- Histories are per dollar committed. Negative flow entries are calls and positive
  entries are distributions. Supply gross entries separately even on the same day;
  the engine cannot recover gross flows from already-netted data.
- Existing `commitment_size` values are ignored and reported as diagnostics.
  Calculated commitments live in the result; input objects are never resized.
- Annual policy rows are integer calendar years; values are decimal percentages.
  Supply every year from inception through the horizon for every configured type,
  including years with no funds. Use explicit zeroes for no target. A policy-only
  fund type is valid even if no fund of that type is in the list.
- Weights sum to 1 within each **actual closing year and fund type**, including
  listed funds beyond a partial-year horizon. One fund defaults to weight 1;
  multiple funds require all weights. Unknown fund names and missing rates fail
  validation. Do not omit later same-year funds whose shares must be reserved.
- No existing/historical commitments, currency conversion, external wealth flows,
  inflation, or stochastic return generation are included.

`Simulation` snapshots the config at construction. Later changes to its source
Series, DataFrame, fund objects or dictionaries do not change this simulation.
Construct another instance to use changed inputs. Every call to `run()` starts
fresh, including after a shortfall.

## Percentage carryforward

Targets accrue once per calendar year, even if the observation grid skips years.
The inception year receives its full supplied rate. Opening carry is zero;
policy years outside the simulation's calendar years are excluded and reported.

For each fund type independently:

1. Add the current annual percentage to missed-year percentages.
2. If no fund closes that year, carry the whole pool onward without spending cash.
3. Otherwise allocate the pool once to all that year's funds using their weights.
   Shares for later closings are **reserved percentages**, not unallocated carry.
4. At each fund's effective closing, multiply its percentage share by the sizing
   liquid NAV. The dollar commitment then remains fixed.

For example, missed BUYOUT targets of 10% and 8%, followed by a 12% target in a
closing year, give a 30% pool. With 60/40 fund weights, the two funds use 18% and
12% of their respective closing NAVs. The pool is not reapplied at each closing.
A zero current-year target does not erase carryforward. Percentages are added,
not compounded, and there is no 100% cap or automatic expiry.

At the horizon, report any unallocated carry and reserved shares separately.
They have no fixed dollar value until a closing NAV exists. Applying a percentage
against zero liquidity produces a zero commitment and consumes that percentage;
there is no automatic retry.

## Calculation order

At the first observation, initialize liquid assets from the index, apply a return
factor of 1, and process same-day closings and flows. Later buckets are
`(previous_date, current_date]`.

Each observation:

1. Snapshot opening liquid/private NAV and accrue any newly reached annual targets.
2. Apply the liquid index return to the opening simulated liquid balance.
3. Add distributions from previously activated funds. This is the sizing balance.
4. Size all funds closing at this observation against that **same** balance.
5. Include newly activated funds' distributions, then aggregate all capital calls.
6. Stop if cash is insufficient; otherwise complete cash and private NAV updates.

Off-grid closings move to the next observation, retaining their actual calendar
year's percentage entitlement. A December closing observed in January does not
absorb January's new target. New funds' same-bucket distributions enter after
sizing to avoid circular commitments. Commitment creation alone never spends cash.

This is a period-level cash convention: all private flows settle after the liquid
return. It does not detect intra-period liquidity deficits. Different observation
grids can therefore give different commitment amounts and portfolio paths.

## Private NAV

Process actual event dates in order, including marks between observations.
Between marks, `NAV = prior NAV + calls - distributions`, with no assumed
investment return. On a mark date, apply that day's net flow and then replace NAV
with the explicit mark. Marks are assumed to include same-day flows. Zero marks
are real resets, not missing data. No future mark is used.

Materially negative inferred NAV raises `ValuationError` with fund/date/value
context. Its check occurs when that period is reached, after the liquidity check,
so a later valuation error does not hide an earlier cash shortfall. Floating-point
negative NAV within `cash_tolerance` is rounded to zero.

## Results

`SimulationResult` contains:

| Attribute | Contents |
| --- | --- |
| `status` | `completed` or `liquidity_shortfall` |
| `portfolio` | Per-date opening/closing balances, returns, calls, distributions, new commitments and valuation P&L |
| `fund_detail` | Date/fund rows with activation, fixed commitment, NAV, flows, cumulative flows and latest mark date |
| `strategy_detail` | Date/type aggregations, including policy-only types |
| `commitment_events` | Actual/effective closing dates, target/carry/pool, weight, effective rate, weighted source-year rates, sizing NAV and dollars |
| `commitment_budget` | Date/type target, used, carried and reserved percentages, including source-year breakdown dictionaries |
| `shortfall` | Failure date, available cash, required calls, deficit, calls by fund and candidate commitment/budget tables, or `None` |
| `diagnostics` | Ignored sizes/policy years, outside-horizon events, outstanding percentages and flow conventions |

Numeric/date column types and table indexes are defined even for empty tables.
Dollar values are floats; percentages are decimals. `commitment` is a cumulative
fixed-commitment total at strategy level, not an unfunded commitment estimate.

On a shortfall, completed tables end **before** the failed observation. Candidate
commitments and annual accruals are in the failure report, not completed history.
Its candidate budget contains `attempted_used_percentage`; attempted shares remain
reserved. The engine does not reduce calls, borrow, or continue. A first-date
failure returns empty completed tables and a populated failure report.

The default cash tolerance is $0.00000001. If a deficit is within tolerance, cash
is rounded to zero and `cash_rounding_adjustment` reports the correction. Accounting
then reconciles as:

```text
liquid_close = liquid_open + liquid_investment_pnl
               + distributions - capital_calls + cash_rounding_adjustment
total_close = liquid_close + private_nav_close
total_close - total_open = liquid_investment_pnl
                           + private_valuation_pnl + cash_rounding_adjustment

For each fund type:
cumulative_target_percentage = cumulative_used_percentage
                              + unallocated_carried_percentage
                              + reserved_percentage
```

Bad input raises `SimulationValidationError` during construction. Computed numeric
overflow raises an arithmetic error; it is not classified as a liquidity shortfall.

## Changes to FundVintage

- Allows future vintage years through Python's calendar limit, independent of the
  computer's current year. A vintage year may differ from its commitment year.
- Validates `HistoryEntry` even when directly constructed; retains zeros and rejects
  malformed/nonfinite entries, negative NAV marks, and duplicate marks. **Invalid
  histories no longer disappear silently**, including entries with `None`/NaN.
- Normalizes midnight datetimes consistently and rejects intraday/timezone inputs.
- Applies validation to `add_*` mutation helpers before modifying the history.
- Adds `nav_on(date)` and `nav_series(dates)` using the generic NAV reconstruction,
  plus `total_called_as_of(date)` and `total_distributed_as_of(date)`.
- Corrects `irr(as_of=...)` to adjust the terminal NAV for flows after its last
  mark. It still uses a bracketed XIRR solver; it does not enumerate multiple roots.
- Preserves the existing scaled views, serialization and workbook-specific
  `monthly_nav()` / `monthly_flows()` methods. The simulator does not call the
  workbook-specific helpers. Properties like `latest_nav` and `total_called` still
  describe the entire supplied history; use the new as-of helpers for dated views.

## Files

- `simulation.py`: config/result types, validation, policy budgets, simulation loop.
- `simulation_events.py`: generic date alignment and chronological unit NAV sweep.
- `vintage.py`: fund model and reporting helpers.
- `examples/basic.py`: worked example, weighted carryforward and shortfall usage.
- `tests/`: accounting, policy, timing, validation and fund-helper regression tests.
