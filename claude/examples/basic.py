"""Three runs of the simulator. From the claude directory: python -m examples.basic

1. The worked example from the design note: a sterling portfolio, two dollar buyout funds,
   a 10% annual rate split 60/40, and the dollar weakening between the two closings.
2. Carry-forward: two years with no fund of the type, each sized on its own year-end balance;
   the dollars wait for the third year's funds.
3. A pacing schedule: amounts per 1 of liquid value on the first commitment date, turned into a share of the
   liquid value by dividing by the pacing model's expected value, which grows at the expected return X.
4. A liquidity shortfall: the run stops at the observation whose calls exceed the cash.
"""
import pandas as pd

from pmsim import AnnualRatePolicy, Fund, Portfolio, Simulator


def worked_example_gbp():
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
        usd_rate=[("2027-01-01", 0.80), ("2027-06-30", 0.75)],   # GBP per 1 USD; sparse, carried forward
    )
    policy = AnnualRatePolicy(portfolio.commitment_rates, funds, weights={"A": 0.6, "B": 0.4})
    return Simulator(portfolio, funds, policy).run()


def carry_forward_example():
    """No BUYOUT fund closes in 2027 or 2028: 10% of 1,000,000 and 8% of 1,250,000 wait, as dollars, for 2029's funds."""
    funds = [Fund("C", "BUYOUT", "2029-03-01"), Fund("D", "BUYOUT", "2029-06-01")]
    portfolio = Portfolio(
        base_currency="USD",
        liquid_levels=[("2027-12-31", 1_000_000), ("2028-12-31", 1_250_000),
                       ("2029-03-31", 1_250_000), ("2029-06-30", 1_500_000)],
        commitment_rates={"BUYOUT": {2027: 0.10, 2028: 0.08, 2029: 0.12}},
    )
    policy = AnnualRatePolicy(portfolio.commitment_rates, funds, weights={"C": 0.6, "D": 0.4},
                              carry_forward=True, years=portfolio.calendar_years)
    return Simulator(portfolio, funds, policy).run()


def pacing_schedule_example():
    """A schedule that grows 5% a year is a constant 2% of a portfolio expected to grow 5% a year."""
    funds = [Fund("P1", "BUYOUT", "2028-12-31"), Fund("P2", "BUYOUT", "2029-12-31"), Fund("P3", "BUYOUT", "2030-12-31")]
    portfolio = Portfolio(
        base_currency="USD",
        liquid_levels=[("2027-12-31", 1_000_000), ("2028-12-31", 1_100_000),
                       ("2029-12-31", 1_100_000), ("2030-12-31", 1_331_000)],
        commitment_rates={"BUYOUT": {2027: 0.0, 2028: 0.02, 2029: 0.021, 2030: 0.02205}},
    )
    policy = AnnualRatePolicy(portfolio.commitment_rates, funds, expected_return=0.05, years=portfolio.calendar_years)
    return Simulator(portfolio, funds, policy).run()


def shortfall_example():
    funds = [Fund("S", "VC", "2027-02-01", unit_calls=[("2027-02-01", 1.0)])]
    portfolio = Portfolio(
        base_currency="USD",
        liquid_levels=[("2027-01-01", 100.0), ("2027-02-01", 100.0)],
        commitment_rates={"VC": {2027: 1.2}},
    )
    return Simulator(portfolio, funds).run()


if __name__ == "__main__":
    pd.options.display.float_format = "{:,.2f}".format
    pd.options.display.width = 200

    result = worked_example_gbp()
    print(f"Worked example ({result.base_currency} base) — {result.status}")
    print(result.periods[["liquid_open", "liquid_pnl", "distributions", "liquid_only", "commitments",
                          "calls", "liquid_close", "private_close", "total_close", "fx_translation"]].T)
    print("\nCommitments (sized in USD; base-currency figures are that day's translation):")
    print(result.commitments[["closing_date", "rate", "sizing_base_usd", "commitment_usd", "usd_rate", "commitment_base"]])
    print("\nFunds:")
    print(result.funds)
    print("\nThe five running values:")
    print(result.tracked_values())
    print("\nBeside the same liquid portfolio with no private programme:")
    print(result.compare_with_liquid_only())
    print("\nPublic market equivalent against that liquid portfolio:")
    print(result.public_market_equivalent().T)

    carry = carry_forward_example()
    print("\nCarry-forward — 2027 and 2028 sized on their own year-end balances, collected 60/40 by 2029's funds:")
    print(carry.commitments[["own_year_rate", "weight", "own_year_usd", "other_years_usd", "drawn_years", "commitment_usd"]])

    pacing = pacing_schedule_example()
    print("\nPacing schedule with X = 5% — expected value 1 on the first commitment date, then 1.05, 1.1025:")
    print(pacing.commitments[["sizing_base_usd", "own_year_rate", "expected_value", "rate", "commitment_usd"]])

    failure = shortfall_example()
    print(f"\nShortfall example — {failure.status}: {failure.shortfall}")
    print(failure.periods[["liquid_only", "commitments", "calls", "liquid_close"]])
