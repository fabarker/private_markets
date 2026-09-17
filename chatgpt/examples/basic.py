"""Run from the repository root with: python -m chatgpt.examples.basic."""
import pandas as pd

from chatgpt.simulation import Simulation, SimulationConfig
from vintage import FundVintage


def worked_example():
    funds = [
        FundVintage("A", "BUYOUT", commitment_date="2027-02-15",
                    normalized_realized_net_cash_flow=[("2027-03-01", -.25), ("2027-06-01", .05)]),
        FundVintage("B", "BUYOUT", commitment_date="2027-05-10",
                    normalized_realized_net_cash_flow=[("2027-05-20", -.25)]),
    ]
    config = SimulationConfig(
        liquid_total_return_index=pd.Series(
            [1e6, 1.1e6, 1.21e6], index=pd.to_datetime(["2027-01-01", "2027-03-31", "2027-06-30"])),
        funds=funds,
        annual_commitment_rates=pd.DataFrame({"BUYOUT": [.1]}, index=[2027]),
        fund_weights={"A": .6, "B": .4},
    )
    return Simulation(config).run()


def carryforward_example():
    config = SimulationConfig(
        liquid_total_return_index=pd.Series(
            [1e6, 1e6, 1e6, 1.2e6],
            index=pd.to_datetime(["2027-01-01", "2028-12-31", "2029-03-31", "2029-06-30"])),
        funds=[FundVintage("C", "BUYOUT", commitment_date="2029-03-01"),
               FundVintage("D", "BUYOUT", commitment_date="2029-06-01")],
        annual_commitment_rates=pd.DataFrame({"BUYOUT": [.10, .08, .12]}, index=[2027, 2028, 2029]),
        fund_weights={"C": .6, "D": .4},
    )
    return Simulation(config).run()


def shortfall_example():
    config = SimulationConfig(
        liquid_total_return_index=pd.Series([100., 100.], index=pd.to_datetime(["2027-01-01", "2027-02-01"])),
        funds=[FundVintage("A", "BUYOUT", commitment_date="2027-02-01",
                           normalized_realized_net_cash_flow=[("2027-02-01", -1.)])],
        annual_commitment_rates=pd.DataFrame({"BUYOUT": [1.2]}, index=[2027]),
    )
    return Simulation(config).run()


if __name__ == "__main__":
    pd.options.display.float_format = "{:,.2f}".format
    result = worked_example()
    print("Worked example:")
    print(result.portfolio[["liquid_close", "private_nav_close", "total_close"]])
    print("\nPercentage carryforward:")
    carry = carryforward_example()
    print(carry.commitment_events[["pooled_rate", "weight", "effective_rate", "commitment"]])
    print("\nShortfall:")
    failure = shortfall_example().shortfall
    print(f"Stopped on {failure.date.date()}: ${failure.deficit:,.2f} shortfall.")
