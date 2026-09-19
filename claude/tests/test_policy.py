from dataclasses import fields
from datetime import date

import pytest

from pmsim import AnnualRatePolicy, Entitlement, Fund, SizingBalances

BALANCES = SizingBalances(t=3, date=date(2029, 3, 31), liquid_usd=1_000_000.0, private_nav_usd=250_000.0)


def fund(name, fund_type="BUYOUT", closing="2027-03-01"):
    return Fund(name, fund_type, closing)


def test_single_fund_takes_the_whole_rate():
    a = fund("A")
    policy = AnnualRatePolicy({"BUYOUT": {2027: 0.1}}, [a])
    assert policy.size_commitments([a], BALANCES) == {"A": pytest.approx(100_000.0)}
    assert policy.explain_commitment("A", BALANCES) == {
        "policy_year": 2027, "current_year_rate": 0.1, "weight": 1.0,
        "current_year_usd": pytest.approx(100_000.0), "carried_usd": 0.0, "carried_years": ""}


def test_sizing_balances_are_us_dollars_only():
    assert BALANCES.total_usd == 1_250_000.0
    # sizing happens exclusively in USD: a policy is never shown a base-currency amount
    assert [f.name for f in fields(SizingBalances)] == ["t", "date", "liquid_usd", "private_nav_usd", "year_end_liquid_usd"]


def test_two_funds_split_equally_by_default_or_by_weight():
    a, b = fund("A"), fund("B", closing="2027-09-01")
    equal = AnnualRatePolicy({"BUYOUT": {2027: 0.1}}, [a, b])
    assert equal.size_commitments([a, b], BALANCES) == {"A": pytest.approx(50_000.0), "B": pytest.approx(50_000.0)}
    weighted = AnnualRatePolicy({"BUYOUT": {2027: 0.1}}, [a, b], weights={"A": 0.6, "B": 0.4})
    assert weighted.size_commitments([a], BALANCES) == {"A": pytest.approx(60_000.0)}
    assert weighted.entitlements["B"] == Entitlement(2027, 0.1, 0.4, {})


@pytest.mark.parametrize("weights, message", [
    ({"A": 0.6}, "given for all of them or none"),
    ({"A": 0.6, "B": 0.6}, "sum to 1.2, not 1"),
    ({"A": -0.2, "B": 1.2}, "finite non-negative"),
    ({"A": 0.6, "B": 0.4, "Z": 1.0}, "not in the fund list"),
])
def test_weight_validation(weights, message):
    with pytest.raises(ValueError, match=message):
        AnnualRatePolicy({"BUYOUT": {2027: 0.1}}, [fund("A"), fund("B")], weights=weights)


def test_weights_only_bind_within_a_year_and_type():
    a, b = fund("A", closing="2027-03-01"), fund("B", closing="2028-03-01")
    policy = AnnualRatePolicy({"BUYOUT": {2027: 0.1, 2028: 0.2}}, [a, b], weights={"A": 1.0, "B": 1.0})
    assert policy.entitlements["A"] == Entitlement(2027, 0.1, 1.0, {}) and policy.entitlements["B"] == Entitlement(2028, 0.2, 1.0, {})


# ---------------------------------------------------------- carry-forward
RATES = {"BUYOUT": {2027: 0.10, 2028: 0.08, 2029: 0.12}}
AT_C = SizingBalances(t=2, date=date(2029, 3, 31), liquid_usd=1_250_000.0, private_nav_usd=0.0,
                      year_end_liquid_usd={2027: 1_000_000.0, 2028: 1_250_000.0})
AT_D = SizingBalances(t=3, date=date(2029, 6, 30), liquid_usd=1_500_000.0, private_nav_usd=0.0,
                      year_end_liquid_usd={2027: 1_000_000.0, 2028: 1_250_000.0})


def test_carry_forward_sizes_each_year_on_its_own_balance_and_accumulates_dollars():
    c, d = fund("C", closing="2029-03-01"), fund("D", closing="2029-06-01")
    policy = AnnualRatePolicy(RATES, [c, d], weights={"C": 0.6, "D": 0.4}, carry_forward=True)
    assert policy.entitlements["C"] == Entitlement(2029, 0.12, 0.6, {2027: 0.10, 2028: 0.08})
    # 2027: 10% of 1,000,000 · 2028: 8% of 1,250,000 · carried 200,000; 2029 is sized where each fund closes
    assert policy.size_commitments([c], AT_C) == {"C": pytest.approx(0.6 * (200_000 + 0.12 * 1_250_000))}  # 210,000
    assert policy.size_commitments([d], AT_D) == {"D": pytest.approx(0.4 * (200_000 + 0.12 * 1_500_000))}  # 152,000
    assert policy.explain_commitment("D", AT_D) == {
        "policy_year": 2029, "current_year_rate": 0.12, "weight": 0.4,
        "current_year_usd": pytest.approx(180_000.0), "carried_usd": pytest.approx(200_000.0), "carried_years": "2027, 2028"}
    # dollars are carried, not percentages: pooling 30% onto the closing balance would have given 225,000 and 180,000
    assert policy.size_commitments([c], AT_C)["C"] != pytest.approx(0.6 * 0.30 * 1_250_000)


def test_without_carry_forward_a_year_with_no_closing_is_not_used():
    c, d = fund("C", closing="2029-03-01"), fund("D", closing="2029-06-01")
    policy = AnnualRatePolicy(RATES, [c, d], weights={"C": 0.6, "D": 0.4})
    assert policy.entitlements["C"] == Entitlement(2029, 0.12, 0.6, {})
    assert policy.size_commitments([c], AT_C) == {"C": pytest.approx(0.6 * 0.12 * 1_250_000)}
    assert policy.explain_commitment("C", AT_C)["carried_usd"] == 0.0


def test_carried_dollars_go_to_the_next_closing_year_only():
    e, f = fund("E", closing="2028-03-01"), fund("F", closing="2029-03-01")
    policy = AnnualRatePolicy({"BUYOUT": {2027: 0.10, 2028: 0.0, 2029: 0.05}}, [e, f], carry_forward=True)
    assert policy.entitlements["E"] == Entitlement(2028, 0.0, 1.0, {2027: 0.10})  # a zero rate of its own, 2027's budget
    assert policy.entitlements["F"] == Entitlement(2029, 0.05, 1.0, {})  # nothing left over: E collected it
    at_e = SizingBalances(t=1, date=date(2028, 3, 31), liquid_usd=2_000_000.0, private_nav_usd=0.0,
                          year_end_liquid_usd={2027: 1_000_000.0})
    assert policy.size_commitments([e], at_e) == {"E": pytest.approx(100_000.0)}  # 10% of 2027's balance, not of today's


def test_fund_types_carry_independently_and_zero_rates_are_not_carried():
    s, b = fund("S", "SECONDARIES", "2027-03-01"), fund("B", "BUYOUT", "2029-03-01")
    policy = AnnualRatePolicy({"BUYOUT": {2027: 0.1, 2028: 0.0, 2029: 0.1}, "SECONDARIES": {2027: 0.05, 2028: 0.05, 2029: 0.05}},
                              [s, b], carry_forward=True)
    assert policy.entitlements["S"] == Entitlement(2027, 0.05, 1.0, {})
    assert policy.entitlements["B"] == Entitlement(2029, 0.1, 1.0, {2027: 0.1})  # S's closing did not consume it; 2028's 0% is skipped


def test_a_carried_year_uses_the_last_balance_known_by_its_end():
    b = fund("B", closing="2030-03-01")
    policy = AnnualRatePolicy({"BUYOUT": {2026: 0.5, 2027: 0.10, 2028: 0.10, 2029: 0.10, 2030: 0.0}}, [b], carry_forward=True)
    sparse = SizingBalances(t=2, date=date(2030, 3, 31), liquid_usd=9_000_000.0, private_nav_usd=0.0,
                            year_end_liquid_usd={2027: 1_000_000.0, 2029: 3_000_000.0})  # no observation fell in 2028
    assert sparse.liquid_usd_at_end_of(2028) == 1_000_000.0 and sparse.liquid_usd_at_end_of(2026) is None
    # 2026 is before the simulation began: no portfolio, no budget. 2028 is sized on the last balance known by its end.
    assert policy.size_commitments([b], sparse) == {"B": pytest.approx(0.10 * 1_000_000 + 0.10 * 1_000_000 + 0.10 * 3_000_000)}


@pytest.mark.parametrize("rates, funds, years, message", [
    ({"BUYOUT": {2027: 0.1}}, [fund("V", "VC")], None, r"no column for fund type\(s\) \['VC'\]"),
    ({"BUYOUT": {2027: 0.1}}, [fund("A", closing="2028-03-01")], None, r"no row for year\(s\) \[2028\]"),
    ({"BUYOUT": {2028: 0.1}}, [fund("A", closing="2028-03-01")], range(2027, 2029), r"no row for year\(s\) \[2027\]"),
])
def test_missing_rates_are_errors(rates, funds, years, message):
    with pytest.raises(ValueError, match=message):
        AnnualRatePolicy(rates, funds, years=years)


def test_no_funds_and_no_types_is_valid():
    policy = AnnualRatePolicy({}, [], years=range(2027, 2030))
    assert policy.size_commitments([], BALANCES) == {} and policy.entitlements == {}


def test_duplicate_fund_names_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        AnnualRatePolicy({"BUYOUT": {2027: 0.1}}, [fund("A"), fund("A")])
