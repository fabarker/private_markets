from datetime import date

import pytest

from pmsim import AnnualRatePolicy, Fund, SizingBalances

BALANCES = SizingBalances(t=3, date=date(2029, 3, 31), liquid=1_000_000.0, private_nav=250_000.0)


def fund(name, fund_type="BUYOUT", closing="2027-03-01"):
    return Fund(name, fund_type, closing)


def test_single_fund_takes_the_whole_rate():
    a = fund("A")
    policy = AnnualRatePolicy({"BUYOUT": {2027: 0.1}}, [a])
    assert policy.size_commitments([a], BALANCES) == {"A": pytest.approx(100_000.0)}
    assert policy.explain_rate("A") == {"policy_year": 2027, "current_year_rate": 0.1, "carried_rate": 0.0,
                                   "pooled_rate": 0.1, "weight": 1.0, "effective_rate": 0.1}


def test_sizing_base_total():
    assert BALANCES.total == 1_250_000.0


def test_two_funds_split_equally_by_default_or_by_weight():
    a, b = fund("A"), fund("B", closing="2027-09-01")
    equal = AnnualRatePolicy({"BUYOUT": {2027: 0.1}}, [a, b])
    assert equal.size_commitments([a, b], BALANCES) == {"A": pytest.approx(50_000.0), "B": pytest.approx(50_000.0)}
    weighted = AnnualRatePolicy({"BUYOUT": {2027: 0.1}}, [a, b], weights={"A": 0.6, "B": 0.4})
    assert weighted.size_commitments([a], BALANCES) == {"A": pytest.approx(60_000.0)}
    assert weighted.entitlements["B"].effective_rate == pytest.approx(0.04)


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
    assert policy.entitlements["A"].effective_rate == 0.1 and policy.entitlements["B"].effective_rate == 0.2


def test_carry_forward_pools_missed_years_once():
    c, d = fund("C", closing="2029-03-01"), fund("D", closing="2029-06-01")
    rates = {"BUYOUT": {2027: 0.10, 2028: 0.08, 2029: 0.12}}
    pooled = AnnualRatePolicy(rates, [c, d], weights={"C": 0.6, "D": 0.4}, carry_forward=True)
    assert pooled.entitlements["C"] == pooled.entitlements["C"].__class__(2029, 0.12, pytest.approx(0.18), pytest.approx(0.30), 0.6, pytest.approx(0.18))
    assert pooled.entitlements["D"].effective_rate == pytest.approx(0.12)
    plain = AnnualRatePolicy(rates, [c, d], weights={"C": 0.6, "D": 0.4})
    assert plain.entitlements["C"].effective_rate == pytest.approx(0.072)
    assert plain.entitlements["D"].effective_rate == pytest.approx(0.048)


def test_carry_is_consumed_by_the_first_closing_year_only():
    e, f = fund("E", closing="2028-03-01"), fund("F", closing="2029-03-01")
    policy = AnnualRatePolicy({"BUYOUT": {2027: 0.10, 2028: 0.0, 2029: 0.05}}, [e, f], carry_forward=True)
    assert policy.entitlements["E"].effective_rate == pytest.approx(0.10)  # zero current-year rate still gets the carry
    assert policy.entitlements["F"].effective_rate == pytest.approx(0.05)  # nothing left over


def test_fund_types_are_independent():
    s, b = fund("S", "SECONDARIES", "2027-03-01"), fund("B", "BUYOUT", "2028-03-01")
    policy = AnnualRatePolicy({"BUYOUT": {2027: 0.1, 2028: 0.1}, "SECONDARIES": {2027: 0.05, 2028: 0.05}},
                              [s, b], carry_forward=True)
    assert policy.entitlements["S"].effective_rate == pytest.approx(0.05)
    assert policy.entitlements["B"].effective_rate == pytest.approx(0.2)  # BUYOUT's 2027 carry is untouched by S


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
