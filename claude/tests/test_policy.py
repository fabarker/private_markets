import math
from dataclasses import fields
from datetime import date

import pytest

from pmsim import (AnnualRatePolicy, Entitlement, Fund, SizingBalances, YearEndBalance,
                   commitment_rounding_unit, round_like_excel)

BALANCES = SizingBalances(t=3, date=date(2029, 3, 31), liquid_only_usd=1_000_000.0, liquid_account_usd=900_000.0,
                          private_nav_usd=250_000.0)


def fund(name, fund_type="BUYOUT", closing="2027-03-01"):
    return Fund(name, fund_type, closing)


def test_single_fund_takes_the_whole_rate():
    a = fund("A")
    policy = AnnualRatePolicy({"BUYOUT": {2027: 0.1}}, [a])
    assert policy.size_commitments([a], BALANCES) == {"A": pytest.approx(100_000.0)}
    explained = policy.explain_commitment("A", BALANCES)
    assert math.isnan(explained.pop("expected_value"))  # no expected return: the rates are shares already
    assert explained == {"policy_year": 2027, "own_year_rate": 0.1, "weight": 1.0,
                         "own_year_usd": pytest.approx(100_000.0), "other_years_usd": 0.0, "drawn_years": "2027"}


def test_sizing_balances_are_us_dollars_only_and_name_the_liquid_only_value():
    assert BALANCES.total_usd == 900_000.0 + 250_000.0  # the account and the private book: what the investor actually holds
    # sizing happens exclusively in USD, and on the liquid-only value: both facts are in the field names
    assert [f.name for f in fields(SizingBalances)] == [
        "t", "date", "liquid_only_usd", "liquid_account_usd", "private_nav_usd", "year_ends", "first_commitment_date"]


def test_commitments_are_sized_on_the_liquid_only_value_not_on_the_account():
    a = fund("A")
    policy = AnnualRatePolicy({"BUYOUT": {2027: 0.1}}, [a])
    drained = SizingBalances(t=3, date=date(2029, 3, 31), liquid_only_usd=1_000_000.0, liquid_account_usd=0.0, private_nav_usd=0.0)
    assert policy.size_commitments([a], drained) == {"A": pytest.approx(100_000.0)}  # calls paid elsewhere change nothing


def test_two_funds_split_equally_by_default_or_by_weight():
    a, b = fund("A"), fund("B", closing="2027-09-01")
    equal = AnnualRatePolicy({"BUYOUT": {2027: 0.1}}, [a, b])
    assert equal.size_commitments([a, b], BALANCES) == {"A": pytest.approx(50_000.0), "B": pytest.approx(50_000.0)}
    weighted = AnnualRatePolicy({"BUYOUT": {2027: 0.1}}, [a, b], weights={"A": 0.6, "B": 0.4})
    assert weighted.size_commitments([a], BALANCES) == {"A": pytest.approx(60_000.0)}
    assert weighted.entitlements["B"] == Entitlement(2027, "BUYOUT", 0.4, {2027: 1.0})


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
    assert policy.entitlements["A"] == Entitlement(2027, "BUYOUT", 1.0, {2027: 1.0})
    assert policy.entitlements["B"] == Entitlement(2028, "BUYOUT", 1.0, {2028: 1.0})


# ---------------------------------------------------------- carry-forward
RATES = {"BUYOUT": {2027: 0.10, 2028: 0.08, 2029: 0.12}}
YEAR_ENDS = {2027: YearEndBalance(date(2027, 12, 31), 1_000_000.0), 2028: YearEndBalance(date(2028, 12, 31), 1_250_000.0)}


def balances(on, liquid_only_usd, year_ends=None, first_commitment_date=None, t=0):
    return SizingBalances(t=t, date=on, liquid_only_usd=liquid_only_usd, liquid_account_usd=liquid_only_usd,
                          private_nav_usd=0.0, year_ends=year_ends or {}, first_commitment_date=first_commitment_date)


AT_C = balances(date(2029, 3, 31), 1_250_000.0, YEAR_ENDS)
AT_D = balances(date(2029, 6, 30), 1_500_000.0, YEAR_ENDS)


def test_carry_forward_sizes_each_year_on_its_own_balance_and_accumulates_dollars():
    c, d = fund("C", closing="2029-03-01"), fund("D", closing="2029-06-01")
    policy = AnnualRatePolicy(RATES, [c, d], weights={"C": 0.6, "D": 0.4}, carry_forward=True)
    assert policy.entitlements["C"] == Entitlement(2029, "BUYOUT", 0.6, {2027: 1.0, 2028: 1.0, 2029: 1.0})
    # 2027: 10% of 1,000,000 · 2028: 8% of 1,250,000 · carried 200,000; 2029 is sized where each fund closes
    assert policy.size_commitments([c], AT_C) == {"C": pytest.approx(0.6 * (200_000 + 0.12 * 1_250_000))}  # 210,000
    assert policy.size_commitments([d], AT_D) == {"D": pytest.approx(0.4 * (200_000 + 0.12 * 1_500_000))}  # 152,000
    explained = policy.explain_commitment("D", AT_D)
    assert (explained["own_year_usd"], explained["other_years_usd"], explained["drawn_years"]) == \
        (pytest.approx(180_000.0), pytest.approx(200_000.0), "2027, 2028, 2029")
    # dollars are carried, not percentages: pooling 30% onto the closing balance would have given 225,000 and 180,000
    assert policy.size_commitments([c], AT_C)["C"] != pytest.approx(0.6 * 0.30 * 1_250_000)


def test_without_carry_forward_a_year_with_no_closing_is_not_used():
    c, d = fund("C", closing="2029-03-01"), fund("D", closing="2029-06-01")
    policy = AnnualRatePolicy(RATES, [c, d], weights={"C": 0.6, "D": 0.4})
    assert policy.entitlements["C"] == Entitlement(2029, "BUYOUT", 0.6, {2029: 1.0})
    assert policy.size_commitments([c], AT_C) == {"C": pytest.approx(0.6 * 0.12 * 1_250_000)}
    assert policy.explain_commitment("C", AT_C)["other_years_usd"] == 0.0


def test_carried_dollars_go_to_the_next_closing_year_only():
    e, f = fund("E", closing="2028-03-01"), fund("F", closing="2029-03-01")
    policy = AnnualRatePolicy({"BUYOUT": {2027: 0.10, 2028: 0.0, 2029: 0.05}}, [e, f], carry_forward=True)
    assert policy.entitlements["E"] == Entitlement(2028, "BUYOUT", 1.0, {2027: 1.0, 2028: 1.0})  # a zero rate of its own, 2027's budget
    assert policy.entitlements["F"] == Entitlement(2029, "BUYOUT", 1.0, {2029: 1.0})  # nothing left over: E collected it
    at_e = balances(date(2028, 3, 31), 2_000_000.0, {2027: YearEndBalance(date(2027, 12, 31), 1_000_000.0)})
    assert policy.size_commitments([e], at_e) == {"E": pytest.approx(100_000.0)}  # 10% of 2027's balance, not of today's


def test_fund_types_carry_independently_and_zero_rates_are_not_carried():
    s, b = fund("S", "SECONDARIES", "2027-03-01"), fund("B", "BUYOUT", "2029-03-01")
    policy = AnnualRatePolicy({"BUYOUT": {2027: 0.1, 2028: 0.0, 2029: 0.1}, "SECONDARIES": {2027: 0.05, 2028: 0.05, 2029: 0.05}},
                              [s, b], carry_forward=True)
    assert policy.entitlements["S"] == Entitlement(2027, "SECONDARIES", 1.0, {2027: 1.0})
    assert policy.entitlements["B"] == Entitlement(2029, "BUYOUT", 1.0, {2027: 1.0, 2029: 1.0})  # S's closing did not consume it; 2028's 0% is skipped


def test_a_carried_year_uses_the_last_balance_known_by_its_end():
    b = fund("B", closing="2030-03-01")
    policy = AnnualRatePolicy({"BUYOUT": {2026: 0.5, 2027: 0.10, 2028: 0.10, 2029: 0.10, 2030: 0.0}}, [b], carry_forward=True)
    sparse = balances(date(2030, 3, 31), 9_000_000.0, {2027: YearEndBalance(date(2027, 12, 31), 1_000_000.0),
                                                       2029: YearEndBalance(date(2029, 12, 31), 3_000_000.0)})  # none fell in 2028
    assert sparse.year_end_on_or_before(2028).liquid_only_usd == 1_000_000.0 and sparse.year_end_on_or_before(2026) is None
    # 2026 is before the simulation began: no portfolio, no budget. 2028 is sized on the last balance known by its end.
    assert policy.size_commitments([b], sparse) == {"B": pytest.approx(0.10 * 1_000_000 + 0.10 * 1_000_000 + 0.10 * 3_000_000)}


# ------------------------------------------------- pacing schedule and expected return
SEED = date(2028, 12, 31)  # the first commitment: the pacing model's liquid value is 1 on this day
SCHEDULE = {"BUYOUT": {2027: 0.0, 2028: 0.02, 2029: 0.021, 2030: 0.02205}}  # 2% growing 5% a year


def test_a_schedule_is_divided_by_the_expected_value_to_become_a_share_of_the_liquid_value():
    p1, p2, p3 = fund("P1", closing="2028-12-31"), fund("P2", closing="2029-12-31"), fund("P3", closing="2030-12-31")
    policy = AnnualRatePolicy(SCHEDULE, [p1, p2, p3], expected_return=0.05)
    assert policy.expected_value(date(2028, 12, 31), SEED) == 1.0  # seeded on the first commitment date
    assert policy.expected_value(date(2029, 12, 31), SEED) == pytest.approx(1.05)
    assert policy.expected_value(date(2030, 12, 31), SEED) == pytest.approx(1.1025)
    assert policy.expected_value(date(2029, 6, 30), SEED) == pytest.approx(1.05 ** (181 / 365))  # between year ends: elapsed time
    # a schedule that grows at X is a constant 2% of the liquid value, whatever the actual value does
    assert policy.size_commitments([p1], balances(date(2028, 12, 31), 1_100_000.0, first_commitment_date=SEED)) == {"P1": pytest.approx(22_000)}
    assert policy.size_commitments([p2], balances(date(2029, 12, 31), 1_100_000.0, first_commitment_date=SEED)) == {"P2": pytest.approx(22_000)}
    assert policy.size_commitments([p3], balances(date(2030, 12, 31), 1_331_000.0, first_commitment_date=SEED)) == {"P3": pytest.approx(26_620)}
    explained = policy.explain_commitment("P3", balances(date(2030, 12, 31), 1_331_000.0, first_commitment_date=SEED))
    assert explained["expected_value"] == pytest.approx(1.1025) and explained["own_year_rate"] == 0.02205


def test_the_planned_amount_is_scaled_by_actual_over_expected_value():
    p2 = fund("P2", closing="2029-12-31")
    policy = AnnualRatePolicy(SCHEDULE, [p2], expected_return=0.05)
    on_plan = balances(date(2029, 12, 31), 1_000_000 * 1.05, first_commitment_date=SEED)       # worth 1,000,000 at the seed
    ahead = balances(date(2029, 12, 31), 1_000_000 * 1.05 * 1.2, first_commitment_date=SEED)   # 20% ahead of plan
    assert policy.size_commitments([p2], on_plan)["P2"] == pytest.approx(0.021 * 1_000_000)      # exactly the planned amount
    assert policy.size_commitments([p2], ahead)["P2"] == pytest.approx(0.021 * 1_000_000 * 1.2)  # scaled by actual / expected


def test_carried_years_use_their_own_expected_value():
    late = fund("L", closing="2030-12-31")
    policy = AnnualRatePolicy({"BUYOUT": {2028: 0.02, 2029: 0.021, 2030: 0.02205}}, [late], carry_forward=True, expected_return=0.05)
    year_ends = {2028: YearEndBalance(date(2028, 12, 31), 1_100_000.0), 2029: YearEndBalance(date(2029, 12, 31), 1_100_000.0)}
    at_closing = balances(date(2030, 12, 31), 1_331_000.0, year_ends, first_commitment_date=SEED)
    # 2028: 2.0% / 1 × 1,100,000 · 2029: 2.1% / 1.05 × 1,100,000 · 2030: 2.205% / 1.1025 × 1,331,000
    assert policy.size_commitments([late], at_closing) == {"L": pytest.approx(22_000 + 22_000 + 26_620)}
    explained = policy.explain_commitment("L", at_closing)
    assert explained["other_years_usd"] == pytest.approx(44_000) and explained["drawn_years"] == "2028, 2029, 2030"


def test_a_year_sized_before_the_first_commitment_uses_an_expected_value_below_one():
    late = fund("L", closing="2029-12-31")
    policy = AnnualRatePolicy({"BUYOUT": {2028: 0.02, 2029: 0.021}}, [late], carry_forward=True, expected_return=0.05)
    seed = date(2029, 12, 31)  # this fund is the run's first commitment, a year after the schedule's first budget
    at_closing = balances(seed, 1_000_000.0, {2028: YearEndBalance(date(2028, 12, 31), 1_000_000.0)}, first_commitment_date=seed)
    assert policy.size_commitments([late], at_closing) == {"L": pytest.approx(0.02 * 1.05 * 1_000_000 + 0.021 * 1_000_000)}


@pytest.mark.parametrize("bad, message", [(5, "write 5% as 0.05"), (1.0, "write 5% as 0.05"), (-1.0, "between -1 and 1"),
                                          (float("nan"), "must be a number"), (True, "must be a number"), ("0.05", "must be a number")])
def test_expected_return_must_be_a_decimal(bad, message):
    with pytest.raises(ValueError, match=message):
        AnnualRatePolicy(SCHEDULE, [], expected_return=bad)


def test_expected_return_needs_the_first_commitment_date():
    p1 = fund("P1", closing="2028-12-31")
    policy = AnnualRatePolicy(SCHEDULE, [p1], expected_return=0.05)
    with pytest.raises(ValueError, match="first_commitment_date"):
        policy.size_commitments([p1], balances(date(2028, 12, 31), 1_000_000.0))


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


# ------------------------------------------------------------- draw plans
# A schedule that grows at exactly X, so each year's rate over its own expected value is a
# constant 2% share. Anything that departs from 2% a year has normalised the wrong date.
DRAW_SCHEDULE = {"BUYOUT": {2028: 0.02, 2029: 0.021, 2030: 0.02205}}


def drawing(plan, closing="2028-12-31", **kwargs):
    b = fund("B", closing=closing)
    return AnnualRatePolicy(DRAW_SCHEDULE, [b], expected_return=0.05, draws={"B": plan}, **kwargs), b


def test_a_plan_replaces_the_years_a_fund_would_otherwise_draw():
    policy, b = drawing({2028: 1.0})
    assert policy.entitlements["B"] == Entitlement(2028, "BUYOUT", 1.0, {2028: 1.0})
    at_closing = balances(date(2028, 12, 31), 1_000_000.0, first_commitment_date=SEED)
    assert policy.size_commitments([b], at_closing) == {"B": pytest.approx(20_000.0)}  # 2% of the liquid value


def test_a_forward_year_is_funded_at_the_closing_but_normalised_on_its_own_date():
    """Convention b: each drawn year keeps the share the schedule meant it to have."""
    policy, b = drawing({2028: 1.0, 2029: 1.0, 2030: 1.0})
    at_closing = balances(date(2028, 12, 31), 1_000_000.0, first_commitment_date=SEED)
    # three years, each 2% of the liquid value on the closing day: 2.1%/1.05 and 2.205%/1.1025 are both 2%
    assert policy.size_commitments([b], at_closing) == {"B": pytest.approx(60_000.0)}
    # normalising every year on the closing date instead would have committed the later years' larger
    # rates against today's smaller portfolio: 5% too much for 2029, 10.25% too much for 2030
    assert policy.size_commitments([b], at_closing)["B"] != pytest.approx((0.02 + 0.021 + 0.02205) * 1_000_000)

    drawn = {d.year: d for d in policy.drawn_years("B", at_closing)}
    assert [d.plan_date.year for d in drawn.values()] == [2028, 2029, 2030]  # each year dated by itself
    assert {d.funding_date for d in drawn.values()} == {date(2028, 12, 31)}  # all funded at the closing
    assert drawn[2030].expected_value == pytest.approx(1.1025) and drawn[2030].rate == 0.02205


def test_a_multiplier_draws_one_year_several_times():
    single, b = drawing({2029: 1.0}, closing="2029-12-31")
    tripled, _ = drawing({2029: 3.0}, closing="2029-12-31")
    at_closing = balances(date(2029, 12, 31), 1_000_000.0, first_commitment_date=SEED)
    assert single.size_commitments([b], at_closing) == {"B": pytest.approx(20_000.0)}
    assert tripled.size_commitments([b], at_closing) == {"B": pytest.approx(60_000.0)}
    assert tripled.entitlements["B"].draws == {2029: 3.0}
    assert [d.multiplier for d in tripled.drawn_years("B", at_closing)] == [3.0]


def test_a_forward_year_never_touches_a_balance_from_its_own_year():
    """The engine hides unfinished years; a policy handed one anyway must still not use it."""
    policy, b = drawing({2028: 1.0, 2030: 1.0})
    without = balances(date(2028, 12, 31), 1_000_000.0, first_commitment_date=SEED)
    leaked = balances(date(2028, 12, 31), 1_000_000.0,
                     {2030: YearEndBalance(date(2030, 12, 31), 9_999_999.0)}, first_commitment_date=SEED)
    assert policy.size_commitments([b], leaked) == policy.size_commitments([b], without)
    assert all(d.funding_date == date(2028, 12, 31) for d in policy.drawn_years("B", leaked))


def test_a_plan_naming_the_carried_years_is_carry_forward():
    late = fund("L", closing="2030-12-31")
    rates = {"BUYOUT": {2028: 0.02, 2029: 0.021, 2030: 0.02205}}
    carried = AnnualRatePolicy(rates, [late], carry_forward=True, expected_return=0.05)
    planned = AnnualRatePolicy(rates, [late], expected_return=0.05, draws={"L": {2028: 1, 2029: 1, 2030: 1}})
    year_ends = {2028: YearEndBalance(date(2028, 12, 31), 1_100_000.0), 2029: YearEndBalance(date(2029, 12, 31), 1_100_000.0)}
    at_closing = balances(date(2030, 12, 31), 1_331_000.0, year_ends, first_commitment_date=SEED)
    assert planned.size_commitments([late], at_closing) == carried.size_commitments([late], at_closing)
    assert planned.entitlements["L"].draws == carried.entitlements["L"].draws


def test_a_drawn_year_before_the_run_began_contributes_nothing():
    late = fund("L", closing="2030-12-31")
    policy = AnnualRatePolicy({"BUYOUT": {2027: 0.9, 2028: 0.02, 2029: 0.021, 2030: 0.02205}}, [late],
                              expected_return=0.05, draws={"L": {2027: 1, 2029: 1, 2030: 1}})
    at_closing = balances(date(2030, 12, 31), 1_331_000.0, {2029: YearEndBalance(date(2029, 12, 31), 1_100_000.0)},
                          first_commitment_date=SEED)
    # 2027 has no year end to size on and is skipped; 2029 and 2030 are 2% each of their own funding values
    assert policy.size_commitments([late], at_closing) == {"L": pytest.approx(22_000 + 26_620)}
    assert [d.year for d in policy.drawn_years("L", at_closing)] == [2029, 2030]


def test_a_plan_switches_carry_forward_off_for_its_own_fund_type_only():
    s, b = fund("S", "SECONDARIES", "2029-03-01"), fund("B", "BUYOUT", "2029-03-01")
    rates = {"BUYOUT": {2027: 0.1, 2028: 0.1, 2029: 0.1}, "SECONDARIES": {2027: 0.05, 2028: 0.05, 2029: 0.05}}
    policy = AnnualRatePolicy(rates, [s, b], carry_forward=True, draws={"S": {2029: 2.0}})
    assert policy.entitlements["S"].draws == {2029: 2.0}                        # the plan, not 2027 and 2028
    assert policy.entitlements["B"].draws == {2027: 1.0, 2028: 1.0, 2029: 1.0}  # buyout still carries


def test_unclaimed_schedule_years_names_the_budget_nobody_draws():
    s = fund("S", "SECONDARIES", "2029-03-01")
    rates = {"SECONDARIES": {2027: 0.05, 2028: 0.0, 2029: 0.05, 2030: 0.05}}
    policy = AnnualRatePolicy(rates, [s], draws={"S": {2027: 1.0, 2029: 1.0}})
    assert policy.unclaimed_schedule_years() == {"SECONDARIES": [2030]}  # 2028's rate is zero: nothing to spend


@pytest.mark.parametrize("plan, message", [
    ({"Z": {2028: 1.0}}, "not in the fund list"),
    ({"B": {}}, "is empty: leave the fund out"),
    ({"B": {2041: 1.0}}, "year 2041, which commitment_rates has no row for"),
    ({"B": {2028: -1.0}}, "finite non-negative"),
    ({"B": {2028: float("inf")}}, "finite non-negative"),
])
def test_draw_plan_validation(plan, message):
    with pytest.raises(ValueError, match=message):
        AnnualRatePolicy(DRAW_SCHEDULE, [fund("B", closing="2028-12-31")], draws=plan)


def test_a_schedule_year_cannot_be_drawn_by_two_different_closings():
    early, late = fund("E", closing="2028-12-31"), fund("L", closing="2029-12-31")
    with pytest.raises(ValueError, match="year 2028 is drawn by funds closing in both 2028 and 2029"):
        AnnualRatePolicy(DRAW_SCHEDULE, [early, late], draws={"E": {2028: 1.0}, "L": {2028: 1.0, 2029: 1.0}})


def test_funds_closing_together_may_draw_the_same_years_and_split_them():
    a, b = fund("A", closing="2028-12-31"), fund("B", closing="2028-12-31")
    plan = {2028: 1.0, 2029: 1.0}
    policy = AnnualRatePolicy(DRAW_SCHEDULE, [a, b], weights={"A": 0.75, "B": 0.25},
                              expected_return=0.05, draws={"A": plan, "B": plan})
    at_closing = balances(date(2028, 12, 31), 1_000_000.0, first_commitment_date=SEED)
    # 4% of the liquid value between them, split by weight
    assert policy.size_commitments([a, b], at_closing) == {"A": pytest.approx(30_000.0), "B": pytest.approx(10_000.0)}


# ---------------------------------------------------------------- rounding
@pytest.mark.parametrize("value, unit, expected", [
    (2_344_999.99, 10_000, 2_340_000.0),   # ROUND(value, -4)
    (2_345_000.00, 10_000, 2_350_000.0),   # a half goes away from zero ...
    (25_000, 10_000, 30_000.0),            # ... where Python's round() would give 20,000
    (5_000, 10_000, 10_000.0),             # ... and 0
    (4_999.99, 10_000, 0.0),
    (-25_000, 10_000, -30_000.0),          # away from zero on both sides
    (2.675, 0.01, 2.68),                   # ROUND(value, 2): 2.675 is a hair below the half in binary
    (1.005, 0.01, 1.01),
    (0.125, 0.01, 0.13),
    (12_345.678, 100, 12_300.0),           # ROUND(value, -2)
    (70_000, 10_000, 70_000.0),            # a multiple stays where it is
])
def test_rounding_follows_excels_round(value, unit, expected):
    assert round_like_excel(value, unit) == expected


def test_the_rounding_unit_keeps_the_relative_precision_of_round_minus_four_on_a_hundred_million():
    assert commitment_rounding_unit(100_000_000) == 10_000.0
    assert commitment_rounding_unit(1_000_000) == 100.0
    assert commitment_rounding_unit(100) == 0.01
    assert commitment_rounding_unit(250_000_000) == 25_000.0  # proportional, not only powers of ten
    for bad in (0, -5, float("nan"), float("inf"), True, "100"):
        with pytest.raises(ValueError, match="starting_value must be a positive number"):
            commitment_rounding_unit(bad)


def test_each_drawn_year_is_rounded_before_its_multiplier_and_weight():
    a, b = fund("A", closing="2028-12-31"), fund("B", closing="2028-12-31")
    plan = {2028: 3.0}
    policy = AnnualRatePolicy({"BUYOUT": {2028: 0.02}}, [a, b], weights={"A": 0.75, "B": 0.25},
                              draws={"A": plan, "B": plan}, rounding_unit_usd=10_000)
    at_closing = balances(date(2028, 12, 31), 1_234_567.0)
    # the year's commitment is 24,691.34 → 20,000; three of those is 60,000; then the split by weight
    assert policy.size_commitments([a, b], at_closing) == {"A": 45_000.0, "B": 15_000.0}
    drawn = policy.drawn_years("A", at_closing)[0]
    assert drawn.year_budget_unrounded_usd == pytest.approx(24_691.34) and drawn.year_budget_usd == 20_000.0
    assert drawn.commitment_usd == 60_000.0
    # rounding the tripled amount instead would have given 70,000
    assert round_like_excel(3 * drawn.year_budget_unrounded_usd, 10_000) == 70_000.0


def test_carried_years_are_rounded_one_year_at_a_time():
    late = fund("L", closing="2029-03-01")
    policy = AnnualRatePolicy(RATES, [late], carry_forward=True, rounding_unit_usd=30_000)
    # 2027: 100,000 → 90,000 · 2028: 100,000 → 90,000 · 2029: 150,000 → 150,000
    assert policy.size_commitments([late], AT_C) == {"L": 330_000.0}
    explained = policy.explain_commitment("L", AT_C)
    assert explained["own_year_usd"] == 150_000.0 and explained["other_years_usd"] == 180_000.0


def test_without_a_rounding_unit_nothing_is_rounded():
    a = fund("A")
    policy = AnnualRatePolicy({"BUYOUT": {2027: 0.1}}, [a])
    drawn = policy.drawn_years("A", balances(date(2027, 3, 31), 1_234_567.89))[0]
    assert policy.rounding_unit_usd is None
    assert drawn.year_budget_usd == drawn.year_budget_unrounded_usd == pytest.approx(123_456.789)


@pytest.mark.parametrize("unit", [0, -10_000, float("nan"), float("inf"), True, "10000"])
def test_a_rounding_unit_must_be_a_positive_number_of_dollars(unit):
    with pytest.raises(ValueError, match="rounding_unit_usd must be a positive number of dollars"):
        AnnualRatePolicy({"BUYOUT": {2027: 0.1}}, [fund("A")], rounding_unit_usd=unit)
