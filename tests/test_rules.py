import pandas as pd
from src.rules import rule_based_score, rule_points


def score(kind, amount):
    return int(rule_based_score(pd.DataFrame({"type": [kind], "amount": [amount]})).iloc[0])


def test_rule_flags_large_transfer():
    assert score("TRANSFER", 250_000.0) == 1


def test_rule_does_not_flag_small_payment():
    assert score("PAYMENT", 25.0) == 0


def test_risky_type_needs_large_amount():
    assert score("CASH_OUT", 199_999.0) == 0
    assert score("CASH_OUT", 200_000.0) == 1


def test_other_types_need_very_large_amount():
    assert score("PAYMENT", 500_000.0) == 0
    assert score("PAYMENT", 1_000_000.0) == 1


def test_every_tier_changes_points():
    df = pd.DataFrame({"type": ["PAYMENT", "TRANSFER", "TRANSFER", "TRANSFER"],
                       "amount": [10.0, 10.0, 300_000.0, 2_000_000.0]})
    assert rule_points(df).tolist() == [0, 1, 2, 3]
