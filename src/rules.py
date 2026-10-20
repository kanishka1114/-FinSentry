# Transparent illustrative baseline, not a production fraud policy.
RISKY_TYPES = ("TRANSFER", "CASH_OUT")
LARGE_AMOUNT = 200_000
VERY_LARGE_AMOUNT = 1_000_000
FLAG_AT_POINTS = 2


def rule_points(df):
    """One point each for: risky type, amount >= 200k, amount >= 1M."""
    amount = df["amount"].astype(float)
    points = df["type"].astype(str).isin(RISKY_TYPES).astype(int)
    points += (amount >= LARGE_AMOUNT).astype(int)
    points += (amount >= VERY_LARGE_AMOUNT).astype(int)
    return points


def rule_based_score(df):
    """Flag = 2+ points: a risky type of at least 200k, or any type of at least 1M."""
    return (rule_points(df) >= FLAG_AT_POINTS).astype(int)


RULE_DESCRIPTION = (f"Flag if 2+ points from: type in {list(RISKY_TYPES)}; amount >= "
                    f"{LARGE_AMOUNT:,}; amount >= {VERY_LARGE_AMOUNT:,}.")
