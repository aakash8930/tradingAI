def fixed_fractional_size(
    balance: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float,
) -> float:
    """
    Classic fixed-fractional position sizing.

    risk_pct = fraction of balance you are willing to lose
    """

    risk_amount = balance * risk_pct
    per_unit_risk = abs(entry_price - stop_price)

    if per_unit_risk <= 0:
        return 0.0

    qty = risk_amount / per_unit_risk
    return max(0.0, qty)
