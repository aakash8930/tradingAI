import pandas as pd

from models.direction import DirectionModel
from risk.sizing import fixed_fractional_size


class StrategyEngine:
    """
    StrategyEngine responsibilities:
    - Entry signal generation
    - Conservative but practical thresholds for spot trading
    """

    def __init__(
        self,
        model: DirectionModel,
        risk_per_trade: float = 0.01,
        min_adx: float = 8.0,
    ):
        self.model = model
        self.risk_per_trade = risk_per_trade
        self.min_adx = min_adx

        metrics = getattr(model, "metadata", {}).get("metrics", {})
        self.model_f1 = float(metrics.get("val_f1", 0.0))

        # Base threshold (conservative default)
        if self.model_f1 >= 0.30:
            self.base_long_th = 0.52
        elif self.model_f1 >= 0.20:
            self.base_long_th = 0.54
        else:
            self.base_long_th = 0.56

    # ----------------------------------
    def generate_signal(self, df: pd.DataFrame):
        row = df.iloc[-1]

        price = row["close"]
        ema200 = row["ema200"]
        atr_pct = row["atr_pct"]
        adx = row["adx"]

        prob_up = self.model.predict_proba(df)

        # Broken model protection
        if prob_up < 0.05 or prob_up > 0.95:
            return None, prob_up

        # Volatility floor
        if atr_pct < 0.001:
            return None, prob_up

        long_th = self.base_long_th

        # Strong trend bonus (THIS IS THE KEY CHANGE)
        if adx >= 30:
            long_th -= 0.02

        # Below EMA200 → be more strict
        if price < ema200:
            long_th += 0.02

        long_th = max(0.50, min(long_th, 0.60))

        if prob_up >= long_th and adx >= self.min_adx:
            return "LONG", prob_up

        print(
            f"DEBUG | prob={prob_up:.3f} | "
            f"f1={self.model_f1:.2f} | "
            f"adx={adx:.1f} | "
            f"atr_pct={atr_pct:.4f} | "
            f"long_th={long_th:.3f}"
        )

        return None, prob_up

    # ----------------------------------
    def position_size(
        self,
        balance: float,
        entry_price: float,
        side: str,
        max_position_notional_pct: float = 1.0,
    ) -> float:
        stop_price = entry_price * 0.99

        return fixed_fractional_size(
            balance=balance,
            risk_pct=self.risk_per_trade,
            entry_price=entry_price,
            stop_price=stop_price,
            max_position_notional_pct=max_position_notional_pct,
        )