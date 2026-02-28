# execution/strategy.py

import pandas as pd

from models.direction import DirectionModel
from risk.sizing import fixed_fractional_size


class StrategyEngine:
    """
    StrategyEngine responsibilities:
    - Entry signal generation
    - Dynamic thresholding based on model quality
    - Position sizing

    ❌ Does NOT manage exits or trailing stops
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

        self.last_entry_price = None
        self.last_entry_prob = None

        # --- Model quality awareness ---
        metrics = getattr(model, "metadata", {}).get("metrics", {})
        self.model_f1 = float(metrics.get("val_f1", 0.0))

        # Dynamic base threshold (THIS IS THE EDGE)
        if self.model_f1 >= 0.30:
            self.base_long_th = 0.50
        elif self.model_f1 >= 0.20:
            self.base_long_th = 0.52
        elif self.model_f1 >= 0.10:
            self.base_long_th = 0.55
        else:
            self.base_long_th = 0.58

    # ==================================================
    # SIGNAL GENERATION
    # ==================================================
    def generate_signal(self, df: pd.DataFrame):
        price = df.iloc[-1]["close"]
        ema200 = df.iloc[-1]["ema200"]
        atr = df.iloc[-1]["atr"]
        adx = df.iloc[-1]["adx"]

        prob_up = self.model.predict_proba(df)

        # --- Broken / OOD model protection
        if prob_up < 0.05 or prob_up > 0.95:
            return None, prob_up

        atr_pct = atr / price

        # ------------------------------
        # Dynamic threshold logic
        # ------------------------------
        long_th = self.base_long_th

        # Strong trend bonus
        if adx >= 25:
            long_th -= 0.015

        # EMA alignment
        if price < ema200:
            long_th += 0.02

        # Volatility gating
        if atr_pct < 0.001:
            return None, prob_up

        # Safety clamp
        long_th = max(0.45, min(long_th, 0.62))

        # ------------------------------
        # LONG ENTRY
        # ------------------------------
        if prob_up >= long_th and adx >= self.min_adx:
            self.last_entry_price = price
            self.last_entry_prob = prob_up
            return "LONG", prob_up

        # ------------------------------
        # DEBUG
        # ------------------------------
        print(
            f"DEBUG | prob={prob_up:.3f} | "
            f"f1={self.model_f1:.2f} | "
            f"adx={adx:.1f} | "
            f"atr_pct={atr_pct:.4f} | "
            f"long_th={long_th:.3f}"
        )

        return None, prob_up

    # ==================================================
    # POSITION SIZING
    # ==================================================
    def position_size(
        self,
        balance: float,
        entry_price: float,
        side: str,
        max_position_notional_pct: float = 1.0,
    ) -> float:
        stop_price = entry_price * (0.99 if side == "LONG" else 1.01)

        return fixed_fractional_size(
            balance=balance,
            risk_pct=self.risk_per_trade,
            entry_price=entry_price,
            stop_price=stop_price,
            max_position_notional_pct=max_position_notional_pct,
        )

    # ==================================================
    # SYMBOL SCORING (used by auto coin selector)
    # ==================================================
    def score_symbol(self, df: pd.DataFrame) -> float:
        prob_up = self.model.predict_proba(df)
        atr_pct = df.iloc[-1]["atr"] / df.iloc[-1]["close"]
        adx = df.iloc[-1]["adx"]

        quality_boost = max(0.5, self.model_f1 * 2.5)
        return float(prob_up * atr_pct * adx * quality_boost)
