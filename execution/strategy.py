# execution/strategy.py

import ta
import pandas as pd

from models.direction import DirectionModel
from risk.sizing import fixed_fractional_size


class StrategyEngine:
    def __init__(
        self,
        model: DirectionModel,
        risk_per_trade: float = 0.01,
        trailing_pct: float = 0.0075,
        long_prob_threshold: float = 0.52,
        short_prob_threshold: float = 0.48,
        min_adx: float = 8.0,
    ):
        self.model = model
        self.risk_per_trade = risk_per_trade
        self.trailing_pct = trailing_pct
        self.long_prob_threshold = long_prob_threshold
        self.short_prob_threshold = short_prob_threshold
        self.min_adx = min_adx

        self.trailing_stop = None
        self.highest_price = None
        self.lowest_price = None

        self.last_entry_price = None
        self.last_entry_prob = None

    def compute_indicators(self, df: pd.DataFrame):
        df = df.copy()

        df["ema200"] = ta.trend.EMAIndicator(
            df["close"], window=200
        ).ema_indicator()

        df["atr"] = ta.volatility.AverageTrueRange(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=14,
        ).average_true_range()

        df["adx"] = ta.trend.ADXIndicator(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=14,
        ).adx()

        return df

    def generate_signal(self, df: pd.DataFrame):
        price = df.iloc[-1]["close"]
        prob_up = self.model.predict_proba(df)
        # --- AI sanity check ---
        # If model is clearly out-of-distribution (e.g. BTC model on SOL),
        # probabilities collapse to extremes (0.0 or 1.0).
        if prob_up < 0.05 or prob_up > 0.95:
            return None, prob_up

        ema200 = df.iloc[-1]["ema200"]
        atr = df.iloc[-1]["atr"]
        adx = df.iloc[-1]["adx"]
        atr_pct = atr / price

        long_th = self.model.long_threshold
        # short_th = self.model.short_threshold
        
        ema_penalty = 0.02 if price < ema200 else 0.0

        if (
            prob_up >= (long_th + ema_penalty)
            and atr_pct > 0.0015
            and adx >= self.min_adx
        ):
            return "LONG", prob_up
        
        # if prob_up <= short_th and price < ema200 and atr_pct > 0.0015 and adx >= self.min_adx:
        #     return "SHORT", prob_up
        
        print(
            f"DEBUG | {df.index[-1]} | "
            f"prob={prob_up:.3f} | "
            f"ema_ok={price > ema200} | "
            f"atr_ok={atr_pct:.4f} | "
            f"adx={adx:.1f}"
        )

        return None, prob_up

    def position_size(self, balance, entry_price, side, max_position_notional_pct=1.0):
        stop_price = entry_price * (0.99 if side == "LONG" else 1.01)

        return fixed_fractional_size(
            balance=balance,
            risk_pct=self.risk_per_trade,
            entry_price=entry_price,
            stop_price=stop_price,
            max_position_notional_pct=max_position_notional_pct,
        )
    
    def score_symbol(self, df: pd.DataFrame) -> float:
        """
        Higher score = better capital allocation candidate
        """
        prob_up = self.model.predict_proba(df)
        atr_pct = df.iloc[-1]["atr"] / df.iloc[-1]["close"]
        adx = df.iloc[-1]["adx"]

        return float(prob_up * atr_pct * adx)

        
