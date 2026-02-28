# execution/regime_controller.py

from enum import Enum
import pandas as pd


class MarketRegime(str, Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    CHOPPY = "choppy"


class RegimeController:
    """
    Detects market regime and provides strategy modifiers.
    """

    def detect(self, df: pd.DataFrame) -> MarketRegime:
        adx = df.iloc[-1]["adx"]
        atr_pct = df.iloc[-1]["atr_pct"]

        if adx >= 25 and atr_pct >= 0.002:
            return MarketRegime.TRENDING

        if adx < 15:
            return MarketRegime.RANGING

        return MarketRegime.CHOPPY

    def risk_multiplier(self, regime: MarketRegime) -> float:
        if regime == MarketRegime.TRENDING:
            return 1.25
        if regime == MarketRegime.RANGING:
            return 0.75
        return 0.50

    def trading_allowed(self, regime: MarketRegime) -> bool:
        if regime == MarketRegime.CHOPPY:
            return False
        return True