# backtest/vector_engine.py

import numpy as np
import pandas as pd

from data.fetcher import MarketDataFetcher
from models.direction import DirectionModel
from features.technicals import compute_core_features


class VectorBacktestEngine:
    """
    Ultra-fast vectorized backtest (signal-level).
    No execution latency, no trailing logic.
    Designed for:
      - strategy validation
      - threshold tuning
      - model comparison
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        model_path: str,
        scaler_path: str,
        lookback: int = 300,
        fee_pct: float = 0.0004,  # binance taker
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback = lookback
        self.fee_pct = fee_pct

        self.data = MarketDataFetcher()
        self.model = DirectionModel(model_path, scaler_path)

    def run(self, limit: int = 10_000) -> pd.DataFrame:
        df = self.data.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
        df = compute_core_features(df)

        # ---- AI inference (vectorized loop, unavoidable) ----
        probs = []
        for i in range(self.lookback, len(df)):
            window = df.iloc[i - self.lookback : i + 1]
            probs.append(self.model.predict_proba(window))

        df = df.iloc[self.lookback :].copy()
        df["prob_up"] = probs

        # ---- Signals ----
        long_th = self.model.long_threshold - 0.03

        df["signal"] = 0
        df.loc[
            (df["prob_up"] >= long_th)
            & (df["atr_pct"] > 0.0012)
            & (df["adx"] >= 8),
            "signal",
        ] = 1

        # ---- Returns ----
        df["ret"] = df["close"].pct_change().shift(-1)
        df["strategy_ret"] = df["signal"] * df["ret"]

        # ---- Fees ----
        df["fees"] = df["signal"].abs() * self.fee_pct
        df["net_ret"] = df["strategy_ret"] - df["fees"]

        # ---- Equity ----
        df["equity"] = (1 + df["net_ret"]).cumprod()

        return df
