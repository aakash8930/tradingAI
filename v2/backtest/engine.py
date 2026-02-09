import pandas as pd
from typing import List, Dict

from data.fetcher import MarketDataFetcher
from models.direction import DirectionModel


class BacktestEngine:
    """
    v2 backtest engine (signal-level, no execution yet)

    Responsibilities:
    - Replay historical candles
    - Run AI inference per step
    - Store signals for later analysis
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        model_path: str,
        scaler_path: str,
        lookback: int = 300,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback = lookback

        self.data = MarketDataFetcher()
        self.model = DirectionModel(model_path, scaler_path)

    def run(self, limit: int = 2000) -> pd.DataFrame:
        df = self.data.fetch_ohlcv(
            self.symbol,
            self.timeframe,
            limit=limit,
        )

        results: List[Dict] = []

        for i in range(self.lookback, len(df)):
            window = df.iloc[i - self.lookback : i + 1]
            price = window.iloc[-1]["close"]
            ts = window.iloc[-1]["time"]

            prob_up = self.model.predict_proba(window)

            results.append({
                "time": ts,
                "price": price,
                "prob_up": prob_up,
            })

        return pd.DataFrame(results)
