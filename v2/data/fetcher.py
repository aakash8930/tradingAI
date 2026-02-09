import ccxt
import pandas as pd
from typing import Optional

from features.technicals import compute_core_features


class MarketDataFetcher:
    """
    Responsible for:
    - Fetching OHLCV data
    - Building training datasets
    """

    def __init__(self, exchange_name: str = "binance"):
        if exchange_name == "binance":
            self.exchange = ccxt.binance({"enableRateLimit": True})
        else:
            raise ValueError(f"Unsupported exchange: {exchange_name}")

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 5000,
    ) -> pd.DataFrame:
        bars = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(
            bars,
            columns=["time", "open", "high", "low", "close", "volume"],
        )
        return df

    def build_training_dataset(
        self,
        symbol: str,
        timeframe: str,
        horizon: int = 3,
        threshold: float = 0.002,
        limit: int = 5000,
    ) -> pd.DataFrame:
        """
        Builds labeled dataset for AI training.

        target = 1 if future return > threshold
        """

        df = self.fetch_ohlcv(symbol, timeframe, limit)
        df = compute_core_features(df)

        df["future_close"] = df["close"].shift(-horizon)
        df["future_return"] = (df["future_close"] - df["close"]) / df["close"]
        df["target"] = (df["future_return"] > threshold).astype(int)

        df.dropna(inplace=True)

        return df
