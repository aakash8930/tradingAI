
#data/fetcher.py

import time
import ccxt
import pandas as pd
from ccxt.base.errors import RequestTimeout, NetworkError


class MarketDataFetcher:
    """
    Shared market data fetcher with retry & timeout safety.
    """

    _exchange = None  # 🔑 singleton

    def __init__(self, exchange_name: str = "binance"):
        if MarketDataFetcher._exchange is None:
            MarketDataFetcher._exchange = self._init_exchange(exchange_name)

        self.exchange = MarketDataFetcher._exchange

    def _init_exchange(self, exchange_name: str):
        if exchange_name != "binance":
            raise ValueError(f"Unsupported exchange: {exchange_name}")

        exchange = ccxt.binance({
            "enableRateLimit": True,
            "timeout": 20000,  # 20s
        })

        # Load markets ONCE
        exchange.load_markets()
        return exchange

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        retries: int = 3,
    ) -> pd.DataFrame:

        for attempt in range(1, retries + 1):
            try:
                bars = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe,
                    limit=limit,
                )

                if not bars:
                    raise RuntimeError("empty OHLCV")

                return pd.DataFrame(
                    bars,
                    columns=["time", "open", "high", "low", "close", "volume"],
                )

            except (RequestTimeout, NetworkError) as e:
                if attempt == retries:
                    raise
                time.sleep(2 * attempt)

        raise RuntimeError("fetch_ohlcv failed after retries")
