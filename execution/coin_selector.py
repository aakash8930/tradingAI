
import ta
from data.fetcher import MarketDataFetcher


class CoinSelector:
    def __init__(
        self,
        timeframe: str = "15m",
        lookback: int = 200,
        top_k: int = 5,
        min_atr_pct: float = 0.00,
        min_volume_ratio: float = 1.0,
    ):
        self.timeframe = timeframe
        self.lookback = lookback
        self.top_k = top_k
        self.min_atr_pct = min_atr_pct
        self.min_volume_ratio = min_volume_ratio
        self.fetcher = MarketDataFetcher()

    def _score_symbol(self, symbol: str) -> float | None:
        try:
            df = self.fetcher.fetch_ohlcv(
                symbol, self.timeframe, limit=self.lookback
            )
            if df is None or len(df) < 100:
                return None

            # Indicators
            atr = ta.volatility.AverageTrueRange(
                df["high"], df["low"], df["close"], window=14
            ).average_true_range()

            adx = ta.trend.ADXIndicator(
                df["high"], df["low"], df["close"], window=14
            ).adx()

            vol_ma = df["volume"].rolling(20).mean()

            atr_pct = atr.iloc[-1] / df["close"].iloc[-1]
            volume_ratio = df["volume"].iloc[-1] / vol_ma.iloc[-1]
            trend_strength = min(adx.iloc[-1], 40.0)

            if atr_pct < self.min_atr_pct:
                return None
            if volume_ratio < self.min_volume_ratio:
                return None

            score = atr_pct * volume_ratio * trend_strength
            return float(score)

        except Exception:
            return None

    def select(self, symbols: list[str]) -> list[str]:
        scores = {}

        for symbol in symbols:
            score = self._score_symbol(symbol)
            if score is not None:
                scores[symbol] = score

        ranked = sorted(scores, key=scores.get, reverse=True)

        # 🔥 SAFETY FALLBACK (VERY IMPORTANT)
        if not ranked:
            print("⚠️ CoinSelector empty → fallback to base symbols")
            return symbols[: self.top_k]

        return ranked[: self.top_k]