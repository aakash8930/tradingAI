import time
from datetime import datetime

from v2.data.fetcher import MarketDataFetcher
from v2.models.direction import DirectionModel

from v2.risk.limits import RiskLimits, RiskState


class TradingRunner:
    """
    v2 execution runner (read-only, no trading yet)

    Responsibilities:
    - Fetch latest candles
    - Run AI inference
    - Emit signals (print only)
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

        self.risk_limits = RiskLimits()
        self.risk_state = RiskState(starting_balance=1000.0)

    def run_once(self):
        self.risk_state.reset_if_new_day()

        if not self.risk_state.trading_allowed(self.risk_limits):
           print("⛔ Trading halted by risk engine")
           return

        df = self.data.fetch_ohlcv(
            self.symbol,
            self.timeframe,
            limit=self.lookback,
        )

        prob_up = self.model.predict_proba(df)

        ts = datetime.now().strftime("%H:%M:%S")
        price = df.iloc[-1]["close"]

        print(
            f"{ts} | {self.symbol} | price={price:.2f} | AI_prob_up={prob_up:.3f} | risk=OK"
        )

    def run_loop(self, sleep_seconds: int = 300):
        print("🚀 v2 Trading Runner started (signal-only)")

        while True:
            try:
                self.run_once()
                time.sleep(sleep_seconds)

            except KeyboardInterrupt:
                print("\n🛑 Runner stopped by user")
                break

            except Exception as e:
                print("⚠️ Runner error:", e)
                time.sleep(30)
