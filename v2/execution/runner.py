# runner.py

import time
import ta

from v2.data.fetcher import MarketDataFetcher
from v2.models.direction import DirectionModel
from v2.risk.limits import RiskLimits, RiskState
from v2.execution.broker import PaperBroker
from v2.risk.sizing import fixed_fractional_size
from v2.logs.logger import TradeLogger




class TradingRunner:
    """
    v2 execution runner (read-only, no trading yet)

    Responsibilities:
    - Fetch latest candles
    - Run AI inference
    - Emit signals (print only)
    """

    def export_trades(self, path):
        import pandas as pd
        pd.DataFrame(self.logger.trades).to_csv(path, index=False)


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

        self.broker = PaperBroker()
        self.max_hold = 20

        self.hold_candles = 0
        self.risk_per_trade = 0.005 #(0.5%)

        self.logger = TradeLogger()
        self.last_entry_price = None
        self.last_entry_prob = None


    def run_once(self):
        self.risk_state.reset_if_new_day()

        print(
            f"Trades today: {self.risk_state.trades_today} | "
            f"Daily loss %: {self.risk_state.daily_loss_pct():.2%}"
        )


        df = self.data.fetch_ohlcv(
            self.symbol,
            self.timeframe,
            limit=self.lookback,
        )

        price = df.iloc[-1]["close"]
        prob_up = self.model.predict_proba(df)
        
        df["ema200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()
        df["atr"] = ta.volatility.AverageTrueRange(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=14,
        ).average_true_range()
        
        ema200 = df.iloc[-1]["ema200"]
        atr = df.iloc[-1]["atr"]
        atr_pct = atr / price

        # ---- EXIT LOGIC ----
        if self.broker.position:
            self.hold_candles += 1

            exit_signal = (
                (self.broker.position.side == "LONG" and prob_up < 0.45)
                or (self.broker.position.side == "SHORT" and prob_up > 0.55)
                or self.hold_candles >= self.max_hold
            )

            if exit_signal:
                pnl = self.broker.close_position(price)
                self.risk_state.register_trade(pnl)

                self.logger.log(
                    side=self.broker.position.side if self.broker.position else "UNKNOWN",
                    entry_price=self.last_entry_price,
                    exit_price=price,
                    qty=self.broker.position.qty if self.broker.position else 0,
                    pnl=pnl,
                    balance=self.risk_state.current_balance,
                    prob_up=self.last_entry_prob,
                )

                self.hold_candles = 0

                print(f"❌ EXIT | PnL={pnl:.2f} | Bal={self.risk_state.current_balance:.2f}")
                return


        # ---- ENTRY LOGIC ----
        if self.broker.position is None:
            if not self.risk_state.trading_allowed(self.risk_limits):
                print("⛔ Trading halted by risk engine")
                return

            side = None
            if prob_up >= 0.65 and price > ema200 and atr_pct > 0.003:
                side = "LONG"
            elif prob_up <= 0.35 and price < ema200 and atr_pct > 0.003:
                side = "SHORT"

            if side:
                stop_price = price * (0.99 if side == "LONG" else 1.01)
                qty = fixed_fractional_size(
                    balance=self.risk_state.current_balance,
                    risk_pct=self.risk_per_trade,
                    entry_price=price,
                    stop_price=stop_price,
                )

                if qty > 0:
                    self.broker.open_position(side, price, qty)
                    self.last_entry_price = price
                    self.last_entry_prob = prob_up

                    self.hold_candles = 0

                    print(
                        f"🟢 ENTER {side} | price={price:.2f} | qty={qty:.4f}"
                    )

        print(
            f"{self.symbol} | price={price:.2f} | AI_prob_up={prob_up:.3f} | Bal={self.risk_state.current_balance:.2f}"
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

def export_trades(self, path):
    import pandas as pd
    pd.DataFrame(self.trades).to_csv(path, index=False)
