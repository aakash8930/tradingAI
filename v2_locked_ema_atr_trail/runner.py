import time
import ta

from v2.data.fetcher import MarketDataFetcher
from v2.models.direction import DirectionModel
from v2.risk.limits import RiskLimits, RiskState
from v2.execution.broker import PaperBroker
from v2.risk.sizing import fixed_fractional_size
from v2.logs.logger import TradeLogger


class TradingRunner:

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
        self.logger = TradeLogger()

        self.risk_per_trade = 0.005
        self.trailing_pct = 0.0075

        self.trailing_stop = None
        self.highest_price = None
        self.lowest_price = None

        self.last_entry_price = None
        self.last_entry_prob = None


    def run_once(self):

        self.risk_state.reset_if_new_day()

        df = self.data.fetch_ohlcv(
            self.symbol,
            self.timeframe,
            limit=self.lookback,
        )

        price = df.iloc[-1]["close"]
        prob_up = self.model.predict_proba(df)

        # Indicators
        df["ema200"] = ta.trend.EMAIndicator(
            df["close"], window=200
        ).ema_indicator()

        df["atr"] = ta.volatility.AverageTrueRange(
            high=df["high"],
            low=df["low"],
            close=df["close"],
            window=14,
        ).average_true_range()

        ema200 = df.iloc[-1]["ema200"]
        atr = df.iloc[-1]["atr"]
        atr_pct = atr / price


        # EXIT
        if self.broker.position:

            side = self.broker.position.side
            entry = self.last_entry_price

            hard_stop = entry * (0.99 if side == "LONG" else 1.01)

            if side == "LONG":
                self.highest_price = max(self.highest_price, price)
                new_trail = self.highest_price * (1 - self.trailing_pct)
                self.trailing_stop = max(self.trailing_stop, new_trail)
                hit_trail = price <= self.trailing_stop
                hit_hard = price <= hard_stop

            else:
                self.lowest_price = min(self.lowest_price, price)
                new_trail = self.lowest_price * (1 + self.trailing_pct)
                self.trailing_stop = min(self.trailing_stop, new_trail)
                hit_trail = price >= self.trailing_stop
                hit_hard = price >= hard_stop

            if hit_trail or hit_hard:

                pnl = self.broker.close_position(price)
                self.risk_state.register_trade(pnl)

                self.logger.log(
                    side=side,
                    entry_price=entry,
                    exit_price=price,
                    qty=self.broker.position.qty,
                    pnl=pnl,
                    balance=self.risk_state.current_balance,
                    prob_up=self.last_entry_prob,
                )

                self.trailing_stop = None
                self.highest_price = None
                self.lowest_price = None

                print(f"EXIT | PnL={pnl:.2f}")
                return


        # ENTRY
        if self.broker.position is None:

            if not self.risk_state.trading_allowed(self.risk_limits):
                print("Risk halt")
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

                    self.trailing_stop = stop_price
                    self.highest_price = price
                    self.lowest_price = price

                    print(f"ENTER {side} @ {price:.2f}")


        print(
            f"{self.symbol} | price={price:.2f} | AI_prob_up={prob_up:.3f} | Bal={self.risk_state.current_balance:.2f}"
        )


    def run_loop(self, sleep_seconds: int = 300):

        print("🚀 v2 Trading Runner started")

        while True:
            try:
                self.run_once()
                time.sleep(sleep_seconds)

            except KeyboardInterrupt:
                print("Stopped")
                break

            except Exception as e:
                print("Runner error:", e)
                time.sleep(30)
