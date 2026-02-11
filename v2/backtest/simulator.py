# simulator.py

import pandas as pd
import ta


from v2.models.direction import DirectionModel
from v2.risk.limits import RiskLimits, RiskState
from v2.execution.broker import PaperBroker
from v2.risk.sizing import fixed_fractional_size


class HistoricalSimulator:

    def step(self, window_df: pd.DataFrame):

        # Reset daily risk based on candle date
        current_time = pd.to_datetime(window_df.iloc[-1]["time"], unit="ms")
        self.risk_state.reset_if_new_day(current_time.date())

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        starting_balance: float = 1000.0,
        lookback: int = 300,
    ):
        self.model = DirectionModel(model_path, scaler_path)

        self.risk_limits = RiskLimits(
            max_daily_loss_pct=1.0,
            max_weekly_loss_pct=1.0,
            max_trades_per_day=100000
        )

        self.risk_state = RiskState(starting_balance=starting_balance)

        self.broker = PaperBroker()

        self.lookback = lookback
        self.max_hold = 20
        self.hold_candles = 0
        self.risk_per_trade = 0.005 #(0.5%)


        self.trades = []

        self.last_entry_price = None
        self.last_entry_prob = None

    def step(self, window_df: pd.DataFrame):
        price = window_df.iloc[-1]["close"]
        prob_up = self.model.predict_proba(window_df)
        
        window_df["ema200"] = ta.trend.EMAIndicator(
            window_df["close"], window=200
        ).ema_indicator()
        
        window_df["atr"] = ta.volatility.AverageTrueRange(
            high=window_df["high"],
            low=window_df["low"],
            close=window_df["close"],
            window=14,
        ).average_true_range()
        
        ema200 = window_df.iloc[-1]["ema200"]
        atr = window_df.iloc[-1]["atr"]
        atr_pct = atr / price

        if not self.risk_state.trading_allowed(self.risk_limits):
            print("HALTED | trades_today:", self.risk_state.trades_today,
                  "| daily_loss:", self.risk_state.daily_loss_pct(),
                  "| weekly_loss:", self.risk_state.weekly_loss_pct())

        # ---- EXIT ----
        if self.broker.position:
            self.hold_candles += 1

            entry = self.last_entry_price
            stop = entry * (0.99 if self.broker.position.side ==
                            "LONG" else 1.01)
            tp = entry * (1.02 if self.broker.position.side ==
                          "LONG" else 0.98)

            hit_stop = (
                (self.broker.position.side == "LONG" and price <= stop)
                or (self.broker.position.side == "SHORT" and price >= stop)
            )

            hit_tp = (
                (self.broker.position.side == "LONG" and price >= tp)
                or (self.broker.position.side == "SHORT" and price <= tp)
            )

            exit_signal = hit_stop or hit_tp or self.hold_candles >= self.max_hold

            if exit_signal:
                pnl = self.broker.close_position(price)
                self.risk_state.register_trade(pnl)

                self.trades.append({
                    "side": self.broker.position.side if self.broker.position else "UNKNOWN",
                    "entry_price": self.last_entry_price,
                    "exit_price": price,
                    "pnl": pnl,
                    "balance": self.risk_state.current_balance,
                    "prob_up_entry": self.last_entry_prob,
                })

                self.hold_candles = 0
                return


        # ---- ENTRY ----
        if self.broker.position is None:
            if not self.risk_state.trading_allowed(self.risk_limits):
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

    def export(self, path: str):
        pd.DataFrame(self.trades).to_csv(path, index=False)
