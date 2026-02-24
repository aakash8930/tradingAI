# backtest/simulator.py

import pandas as pd
import ta

from models.direction import DirectionModel
from risk.limits import RiskLimits, RiskState
from execution.broker import PaperBroker
from risk.sizing import fixed_fractional_size


class HistoricalSimulator:
    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        starting_balance: float = 500.0,
        lookback: int = 300,
    ):
        self.model = DirectionModel(model_path, scaler_path)

        # Disable limits for backtest
        self.risk_limits = RiskLimits(
            max_daily_loss_pct=1.0,
            max_weekly_loss_pct=1.0,
            max_trades_per_day=100000
        )

        self.risk_state = RiskState(starting_balance=starting_balance)
        self.broker = PaperBroker()

        self.lookback = lookback
        self.risk_per_trade = 0.01  # 1% risk

        self.trades = []

        self.last_entry_price = None
        self.last_entry_prob = None

        # ---- Trailing Stop State ----
        self.trailing_pct = 0.0075  # 0.75%
        self.trailing_stop = None
        self.highest_price = None
        self.lowest_price = None

    def step(self, window_df: pd.DataFrame):

        # Reset daily risk using candle time
        current_time = pd.to_datetime(window_df.iloc[-1]["time"], unit="ms")
        self.risk_state.reset_if_new_day(current_time.date())

        price = window_df.iloc[-1]["close"]
        prob_up = self.model.predict_proba(window_df)

        # ---- Indicators ----
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

        # =========================
        # EXIT LOGIC
        # =========================
        if self.broker.position:

            side = self.broker.position.side

            if side == "LONG":

                # update highest price
                self.highest_price = max(self.highest_price, price)

                new_trail = self.highest_price * (1 - self.trailing_pct)
                self.trailing_stop = max(self.trailing_stop, new_trail)

                hit_stop = price <= self.trailing_stop

            else:  # SHORT

                self.lowest_price = min(self.lowest_price, price)

                new_trail = self.lowest_price * (1 + self.trailing_pct)
                self.trailing_stop = min(self.trailing_stop, new_trail)

                hit_stop = price >= self.trailing_stop

            if hit_stop:
                # Apply conservative slippage
                exit_price = price * 0.9995
                pnl = self.broker.close_position(exit_price)
                self.risk_state.register_trade(pnl)

                self.trades.append({
                    "side": side,
                    "entry_price": self.last_entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "balance": self.risk_state.current_balance,
                    "prob_up_entry": self.last_entry_prob,
                })

                # reset trailing state
                self.trailing_stop = None
                self.highest_price = None
                self.lowest_price = None

                return  # exit takes priority

        # =========================
        # ENTRY LOGIC
        # =========================
        if self.broker.position is None:

            if not self.risk_state.trading_allowed(self.risk_limits):
                return

            side = None

            # Stronger AI filter
            if prob_up >= 0.40 and price > ema200 and atr_pct > 0.0015:
                side = "LONG"

            elif prob_up <= 0.25 and price < ema200 and atr_pct > 0.0015:
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

                    # Initialize trailing logic
                    if side == "LONG":
                        self.highest_price = price
                        self.trailing_stop = stop_price
                    else:
                        self.lowest_price = price
                        self.trailing_stop = stop_price

    def export(self, path: str):
        pd.DataFrame(self.trades).to_csv(path, index=False)
