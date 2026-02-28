# backtest/simulator.py

import pandas as pd
import ta

from execution.broker import PaperBroker
from models.direction import DirectionModel
from risk.limits import RiskLimits, RiskState
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

        self.risk_limits = RiskLimits(
            max_daily_loss_pct=1.0,
            max_weekly_loss_pct=1.0,
            max_trades_per_day=100_000,
        )

        self.risk_state = RiskState(starting_balance)
        self.broker = PaperBroker()

        self.lookback = lookback
        self.risk_per_trade = 0.01

        self.trades: list[dict] = []

        self.last_entry_price: float | None = None
        self.last_entry_prob: float | None = None

        self.trailing_pct = 0.0075
        self.trailing_stop: float | None = None
        self.highest_price: float | None = None
        self.lowest_price: float | None = None

    def step(self, df: pd.DataFrame) -> None:
        current_date = pd.to_datetime(df.iloc[-1]["time"], unit="ms").date()
        self.risk_state.reset_if_new_day(current_date)

        price = df.iloc[-1]["close"]
        prob_up = self.model.predict_proba(df)

        df = df.copy()
        df["ema200"] = ta.trend.EMAIndicator(df["close"], 200).ema_indicator()
        df["atr"] = ta.volatility.AverageTrueRange(
            df["high"], df["low"], df["close"], 14
        ).average_true_range()

        ema200 = df.iloc[-1]["ema200"]
        atr_pct = df.iloc[-1]["atr"] / price

        # ================= EXIT =================
        if self.broker.position:
            side = self.broker.position.side

            if side == "LONG":
                self.highest_price = max(self.highest_price, price)
                self.trailing_stop = max(
                    self.trailing_stop,
                    self.highest_price * (1 - self.trailing_pct),
                )
                hit_stop = price <= self.trailing_stop
            else:
                self.lowest_price = min(self.lowest_price, price)
                self.trailing_stop = min(
                    self.trailing_stop,
                    self.lowest_price * (1 + self.trailing_pct),
                )
                hit_stop = price >= self.trailing_stop

            if hit_stop:
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

                self.trailing_stop = None
                self.highest_price = None
                self.lowest_price = None
                return

        # ================= ENTRY =================
        if self.broker.position is None:
            if not self.risk_state.trading_allowed(self.risk_limits):
                return

            side = None
            if prob_up >= 0.40 and price > ema200 and atr_pct > 0.0015:
                side = "LONG"
            elif prob_up <= 0.25 and price < ema200 and atr_pct > 0.0015:
                side = "SHORT"

            if not side:
                return

            stop_price = price * (0.99 if side == "LONG" else 1.01)
            qty = fixed_fractional_size(
                balance=self.risk_state.current_balance,
                risk_pct=self.risk_per_trade,
                entry_price=price,
                stop_price=stop_price,
            )

            if qty <= 0:
                return

            self.broker.open_position(side, price, qty)
            self.last_entry_price = price
            self.last_entry_prob = prob_up

            if side == "LONG":
                self.highest_price = price
            else:
                self.lowest_price = price

            self.trailing_stop = stop_price

    def export(self, path: str) -> None:
        pd.DataFrame(self.trades).to_csv(path, index=False)

