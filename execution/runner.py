# execution/runner.py

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

from data.fetcher import MarketDataFetcher
from models.direction import DirectionModel
from risk.limits import RiskLimits, RiskState
from execution.broker import LiveBroker, PaperBroker
from logs.logger import TradeLogger
from execution.strategy import StrategyEngine

USD_TO_INR = 90.0  # reporting only


def _fmt_money(value: float) -> str:
    if abs(value) >= 1:
        return f"{value:.2f}"
    if abs(value) >= 0.01:
        return f"{value:.4f}"
    return f"{value:.6f}"


class TradingRunner:
    """
    Live / paper trading runner.
    Uses StrategyEngine as the SINGLE source of truth.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        lookback: int = 300,
        mode: str = "paper",
        exchange_name: str = "binance",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = True,
        allow_live_trading: bool = False,
        starting_balance_usdt: float = 500.0,
        min_balance_usdt: float = 100.0,
        cooldown_minutes: int = 30,
        risk_per_trade: float = 0.01,
        max_position_notional_pct: float = 0.20,
        long_prob_threshold: float = 0.50,
        short_prob_threshold: float = 0.45,
        min_adx: float = 20.0,
        take_profit_rr: float = 2.0,
        breakeven_rr: float = 1.0,
        balance_allocation_pct: float = 1.0,
        data_retry_attempts: int = 3,
        on_before_entry: Optional[Callable[[str], bool]] = None,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback = lookback
        self.mode = mode.lower()
        self.max_position_notional_pct = max_position_notional_pct
        self.take_profit_rr = take_profit_rr
        self.breakeven_rr = breakeven_rr
        self.balance_allocation_pct = balance_allocation_pct
        self.data_retry_attempts = max(1, data_retry_attempts)
        self.on_before_entry = on_before_entry

        # --- Core components ---
        self.data = MarketDataFetcher()

        if model_path and scaler_path and Path(model_path).exists() and Path(scaler_path).exists():
            self.model = DirectionModel(model_path, scaler_path)
        else:
            self.model = DirectionModel.for_symbol(symbol)

        self.strategy = StrategyEngine(
            self.model,
            risk_per_trade=risk_per_trade,
            long_prob_threshold=long_prob_threshold,
            short_prob_threshold=short_prob_threshold,
            min_adx=min_adx,
        )

        self.risk_limits = RiskLimits()

        if self.mode == "live":
            if not allow_live_trading:
                raise ValueError(
                    "Live trading lock active. Set ALLOW_LIVE_TRADING=YES_I_UNDERSTAND."
                )
            if not api_key or not api_secret:
                raise ValueError("Live mode requires exchange API key and secret.")

            self.broker = LiveBroker(
                exchange_name=exchange_name,
                api_key=api_key,
                api_secret=api_secret,
                testnet=testnet,
            )

            detected_balance = self.broker.get_balance_usdt()
            if detected_balance <= 0:
                raise RuntimeError(
                    "Live account balance is zero or unavailable. Fund account or check API permissions."
                )

            allocated_balance = detected_balance * balance_allocation_pct
            if allocated_balance <= 0:
                raise RuntimeError("Allocated live balance is zero. Check CAPITAL_ALLOCATION_PER_SYMBOL_PCT")

            self.risk_state = RiskState(starting_balance=allocated_balance)
        else:
            if balance_allocation_pct <= 0:
                raise ValueError("balance_allocation_pct must be > 0")
            self.broker = PaperBroker()
            self.risk_state = RiskState(starting_balance=starting_balance_usdt * balance_allocation_pct)

        self.logger = TradeLogger()

        # --- Trade state ---
        self.trailing_stop = None
        self.initial_stop = None
        self.initial_risk = None
        self.take_profit_price = None
        self.highest_price = None
        self.lowest_price = None

        self.last_entry_price = None
        self.last_entry_prob = None

        # --- Safety controls ---
        effective_min_balance = min_balance_usdt * self.balance_allocation_pct
        if effective_min_balance >= self.risk_state.starting_balance:
            effective_min_balance = self.risk_state.starting_balance * 0.90

        self.min_balance_usdt = max(0.0, effective_min_balance)
        self.last_trade_time = None
        self.cooldown = timedelta(minutes=cooldown_minutes)

        self._preflight_check()

    def _fetch_ohlcv_with_retry(self, limit: int):
        last_error = None
        for attempt in range(1, self.data_retry_attempts + 1):
            try:
                return self.data.fetch_ohlcv(
                    self.symbol,
                    self.timeframe,
                    limit=limit,
                )
            except Exception as err:
                last_error = err
                print(f"{self.symbol} fetch attempt {attempt}/{self.data_retry_attempts} failed: {err}")
                if attempt < self.data_retry_attempts:
                    time.sleep(1)

        raise RuntimeError(f"Market data fetch failed for {self.symbol}: {last_error}")

    def _preflight_check(self) -> None:
        df = self._fetch_ohlcv_with_retry(limit=max(self.lookback, 220))

        if len(df) < self.lookback:
            raise RuntimeError(
                f"Insufficient candles for {self.symbol} {self.timeframe}: {len(df)} < {self.lookback}"
            )

        df = self.strategy.compute_indicators(df)
        _, prob_up = self.strategy.generate_signal(df)

        if not (0.0 <= prob_up <= 1.0):
            raise RuntimeError(f"Model returned invalid probability: {prob_up}")

    def run_once(self):
        # =========================
        # KILL SWITCH
        # =========================
        if self.risk_state.current_balance <= self.min_balance_usdt:
            print(f"🛑 KILL SWITCH [{self.symbol}]: balance too low. Trading stopped.")
            raise KeyboardInterrupt

        # Reset daily risk
        self.risk_state.reset_if_new_day(datetime.utcnow().date())

        # Fetch market data
        df = self._fetch_ohlcv_with_retry(limit=self.lookback)

        price = df.iloc[-1]["close"]

        # Indicators + signal
        df = self.strategy.compute_indicators(df)
        signal_side, prob_up = self.strategy.generate_signal(df)

        # =========================
        # EXIT LOGIC
        # =========================
        if self.broker.position is not None:
            side = self.broker.position.side
            entry = self.last_entry_price
            qty = self.broker.position.qty

            hard_stop = self.initial_stop if self.initial_stop is not None else entry * (0.99 if side == "LONG" else 1.01)

            if side == "LONG":
                self.highest_price = max(self.highest_price, price)
                atr = df.iloc[-1]["atr"]
                new_trail = self.highest_price - (1.5 * atr)
                self.trailing_stop = max(self.trailing_stop, new_trail)

                if self.initial_risk is not None and price >= entry + (self.breakeven_rr * self.initial_risk):
                    self.trailing_stop = max(self.trailing_stop, entry)

                hit_take_profit = self.take_profit_price is not None and price >= self.take_profit_price
                hit_trail = price <= self.trailing_stop
                hit_hard = price <= hard_stop
            else:
                self.lowest_price = min(self.lowest_price, price)
                atr = df.iloc[-1]["atr"]
                new_trail = self.lowest_price + (1.5 * atr)
                self.trailing_stop = min(self.trailing_stop, new_trail)

                if self.initial_risk is not None and price <= entry - (self.breakeven_rr * self.initial_risk):
                    self.trailing_stop = min(self.trailing_stop, entry)

                hit_take_profit = self.take_profit_price is not None and price <= self.take_profit_price
                hit_trail = price >= self.trailing_stop
                hit_hard = price >= hard_stop

            if hit_take_profit or hit_trail or hit_hard:
                pnl = self.broker.close_position(price, symbol=self.symbol)
                self.risk_state.register_trade(pnl)

                self.logger.log(
                    symbol=self.symbol,
                    side=side,
                    entry_price=entry,
                    exit_price=price,
                    qty=qty,
                    pnl=pnl,
                    balance=self.risk_state.current_balance,
                    prob_up=self.last_entry_prob,
                )

                self.trailing_stop = None
                self.initial_stop = None
                self.initial_risk = None
                self.take_profit_price = None
                self.highest_price = None
                self.lowest_price = None

                exit_reason = "TP" if hit_take_profit else ("TRAIL" if hit_trail else "HARD_STOP")
                print(f"EXIT {self.symbol} {side} [{exit_reason}] | PnL={_fmt_money(pnl)}")
                return

        # =========================
        # ENTRY LOGIC
        # =========================
        if self.broker.position is None:
            if not self.risk_state.trading_allowed(self.risk_limits):
                print("Risk halt")
                return

            # Cooldown check (ENTRY ONLY)
            now = datetime.utcnow()
            if self.last_trade_time and now - self.last_trade_time < self.cooldown:
                return

            if signal_side is not None:
                if self.mode == "live" and signal_side == "SHORT":
                    print("Live spot mode: SHORT signal ignored.")
                    return

                if self.on_before_entry is not None and not self.on_before_entry(self.symbol):
                    print(f"{self.symbol} entry blocked by portfolio guard")
                    return

                qty = self.strategy.position_size(
                    balance=self.risk_state.current_balance,
                    entry_price=price,
                    side=signal_side,
                    max_position_notional_pct=self.max_position_notional_pct,
                )

                if qty > 0:
                    self.broker.open_position(signal_side, price, qty, symbol=self.symbol)

                    self.last_entry_price = price
                    self.last_entry_prob = prob_up
                    self.last_trade_time = now  # ✅ cooldown timestamp

                    self.initial_stop = price * (0.99 if signal_side == "LONG" else 1.01)
                    self.initial_risk = abs(price - self.initial_stop)
                    self.take_profit_price = (
                        price + (self.take_profit_rr * self.initial_risk)
                        if signal_side == "LONG"
                        else price - (self.take_profit_rr * self.initial_risk)
                    )
                    self.trailing_stop = self.initial_stop
                    self.highest_price = price
                    self.lowest_price = price

                    print(f"ENTER {self.symbol} {signal_side} @ {price:.2f}")

        # =========================
        # STATUS PRINT
        # =========================
        balance_usdt = self.risk_state.current_balance
        balance_inr = balance_usdt * USD_TO_INR

        print(
            f"{self.mode.upper()} | {self.symbol} | price={price:.2f} | "
            f"AI_prob_up={prob_up:.3f} | "
            f"Balance={_fmt_money(balance_usdt)} USDT (~₹{balance_inr:.0f})"
        )

    def run_loop(self, sleep_seconds: int = 900):
        print("🚀 Trading Runner started")

        while True:
            try:
                self.run_once()
                time.sleep(sleep_seconds)

            except KeyboardInterrupt:
                print("Stopped by user / kill-switch")
                break

            except Exception as e:
                print("Runner error:", e)
                time.sleep(30)
