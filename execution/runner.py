# execution/runner.py

import time
from datetime import datetime, timedelta

from data.fetcher import MarketDataFetcher
from models.direction import DirectionModel
from models.ensemble import EnsembleDirectionModel
from execution.strategy import StrategyEngine
from execution.shadow_broker import ShadowBroker
from execution.regime_controller import RegimeController
from execution.ai_supervisor import AISupervisor
from execution.market_guard import MarketGuard
from risk.limits import RiskLimits, RiskState
from features.technicals import compute_core_features
from metrics.self_report import DailyAIReport


class TradingRunner:
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        lookback: int = 300,
        mode: str = "shadow",
        starting_balance_usdt: float = 500.0,
        cooldown_minutes: int = 30,
        risk_per_trade: float = 0.01,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.lookback = lookback

        self.data = MarketDataFetcher()

        base_model = DirectionModel.for_symbol(symbol)
        models = [base_model]
        if symbol != "BTC/USDT":
            try:
                models.append(DirectionModel.for_symbol("BTC/USDT"))
            except Exception:
                pass

        self.model = EnsembleDirectionModel(models)
        self.strategy = StrategyEngine(self.model, risk_per_trade)

        self.supervisor = AISupervisor()
        self.regime_ctrl = RegimeController()
        self.market_guard = MarketGuard()

        self.risk_limits = RiskLimits()
        self.risk_state = RiskState(starting_balance_usdt)

        self.broker = ShadowBroker()
        self.report = DailyAIReport()

        self.cooldown = timedelta(minutes=cooldown_minutes)
        self.last_trade_time = None

        self.daily = {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "net_pnl": 0.0,
            "peak": starting_balance_usdt,
        }

        print(f"[AUTONOMOUS AI] {symbol} ready")

    # --------------------------------------------------
    def run_once(self):
        df = self.data.fetch_ohlcv(self.symbol, self.timeframe, self.lookback)
        df = compute_core_features(df)

        today = datetime.utcnow().date()

        self.risk_state.reset_if_new_day(today)
        self.supervisor.update_equity(self.risk_state.current_balance)

        # -------- GLOBAL SAFETY --------
        if not self.market_guard.allow_trading(
            balance=self.risk_state.current_balance,
            today=today,
        ):
            return

        regime = self.regime_ctrl.detect(df)
        if not self.regime_ctrl.trading_allowed(regime):
            return

        decision = self.supervisor.decide()
        if not decision.trade_allowed:
            return

        # -------- EXIT --------
        if self.broker.position:
            price = float(df.iloc[-1]["close"])
            pnl = self.broker.close_position(price, self.symbol)

            self.market_guard.register_trade(pnl)
            self.risk_state.register_trade(pnl)
            self.supervisor.register_trade(pnl)

            self.daily["trades"] += 1
            self.daily["net_pnl"] += pnl
            self.daily["peak"] = max(
                self.daily["peak"], self.risk_state.current_balance
            )

            if pnl > 0:
                self.daily["wins"] += 1
            else:
                self.daily["losses"] += 1

            return

        # -------- ENTRY --------
        if self.last_trade_time and datetime.utcnow() - self.last_trade_time < self.cooldown:
            return

        signal, _ = self.strategy.generate_signal(df)
        if not signal:
            return

        price = float(df.iloc[-1]["close"])
        risk_mult = decision.risk_multiplier * self.regime_ctrl.risk_multiplier(regime)

        qty = self.strategy.position_size(
            balance=self.risk_state.current_balance * risk_mult,
            entry_price=price,
            side=signal,
        )

        if qty <= 0:
            return

        self.broker.open_position(signal, price, qty, self.symbol)
        self.last_trade_time = datetime.utcnow()

    # --------------------------------------------------
    def run_loop(self, sleep_seconds: int = 900):
        print(f"🚀 Autonomous AI Trader running [{self.symbol}]")

        last_day = None

        while True:
            try:
                self.run_once()

                today = datetime.utcnow().date()
                if last_day != today and self.daily["trades"] > 0:
                    dd = (
                        (self.daily["peak"] - self.risk_state.current_balance)
                        / self.daily["peak"]
                    )

                    self.report.write(
                        symbol=self.symbol,
                        trades=self.daily["trades"],
                        wins=self.daily["wins"],
                        losses=self.daily["losses"],
                        net_pnl=self.daily["net_pnl"],
                        max_dd=dd,
                        notes="market-guard-active",
                    )

                    self.daily = {
                        "trades": 0,
                        "wins": 0,
                        "losses": 0,
                        "net_pnl": 0.0,
                        "peak": self.risk_state.current_balance,
                    }
                    last_day = today

                time.sleep(sleep_seconds)

            except KeyboardInterrupt:
                print("Stopped by user")
                break
            except Exception as e:
                print("Runner error:", e)
                time.sleep(30)