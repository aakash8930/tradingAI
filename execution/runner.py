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
from risk.limits import RiskLimits, RiskState
from features.technicals import compute_core_features
from metrics.self_report import DailyAIReport


class TradingRunner:
    """
    Fully autonomous AI trading runner.
    """

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
        self.mode = mode

        self.data = MarketDataFetcher()

        base = DirectionModel.for_symbol(symbol)
        models = [base]
        if symbol != "BTC/USDT":
            try:
                models.append(DirectionModel.for_symbol("BTC/USDT"))
            except Exception:
                pass

        self.model = EnsembleDirectionModel(models)
        self.strategy = StrategyEngine(self.model, risk_per_trade)

        self.supervisor = AISupervisor()
        self.regime = RegimeController()

        self.risk_limits = RiskLimits()
        self.risk_state = RiskState(starting_balance_usdt)

        self.broker = ShadowBroker()
        self.report = DailyAIReport()

        self.cooldown = timedelta(minutes=cooldown_minutes)
        self.last_trade_time = None

        self.daily_stats = {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "net_pnl": 0.0,
            "peak_equity": starting_balance_usdt,
        }

        print(f"[AUTONOMOUS AI] Runner active for {symbol}")

    # ------------------------------------------------
    def run_once(self):
        df = self.data.fetch_ohlcv(self.symbol, self.timeframe, self.lookback)
        df = compute_core_features(df)

        today = datetime.utcnow().date()
        self.risk_state.reset_if_new_day(today)

        self.supervisor.update_equity(self.risk_state.current_balance)

        # ---------------- REGIME ----------------
        regime = self.regime.detect(df)
        if not self.regime.trading_allowed(regime):
            return

        regime_risk_mult = self.regime.risk_multiplier(regime)

        # ---------------- SUPERVISOR ----------------
        decision = self.supervisor.decide()
        if not decision.trade_allowed:
            return

        # ---------------- EXIT ----------------
        if self.broker.position:
            price = float(df.iloc[-1]["close"])
            pnl = self.broker.close_position(price, self.symbol)

            self.risk_state.register_trade(pnl)
            self.supervisor.register_trade(pnl)

            self.daily_stats["trades"] += 1
            self.daily_stats["net_pnl"] += pnl
            self.daily_stats["peak_equity"] = max(
                self.daily_stats["peak_equity"],
                self.risk_state.current_balance,
            )

            if pnl > 0:
                self.daily_stats["wins"] += 1
            else:
                self.daily_stats["losses"] += 1

            return

        # ---------------- ENTRY ----------------
        if not self.risk_state.trading_allowed(self.risk_limits):
            return

        if self.last_trade_time and datetime.utcnow() - self.last_trade_time < self.cooldown:
            return

        signal, prob = self.strategy.generate_signal(df)
        if not signal:
            return

        price = float(df.iloc[-1]["close"])
        risk_balance = (
            self.risk_state.current_balance
            * decision.risk_multiplier
            * regime_risk_mult
        )

        qty = self.strategy.position_size(
            balance=risk_balance,
            entry_price=price,
            side=signal,
        )

        if qty <= 0:
            return

        self.broker.open_position(signal, price, qty, self.symbol)
        self.last_trade_time = datetime.utcnow()

    # ------------------------------------------------
    def run_loop(self, sleep_seconds: int = 900):
        print(f"🚀 Autonomous AI Trader running [{self.symbol}]")

        last_report_day = None

        while True:
            try:
                self.run_once()

                today = datetime.utcnow().date()
                if last_report_day != today:
                    if self.daily_stats["trades"] > 0:
                        peak = self.daily_stats["peak_equity"]
                        dd = (
                            (peak - self.risk_state.current_balance) / peak
                            if peak > 0
                            else 0.0
                        )

                        self.report.write(
                            symbol=self.symbol,
                            trades=self.daily_stats["trades"],
                            wins=self.daily_stats["wins"],
                            losses=self.daily_stats["losses"],
                            net_pnl=self.daily_stats["net_pnl"],
                            max_dd=dd,
                            notes="autonomous-run",
                        )

                    self.daily_stats = {
                        "trades": 0,
                        "wins": 0,
                        "losses": 0,
                        "net_pnl": 0.0,
                        "peak_equity": self.risk_state.current_balance,
                    }
                    last_report_day = today

                time.sleep(sleep_seconds)

            except KeyboardInterrupt:
                print("Stopped by user")
                break

            except Exception as e:
                print("Runner error:", e)
                time.sleep(30)