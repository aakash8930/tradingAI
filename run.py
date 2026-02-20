# run.py

import os
import sys
from execution.runner import TradingRunner
from execution.multi_runner import MultiSymbolTradingSystem
from config.env_loader import load_env_file
from config.live import LiveSettings

# Ensure v2 is the import root, regardless of where we run from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)



if __name__ == "__main__":
    load_env_file()
    settings = LiveSettings.from_env()
    settings.validate()

    if len(settings.symbols) > 1:
        multi = MultiSymbolTradingSystem(settings)
        multi.run_loop()
    else:
        runner = TradingRunner(
            symbol=settings.symbol,
            timeframe=settings.timeframe,
            lookback=settings.lookback,
            mode=settings.mode,
            exchange_name=settings.exchange_name,
            api_key=settings.api_key,
            api_secret=settings.api_secret,
            testnet=settings.testnet,
            allow_live_trading=settings.allow_live_trading,
            starting_balance_usdt=settings.starting_balance_usdt,
            min_balance_usdt=settings.min_balance_usdt,
            cooldown_minutes=settings.cooldown_minutes,
            risk_per_trade=settings.risk_per_trade,
            max_position_notional_pct=settings.max_position_notional_pct,
            long_prob_threshold=settings.long_prob_threshold,
            short_prob_threshold=settings.short_prob_threshold,
            min_adx=settings.min_adx,
            take_profit_rr=settings.take_profit_rr,
            breakeven_rr=settings.breakeven_rr,
            balance_allocation_pct=settings.capital_allocation_per_symbol_pct,
            data_retry_attempts=settings.data_retry_attempts,
        )

        runner.run_loop(sleep_seconds=settings.sleep_seconds)
