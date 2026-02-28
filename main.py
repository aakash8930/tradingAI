# main.py

from config.env_loader import load_env_file
from config.live import LiveSettings
from execution.runner import TradingRunner
from execution.multi_runner import MultiSymbolTradingSystem


def main():
    # Load .env
    load_env_file()

    # Load settings
    settings = LiveSettings.from_env()
    settings.validate()

    # -------------------------------
    # MULTI-SYMBOL AUTONOMOUS MODE
    # -------------------------------
    if len(settings.symbols) > 1:
        system = MultiSymbolTradingSystem(settings)
        system.run_loop()
        return

    # -------------------------------
    # SINGLE-SYMBOL MODE
    # -------------------------------
    symbol = settings.symbols[0]

    runner = TradingRunner(
        symbol=symbol,
        timeframe=settings.timeframe,
        lookback=settings.lookback,
        mode=settings.mode,
        starting_balance_usdt=settings.starting_balance_usdt,
        cooldown_minutes=settings.cooldown_minutes,
        risk_per_trade=settings.risk_per_trade,
    )

    runner.run_loop(sleep_seconds=settings.sleep_seconds)


if __name__ == "__main__":
    main()