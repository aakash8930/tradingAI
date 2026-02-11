 from v2.execution.runner import TradingRunner


if __name__ == "__main__":
    runner = TradingRunner(
        symbol="BTC/USDT",
        timeframe="15m",
        model_path="v2/models/ai_model.pt",
        scaler_path="v2/models/scaler.save",
    )

    runner.run_loop()
