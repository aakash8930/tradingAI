from execution.runner import TradingRunner


if __name__ == "__main__":
    runner = TradingRunner(
        symbol="BTC/USDT",
        timeframe="5m",
        model_path="ai_model.pt",
        scaler_path="scaler.save",
    )

    runner.run_loop()
