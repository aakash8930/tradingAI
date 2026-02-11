from v2.backtest.engine import BacktestEngine


if __name__ == "__main__":
    bt = BacktestEngine(
        symbol="BTC/USDT",
        timeframe="15m",
        model_path="ai_model.pt",
        scaler_path="scaler.save",
    )

    df = bt.run(limit=1000)
    print(df.head())
    print("Signals:", len(df))
