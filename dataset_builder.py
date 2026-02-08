# dataset_builder.py

import ccxt
import pandas as pd
import ta

SYMBOL = "BTC/USDT"
TIMEFRAME = "5m"
CANDLES = 5000

exchange = ccxt.binance({"enableRateLimit": True})

def build_dataset():
    df = pd.DataFrame(
        exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=CANDLES),
        columns=["time","open","high","low","close","volume"]
    )

    df["ema9"] = ta.trend.EMAIndicator(df["close"], 9).ema_indicator()
    df["ema21"] = ta.trend.EMAIndicator(df["close"], 21).ema_indicator()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
    df["ret"] = df["close"].pct_change()
    df["vol"] = df["ret"].rolling(10).std()

    df["future"] = df["close"].shift(-3)
    df["target"] = (df["future"] > df["close"] * 1.002).astype(int)

    df.dropna(inplace=True)

    df[["ema9","ema21","rsi","ret","vol","target"]].to_csv(
        "training_data.csv", index=False
    )

    print(f"✅ Dataset built: {len(df)} rows")

if __name__ == "__main__":
    build_dataset()
