# dataset_builder.py
import ccxt
import pandas as pd
import ta

SYMBOL = "BTC/USDT"
TIMEFRAME = "5m"
CANDLES = 6000

TP = 0.008      # 0.8%
SL = 0.004      # 0.4%
FUTURE_WINDOW = 24  # 2 hours on 5m

exchange = ccxt.binance({"enableRateLimit": True})

def fetch():
    bars = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=CANDLES)
    return pd.DataFrame(
        bars, columns=["time","open","high","low","close","volume"]
    )

def build_features(df):
    df = df.copy()
    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], 9).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], 21).ema_indicator()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
    df["returns"] = df["close"].pct_change()
    df["volatility"] = df["returns"].rolling(10).std()
    df["ema_dist"] = (df["ema_fast"] - df["ema_slow"]) / df["close"]
    df.dropna(inplace=True)
    return df

def build_dataset():
    df = build_features(fetch())

    X, y = [], []

    for i in range(len(df) - FUTURE_WINDOW):
        entry = df.iloc[i]["close"]
        future = df.iloc[i+1:i+FUTURE_WINDOW+1]

        mfe = (future["high"].max() - entry) / entry
        mae = (future["low"].min() - entry) / entry

        if mfe >= TP and mae > -SL:
            label = 1
        elif mae <= -SL:
            label = 0
        else:
            continue

        r = df.iloc[i]
        X.append([
            r["ema_fast"], r["ema_slow"], r["rsi"],
            r["returns"], r["volatility"], r["ema_dist"]
        ])
        y.append(label)

    out = pd.DataFrame(X, columns=[
        "ema_fast","ema_slow","rsi",
        "returns","volatility","ema_dist"
    ])
    out["target"] = y
    out.to_csv("training_data.csv", index=False)

    print(f"✅ Dataset built: {len(out)} rows")

if __name__ == "__main__":
    build_dataset()
