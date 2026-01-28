import ccxt
import pandas as pd
import ta

# ===============================
# SETTINGS
# ===============================
SYMBOL = "BTC/USDT"
TIMEFRAME = "15m"     # higher TF = cleaner trends
CANDLES = 3500        # enough for EMA200 + learning

exchange = ccxt.binance({"timeout": 30000})

# ===============================
# DATA FETCH
# ===============================
def get_data():
    bars = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=CANDLES)
    df = pd.DataFrame(
        bars, columns=["time", "open", "high", "low", "close", "volume"]
    )
    return df

# ===============================
# FEATURE ENGINEERING
# ===============================
def build_features(df):
    df = df.copy()

    # --- Trend EMAs (MATCH main.py) ---
    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], window=9).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
    df["ema_200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()

    # ✅ EMA distance (CRITICAL for confidence spread)
    df["ema_dist"] = (df["ema_fast"] - df["ema_slow"]) / df["close"]

    # --- Momentum ---
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    # --- Volatility ---
    df["returns"] = df["close"].pct_change()
    df["volatility"] = df["returns"].rolling(10).std()

    # --- Trend strength ---
    adx = ta.trend.ADXIndicator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14
    )
    df["adx"] = adx.adx()

    # --- Trend regime (saved for Day 4 use) ---
    df["trend"] = (
        (df["close"] > df["ema_200"]) &
        (df["adx"] > 20)
    ).astype(int)

    df.dropna(inplace=True)
    return df

# ===============================
# DATASET CREATION (DAY 3.6)
# ===============================
def build_dataset():
    df = get_data()
    df = build_features(df)

    X = []
    y = []

    FUTURE_WINDOW = 3          # candles ahead
    MIN_MOVE = 0.002           # 0.2% required move

    for i in range(len(df) - FUTURE_WINDOW):
        row = df.iloc[i]
        future_price = df.iloc[i + FUTURE_WINDOW]["close"]

        future_return = (future_price - row["close"]) / row["close"]

        features = [
            row["ema_fast"],
            row["ema_slow"],
            row["rsi"],
            row["returns"],
            row["volatility"],
            row["ema_dist"]
        ]

        # ✅ strength-aware label
        label = 1 if future_return > MIN_MOVE else 0

        X.append(features)
        y.append(label)

    dataset = pd.DataFrame(
        X,
        columns=[
            "ema_fast",
            "ema_slow",
            "rsi",
            "returns",
            "volatility",
            "ema_dist"
        ]
    )

    dataset["target"] = y

    dataset.to_csv("training_data.csv", index=False)
    print(f"✅ Dataset created: {len(dataset)} rows (Day 3.6 strength-labeled)")

if __name__ == "__main__":
    build_dataset()
