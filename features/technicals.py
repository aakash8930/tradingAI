
#features/technicals.py

import ta
import pandas as pd


def compute_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Core indicators used by:
    - models
    - regime detection
    - strategy
    """

    df = df.copy()

    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], 9).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], 21).ema_indicator()
    df["ema200"] = ta.trend.EMAIndicator(df["close"], 200).ema_indicator()

    df["rsi"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()

    df["ret"] = df["close"].pct_change()
    df["vol"] = df["ret"].rolling(10).std()

    atr = ta.volatility.AverageTrueRange(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14,
    )
    df["atr"] = atr.average_true_range()
    df["atr_pct"] = df["atr"] / df["close"]

    adx = ta.trend.ADXIndicator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14,
    )
    df["adx"] = adx.adx()

    df.dropna(inplace=True)
    return df
