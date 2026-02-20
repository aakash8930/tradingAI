# features/technicals.py

import ta
import pandas as pd


def compute_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Core features shared by:
    - training
    - inference
    - backtesting

    DO NOT add strategy-specific logic here.
    """

    df = df.copy()

    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], 9).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], 21).ema_indicator()
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

def compute_backtest_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features used only for backtesting / risk control
    """
    df = df.copy()

    df["ema_200"] = ta.trend.EMAIndicator(df["close"], 200).ema_indicator()

    atr = ta.volatility.AverageTrueRange(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14,
    )
    df["atr"] = atr.average_true_range()

    adx = ta.trend.ADXIndicator(
        high=df["high"],
        low=df["low"],
        close=df["close"],
        window=14,
    )
    df["adx"] = adx.adx()

    df.dropna(inplace=True)
    return df
