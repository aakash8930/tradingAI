import torch
import ta
import joblib
import pandas as pd
import numpy as np

model = torch.nn.Sequential(
    torch.nn.Linear(6,32),
    torch.nn.ReLU(),
    torch.nn.Linear(32,16),
    torch.nn.ReLU(),
    torch.nn.Linear(16,1),
    torch.nn.Sigmoid()
)

model.load_state_dict(torch.load("ai_model.pt", map_location="cpu"))
model.eval()

scaler = joblib.load("scaler.save")


def prepare_features(df):
    df = df.copy()
    df["ema_fast"] = ta.trend.EMAIndicator(df["close"],9).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"],21).ema_indicator()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"],14).rsi()
    df["returns"] = df["close"].pct_change()
    df["volatility"] = df["returns"].rolling(10).std()
    df["ema_dist"] = (df["ema_fast"] - df["ema_slow"]) / df["close"]
    df.dropna(inplace=True)
    return df


def predict_probability(df):
    if df is None or len(df) < 5:
        return 0.5

    r = df.iloc[-1]
    X = [[
        r["ema_fast"],
        r["ema_slow"],
        r["rsi"],
        r["returns"],
        r["volatility"],
        r["ema_dist"]
    ]]

    X = scaler.transform(X)
    X = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        p = model(X).item()

    # controlled confidence expansion
    p = np.clip(p, 0.1, 0.9)
    p = 0.5 + (p - 0.5) * 1.8
    return float(np.clip(p, 0.05, 0.95))
