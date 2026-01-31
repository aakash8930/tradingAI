# ai_predictor.py
import torch
import ta
import joblib
import pandas as pd
import numpy as np

model = torch.nn.Sequential(
    torch.nn.Linear(6, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 1),
    torch.nn.Sigmoid()
)

model.load_state_dict(torch.load("ai_model.pt", map_location="cpu"))
model.eval()

scaler = joblib.load("scaler.save")

def prepare_features(df):
    df = df.copy()
    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], 9).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], 21).ema_indicator()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
    df["returns"] = df["close"].pct_change()
    df["volatility"] = df["returns"].rolling(10).std()
    df["ema_dist"] = (df["ema_fast"] - df["ema_slow"]) / df["close"]
    df.dropna(inplace=True)
    return df

_ai_buf = []

def predict_probability(df):
    if df is None or len(df) == 0:
        return 0.5

    r = df.iloc[-1]

    features = [[
        r["ema_fast"], r["ema_slow"], r["rsi"],
        r["returns"], r["volatility"], r["ema_dist"]
    ]]

    features = scaler.transform(features)
    features = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        raw = model(features).item()

    _ai_buf.append(raw)
    if len(_ai_buf) > 200:
        _ai_buf.pop(0)

    mean = np.mean(_ai_buf)
    std = np.std(_ai_buf) + 1e-6
    z = (raw - mean) / std

    calibrated = 0.55 + z * 0.18
    return float(np.clip(calibrated, 0.35, 0.85))
