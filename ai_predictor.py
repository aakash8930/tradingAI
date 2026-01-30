import torch
import ta
import joblib
import pandas as pd
import numpy as np

# ================= LOAD MODEL =================
model = torch.nn.Sequential(
    torch.nn.Linear(6, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 1),
    torch.nn.Sigmoid()
)

model.load_state_dict(torch.load("ai_model.pt", map_location="cpu"))
model.eval()

scaler = joblib.load("scaler.save")

# ================= FEATURES =================
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

# ================= AI PREDICTION =================
_ai_buffer = []

def predict_probability(df):
    if df is None or len(df) == 0:
        return 0.5

    row = df.iloc[-1]

    features = [[
        row["ema_fast"],
        row["ema_slow"],
        row["rsi"],
        row["returns"],
        row["volatility"],
        row["ema_dist"],
    ]]

    features = scaler.transform(features)
    features = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        raw = model(features).item()

    # ---- SIMPLE, SAFE CALIBRATION ----
    _ai_buffer.append(raw)
    if len(_ai_buffer) > 200:
        _ai_buffer.pop(0)

    mean = np.mean(_ai_buffer)
    std = np.std(_ai_buffer) + 1e-6

    z = (raw - mean) / std
    calibrated = 0.5 + z * 0.12   # gentle spread

    return float(np.clip(calibrated, 0.25, 0.75))
