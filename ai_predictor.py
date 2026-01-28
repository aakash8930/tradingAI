import torch
import pandas as pd
import ta
import joblib

# ===============================
# LOAD MODEL (6 FEATURES)
# ===============================
model = torch.nn.Sequential(
    torch.nn.Linear(6, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 1),
    torch.nn.Sigmoid(),
)

model.load_state_dict(torch.load("ai_model.pt"))
model.eval()

# Load scaler
scaler = joblib.load("scaler.save")


# ===============================
# FEATURE ENGINEERING
# ===============================
def prepare_features(df):
    df = df.copy()

    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], window=9).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    df["returns"] = df["close"].pct_change()
    df["volatility"] = df["returns"].rolling(10).std()

    # EMA distance (trend pressure)
    df["ema_dist"] = (df["ema_fast"] - df["ema_slow"]) / df["close"]

    df.dropna(inplace=True)
    return df


# ===============================
# AI PREDICTION (Day 3.7.1)
# ===============================
def predict_probability(df):
    """
    Decompressed & calibrated AI confidence (0–1)
    """

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

    # ===============================
    # CONFIDENCE DE-COMPRESSION
    # ===============================

    # 1️⃣ Floor (prevents 0.00 lock)
    raw = max(raw, 0.15)

    # 2️⃣ Gamma expansion (expands mid-range)
    gamma = 0.6
    expanded = raw ** gamma

    # 3️⃣ Adaptive asymmetric stretch
    if expanded < 0.5:
        calibrated = 0.5 - (0.5 - expanded) * 1.1
    else:
        calibrated = 0.5 + (expanded - 0.5) * 1.8

    # 4️⃣ Clamp (safety)
    calibrated = max(0.05, min(0.95, calibrated))

    return float(calibrated)
