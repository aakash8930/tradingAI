# ai_predictor.py

import torch
import ta
import joblib
import pandas as pd
import numpy as np
from collections import deque

# ================= MODEL =================
MODEL_FEATURES = [
    "ema_fast",
    "ema_slow",
    "rsi",
    "returns",
    "ema_dist",
]

class AIModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(5, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

model = AIModel()

# 👇 THIS MATCHES YOUR SAVED MODEL
state_dict = torch.load(
    "ai_model.pt",
    map_location="cpu",
    weights_only=False
)
model.load_state_dict(state_dict)
model.eval()

scaler = joblib.load("scaler.save")

# ================= FEATURE ENGINEERING =================
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], 9).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], 21).ema_indicator()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
    df["returns"] = df["close"].pct_change()
    df["ema_dist"] = (df["ema_fast"] - df["ema_slow"]) / df["close"]

    df.dropna(inplace=True)
    return df

# ================= AI PROBABILITY =================
_ai_buffer = deque(maxlen=300)

def predict_probability(df: pd.DataFrame) -> float:
    if df is None or len(df) == 0:
        return 0.5

    row = df.iloc[-1]
    features = [[row[f] for f in MODEL_FEATURES]]

    features = scaler.transform(features)
    features = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        raw = model(features).item()

    _ai_buffer.append(raw)

    mean = np.mean(_ai_buffer)
    std = np.std(_ai_buffer) + 1e-6
    z = (raw - mean) / std

    calibrated = 0.5 + z * 0.15
    return float(np.clip(calibrated, 0.25, 0.75))
