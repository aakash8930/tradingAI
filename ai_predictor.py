# ai_predictor.py

import torch
import ta
import joblib
import numpy as np

# ================= MODEL DEFINITION =================
class AIModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(5, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ================= LOAD MODEL =================
model = AIModel()
state_dict = torch.load("ai_model.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# ================= LOAD SCALER =================
scaler = joblib.load("scaler.save")

# ================= FEATURES =================
def prepare_features(df):
    df = df.copy()

    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], 9).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], 21).ema_indicator()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()

    df["ret"] = df["close"].pct_change()
    df["vol"] = df["ret"].rolling(10).std()

    df.dropna(inplace=True)
    return df

# ================= AI PREDICTION =================
_ai_buffer = []

def predict_signal(df):
    row = df.iloc[-1]

    X = [[
        row["ema_fast"],
        row["ema_slow"],
        row["rsi"],
        row["ret"],
        row["vol"],
    ]]

    X = scaler.transform(X)
    X = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        raw = model(X).item()

    # ---- calibration ----
    _ai_buffer.append(raw)
    if len(_ai_buffer) > 200:
        _ai_buffer.pop(0)

    mean = np.mean(_ai_buffer)
    std = np.std(_ai_buffer) + 1e-6
    z = (raw - mean) / std
    ai = float(np.clip(0.5 + z * 0.12, 0.25, 0.75))

    # ---- decision zones ----
    if ai >= 0.62:
        return "LONG", ai
    if ai <= 0.38:
        return "SHORT", ai

    return None, ai
