import torch
import ta
import joblib
import numpy as np

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

model = AIModel()
model.load_state_dict(torch.load("ai_model.pt", map_location="cpu"))
model.eval()

scaler = joblib.load("scaler.save")

def prepare_features(df):
    df = df.copy()
    df["ema9"] = ta.trend.EMAIndicator(df["close"], 9).ema_indicator()
    df["ema21"] = ta.trend.EMAIndicator(df["close"], 21).ema_indicator()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()
    df["ret"] = df["close"].pct_change()
    df["vol"] = df["ret"].rolling(10).std()
    df.dropna(inplace=True)
    return df

_ai_buffer = []

def predict_signal(df):
    row = df.iloc[-1]

    features = [[
        row["ema9"],
        row["ema21"],
        row["rsi"],
        row["ret"],
        row["vol"]
    ]]

    features = scaler.transform(features)
    features = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        raw = model(features).item()

    _ai_buffer.append(raw)
    if len(_ai_buffer) > 200:
        _ai_buffer.pop(0)

    mean = np.mean(_ai_buffer)
    std = np.std(_ai_buffer) + 1e-6
    z = (raw - mean) / std

    confidence = float(np.clip(0.5 + z * 0.15, 0.25, 0.75))
    signal = "LONG" if confidence > 0.52 else "SHORT" if confidence < 0.48 else "NONE"

    return signal, confidence
