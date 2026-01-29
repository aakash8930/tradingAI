import torch
import ta
import joblib
import pandas as pd

# ---------------- LOAD MODEL ----------------
model = torch.nn.Sequential(
    torch.nn.Linear(6, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 1),
    torch.nn.Sigmoid(),
)

model.load_state_dict(torch.load("ai_model.pt", map_location="cpu"))
model.eval()

scaler = joblib.load("scaler.save")

# ---------------- FEATURES ----------------
def prepare_features(df: pd.DataFrame):
    df = df.copy()

    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], 9).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], 21).ema_indicator()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()

    df["returns"] = df["close"].pct_change()
    df["volatility"] = df["returns"].rolling(10).std()
    df["ema_dist"] = (df["ema_fast"] - df["ema_slow"]) / df["close"]

    df.dropna(inplace=True)
    return df

# ---------------- PREDICTION ----------------
def predict_probability(df: pd.DataFrame) -> float:
    if df is None or len(df) == 0:
        return 0.5

    row = df.iloc[-1]

    X = [[
        row["ema_fast"],
        row["ema_slow"],
        row["rsi"],
        row["returns"],
        row["volatility"],
        row["ema_dist"],
    ]]

    X = scaler.transform(X)
    X = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        raw = model(X).item()

    # Decompress mid-range confidence
    gamma = 0.6
    raw = max(raw, 0.15)
    expanded = raw ** gamma

    if expanded < 0.5:
        expanded = 0.5 - (0.5 - expanded) * 1.1
    else:
        expanded = 0.5 + (expanded - 0.5) * 1.6

    return float(max(0.1, min(0.9, expanded)))
