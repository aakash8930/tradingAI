import torch, joblib, ta, numpy as np

model = torch.load("ai_model.pt", map_location="cpu")
model.eval()
scaler = joblib.load("scaler.save")

def prepare_features(df):
    df = df.copy()
    df["ema9"] = ta.trend.EMAIndicator(df["close"],9).ema_indicator()
    df["ema21"] = ta.trend.EMAIndicator(df["close"],21).ema_indicator()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"],14).rsi()
    df["ret"] = df["close"].pct_change()
    df["vol"] = df["ret"].rolling(10).std()
    df.dropna(inplace=True)
    return df

def predict_signal(df):
    row = df.iloc[-1]
    X = [[row["ema9"], row["ema21"], row["rsi"], row["ret"], row["vol"]]]
    X = scaler.transform(X)
    X = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        prob = model(X).item()

    if prob > 0.55:
        return "LONG", prob
    elif prob < 0.45:
        return "SHORT", 1 - prob
    return "NONE", prob
