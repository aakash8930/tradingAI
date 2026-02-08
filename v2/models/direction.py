import torch
import joblib
import numpy as np
from pathlib import Path

from v2.features.technicals import compute_core_features


class DirectionModel:
    """
    Directional AI model.
    Responsibility:
    - Load trained model & scaler
    - Produce probability of upward move
    """

    def __init__(self, model_path: str, scaler_path: str):
        self.device = torch.device("cpu")

        self.model = self._load_model(model_path)
        self.scaler = joblib.load(scaler_path)

    def _load_model(self, model_path: str):
        model = torch.nn.Sequential(
            torch.nn.Linear(5, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid(),
        )

        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def predict_proba(self, df):
        """
        Returns probability of price going UP in next horizon
        """
        df = compute_core_features(df)
        row = df.iloc[-1]

        X = np.array([[
            row["ema_fast"],
            row["ema_slow"],
            row["rsi"],
            row["ret"],
            row["vol"],
        ]])

        X = self.scaler.transform(X)
        X = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            prob_up = self.model(X).item()

        return float(prob_up)
