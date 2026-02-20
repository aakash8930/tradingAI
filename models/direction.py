import os
import json
import torch
import joblib
import numpy as np

from features.technicals import compute_core_features
from models.model_identity import MODEL_NAME, MODEL_VERSION


class DirectionModel:
    """
    Directional AI model.

    Responsibilities:
    - Load trained model, scaler, metadata
    - Produce probability of upward move
    - Auto-derive per-symbol probability thresholds
    """

    @classmethod
    def for_symbol(cls, symbol: str):
        folder = f"models/{symbol.replace('/', '_')}"
        model_path = f"{folder}/model.pt"
        scaler_path = f"{folder}/scaler.save"
        metadata_path = f"{folder}/metadata.json"

        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            raise FileNotFoundError(f"No trained model for {symbol}")

        return cls(
            model_path=model_path,
            scaler_path=scaler_path,
            metadata_path=metadata_path,
        )

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        metadata_path: str | None = None,
    ):
        self.device = torch.device("cpu")
        self.model_name = MODEL_NAME
        self.model_version = MODEL_VERSION

        # -------------------------
        # Defaults (safe fallback)
        # -------------------------
        self.feature_columns = [
            "ema_fast",
            "ema_slow",
            "rsi",
            "ret",
            "vol",
            "atr_pct",
            "adx",
        ]

        # -------------------------
        # Load metadata (if exists)
        # -------------------------
        self.metadata = {}
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

            self.feature_columns = self.metadata.get(
                "feature_columns",
                self.feature_columns,
            )
            self.model_name = self.metadata.get(
                "model_name",
                self.model_name,
            )
            self.model_version = self.metadata.get(
                "model_version",
                self.model_version,
            )

        # -------------------------
        # Load model & scaler
        # -------------------------
        self.model = self._load_model(model_path)
        self.scaler = joblib.load(scaler_path)

        # -------------------------
        # Auto-derived thresholds
        # -------------------------
        self._init_thresholds()

    # ==================================================
    # INTERNALS
    # ==================================================
    def _load_model(self, model_path: str) -> torch.nn.Module:
        class AIModel(torch.nn.Module):
            def __init__(self, input_dim: int):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, 32),
                    torch.nn.ReLU(),
                    torch.nn.Linear(32, 16),
                    torch.nn.ReLU(),
                    torch.nn.Linear(16, 1),
                    torch.nn.Sigmoid(),
                )

            def forward(self, x):
                return self.net(x)

        model = AIModel(input_dim=len(self.feature_columns))
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _init_thresholds(self) -> None:
        """
        Derive probability thresholds from validation metadata.
        This is the MAIN EDGE IMPROVEMENT.
        """
        metrics = self.metadata.get("metrics", {})

        pos_rate = float(metrics.get("val_positive_rate", 0.5))
        f1 = float(metrics.get("val_f1", 0.0))

        # Base threshold from class balance
        if pos_rate < 0.20:
            long_th = 0.45
        elif pos_rate < 0.30:
            long_th = 0.50
        else:
            long_th = 0.55

        # Penalize weak models
        if f1 < 0.20:
            long_th += 0.05

        # Safety clamp
        long_th = float(np.clip(long_th, 0.40, 0.65))

        self.long_threshold = long_th
        self.short_threshold = 1.0 - long_th

        print(
            f"[MODEL] {self.model_name} {self.model_version} | "
            f"LONG_TH={self.long_threshold:.2f} "
            f"SHORT_TH={self.short_threshold:.2f}"
        )

    # ==================================================
    # PUBLIC API
    # ==================================================
    def predict_proba(self, df):
        """
        Returns probability of price going UP in next horizon
        """
        df = compute_core_features(df)
        row = df.iloc[-1]

        X = np.array(
            [[row[col] for col in self.feature_columns]],
            dtype=np.float32,
        )

        X = self.scaler.transform(X)
        X = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            prob_up = self.model(X).item()

        return float(prob_up)
