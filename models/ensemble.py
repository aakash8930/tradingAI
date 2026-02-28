# models/ensemble.py

from typing import List
import numpy as np

from execution.regime import detect_regime, MarketRegime


class EnsembleDirectionModel:
    """
    Combines multiple DirectionModels using
    regime-aware weighted averaging.
    """

    def __init__(self, models: List):
        if not models:
            raise ValueError("Ensemble requires at least one model")
        self.models = models

        # Default weights (will be adjusted dynamically)
        self.base_weights = np.ones(len(models)) / len(models)

    def predict_proba(self, df):
        probs = np.array([m.predict_proba(df) for m in self.models])

        regime = detect_regime(df)
        weights = self._weights_for_regime(regime)

        prob = float(np.average(probs, weights=weights))
        return prob

    def _weights_for_regime(self, regime: MarketRegime):
        n = len(self.models)

        if n == 1:
            return self.base_weights

        # Convention:
        # models[0] = symbol model
        # models[1] = BTC context model (if exists)

        if regime == MarketRegime.TRENDING:
            return np.array([0.7, 0.3])[:n]

        if regime == MarketRegime.RANGING:
            return np.array([0.85, 0.15])[:n]

        # CHOPPY → be conservative
        return np.array([0.6, 0.4])[:n]

