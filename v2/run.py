import os
import sys
from v2.execution.runner import TradingRunner

# Ensure v2 is the import root, regardless of where we run from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)



if __name__ == "__main__":
    runner = TradingRunner(
        symbol="BTC/USDT",
        timeframe="5m",
        model_path=os.path.join(BASE_DIR, "..", "ai_model.pt"),
        scaler_path=os.path.join(BASE_DIR, "..", "scaler.save"),
    )

    runner.run_loop()
