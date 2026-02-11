# run_historical.py

import ccxt
import pandas as pd

from v2.backtest.simulator import HistoricalSimulator


SYMBOL = "BTC/USDT"
TIMEFRAME = "15m"
CANDLES = 10000  # increase later


exchange = ccxt.binance({"enableRateLimit": True})

print("Fetching historical data...")
all_bars = []
since = exchange.milliseconds() - (CANDLES * 5 * 60 * 1000)

while len(all_bars) < CANDLES:
    bars = exchange.fetch_ohlcv(
        SYMBOL,
        TIMEFRAME,
        since=since,
        limit=1000
    )
    
    if not bars:
        break

    all_bars.extend(bars)
    since = bars[-1][0] + 1

print(f"Fetched {len(all_bars)} candles")


df = pd.DataFrame(
    all_bars, columns=["time","open","high","low","close","volume"]
)


sim = HistoricalSimulator(
    model_path="v2/models/ai_model.pt",
    scaler_path="v2/models/scaler.save",
)

print("Running simulation...")

for i in range(sim.lookback, len(df)):
    window = df.iloc[i-sim.lookback:i].copy()
    sim.step(window)

print("Simulation finished.")

sim.export("v2/data_outputs/v2_backtest_trades.csv")
