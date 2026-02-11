import ccxt
import pandas as pd

from v2.backtest.simulator import HistoricalSimulator


SYMBOL = "BTC/USDT"
TIMEFRAME = "5m"
TOTAL_CANDLES = 20000

TRAIN_SIZE = 8000
TEST_SIZE = 3000


exchange = ccxt.binance({"enableRateLimit": True})

print("Fetching historical data...")

all_bars = []
since = exchange.milliseconds() - (TOTAL_CANDLES * 5 * 60 * 1000)

while len(all_bars) < TOTAL_CANDLES:
    bars = exchange.fetch_ohlcv(
        SYMBOL,
        TIMEFRAME,
        since=since,
        limit=1000,
    )

    if not bars:
        break

    all_bars.extend(bars)
    since = bars[-1][0] + 1

print(f"Fetched {len(all_bars)} candles")

df = pd.DataFrame(
    all_bars,
    columns=["time", "open", "high", "low", "close", "volume"],
)

segments = []
start = 0

while start + TRAIN_SIZE + TEST_SIZE <= len(df):
    train_df = df.iloc[start : start + TRAIN_SIZE].copy()
    test_df = df.iloc[start + TRAIN_SIZE : start + TRAIN_SIZE + TEST_SIZE].copy()

    segments.append((train_df, test_df))
    start += TEST_SIZE  # rolling forward


results = []

for idx, (_, test_df) in enumerate(segments):
    print(f"\n--- Walk Forward Segment {idx+1} ---")

    sim = HistoricalSimulator(
        model_path="v2/models/ai_model.pt",
        scaler_path="v2/models/scaler.save",
        starting_balance=1000.0,
    )

    for i in range(sim.lookback, len(test_df)):
        window = test_df.iloc[i - sim.lookback : i].copy()
        sim.step(window)

    trades = pd.DataFrame(sim.trades)

    if len(trades) == 0:
        print("No trades in this segment.")
        continue

    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]

    win_rate = len(wins) / len(trades)
    avg_win = wins["pnl"].mean()
    avg_loss = abs(losses["pnl"].mean())

    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

    equity = trades["balance"]
    peak = equity.cummax()
    drawdown = (peak - equity) / peak
    max_dd = drawdown.max()

    print(f"Trades: {len(trades)}")
    print(f"Win rate: {win_rate:.4f}")
    print(f"Expectancy: {expectancy:.4f}")
    print(f"Max DD: {max_dd:.4f}")

    results.append({
        "segment": idx + 1,
        "trades": len(trades),
        "win_rate": win_rate,
        "expectancy": expectancy,
        "max_dd": max_dd,
    })


print("\n========== SUMMARY ==========")

summary = pd.DataFrame(results)
print(summary)

print("\nAverage Expectancy:", summary["expectancy"].mean())
print("Worst Drawdown:", summary["max_dd"].max())
