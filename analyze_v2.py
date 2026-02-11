import pandas as pd

df = pd.read_csv("v2/data_outputs/v2_backtest_trades.csv")

print("Total trades:", len(df))

wins = df[df["pnl"] > 0]
losses = df[df["pnl"] <= 0]

win_rate = len(wins) / len(df)
avg_win = wins["pnl"].mean()
avg_loss = abs(losses["pnl"].mean())
expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

equity = df["balance"]
peak = equity.cummax()
drawdown = (peak - equity) / peak
max_dd = drawdown.max()

print("Win rate:", round(win_rate, 4))
print("Avg win:", round(avg_win, 4))
print("Avg loss:", round(avg_loss, 4))
print("Expectancy:", round(expectancy, 4))
print("Max drawdown:", round(max_dd, 4))
