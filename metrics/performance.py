# metrics/performance.py

import pandas as pd


def performance_summary(df: pd.DataFrame) -> dict:
    wins = df[df["pnl"] > 0]
    losses = df[df["pnl"] <= 0]

    win_rate = len(wins) / len(df) if len(df) else 0
    avg_win = wins["pnl"].mean() if len(wins) else 0
    avg_loss = losses["pnl"].abs().mean() if len(losses) else 0

    expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

    peak = df["balance"].cummax()
    drawdown = (peak - df["balance"]) / peak
    max_dd = drawdown.max()

    return {
        "trades": len(df),
        "win_rate": round(win_rate, 3),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "expectancy": round(expectancy, 2),
        "max_drawdown": round(max_dd, 3),
    }
