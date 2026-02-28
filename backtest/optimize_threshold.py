# backtest/optimize_threshold.py

import numpy as np
import pandas as pd

from backtest.vector_engine import VectorBacktestEngine


def optimize_long_threshold(
    symbol: str,
    model_path: str,
    scaler_path: str,
    timeframe: str = "15m",
    lookback: int = 300,
    limit: int = 20_000,
):
    bt = VectorBacktestEngine(
        symbol=symbol,
        timeframe=timeframe,
        model_path=model_path,
        scaler_path=scaler_path,
        lookback=lookback,
    )

    df = bt.run(limit=limit)

    results = []

    for th in np.arange(0.48, 0.61, 0.01):
        mask = (
            (df["prob_up"] >= th)
            & (df["atr_pct"] > 0.0012)
            & (df["adx"] >= 8)
        )

        trades = df[mask].copy()
        if len(trades) < 50:
            continue

        trades["ret"] = trades["close"].pct_change().shift(-1)
        trades["net_ret"] = trades["ret"] - bt.fee_pct

        expectancy = trades["net_ret"].mean()
        win_rate = (trades["net_ret"] > 0).mean()

        equity = (1 + trades["net_ret"]).cumprod()
        peak = equity.cummax()
        drawdown = (peak - equity) / peak
        max_dd = drawdown.max()

        score = expectancy / (max_dd + 1e-6)

        results.append({
            "threshold": round(th, 3),
            "trades": len(trades),
            "expectancy": expectancy,
            "win_rate": win_rate,
            "max_dd": max_dd,
            "score": score,
        })

    res = pd.DataFrame(results).sort_values("score", ascending=False)

    if res.empty:
        raise RuntimeError("No viable thresholds found.")

    return res.iloc[0], res
