import ccxt
import pandas as pd
import ta
import time
import sys
import numpy as np
from datetime import datetime, date

from ai_predictor import prepare_features, predict_probability

# ================= MODE =================
LIVE_TRADING = False
USE_TESTNET = True
# =======================================

# ================= SETTINGS =================
SYMBOLS = ["BTC/USDT", "ETH/USDT"]

START_BALANCE = 1000.0
BASE_TRADE_SIZE = 150.0

STOP_LOSS = 0.015
MAX_HOLD_CANDLES = 50
AI_EXIT_WEAK = 0.38

MIN_CANDLES = 210
ETH_CORRELATION_REDUCTION = 0.5
SLEEP_SECONDS = 60

DAILY_TARGET_PCT = 0.01
DAILY_STOP_PCT = -0.01
# ==========================================

# ================= EXCHANGE =================
if USE_TESTNET:
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
        "urls": {
            "api": {
                "public": "https://testnet.binancefuture.com/fapi/v1",
                "private": "https://testnet.binancefuture.com/fapi/v1"
            }
        }
    })
else:
    exchange = ccxt.binance({"enableRateLimit": True})
# ============================================

# ================= PORTFOLIO =================
portfolio_balance = START_BALANCE
peak_balance = START_BALANCE

daily_start_balance = START_BALANCE
current_day = date.today()
daily_trading_enabled = True

trade_log = []   # ← in-memory trade log
# ============================================

state = {
    s: {"pos":None,"entry":0.0,"qty":0.0,"partial":False,"hold":0}
    for s in SYMBOLS
}

# ================= HELPERS =================
def get_data(symbol, tf="1m", limit=250):
    return pd.DataFrame(
        exchange.fetch_ohlcv(symbol, tf, limit=limit),
        columns=["time","open","high","low","close","volume"]
    )


def get_5m_bias(symbol):
    df = get_data(symbol, "5m", 120)
    ema20 = ta.trend.EMAIndicator(df["close"],20).ema_indicator().iloc[-1]
    ema50 = ta.trend.EMAIndicator(df["close"],50).ema_indicator().iloc[-1]
    if ema20 > ema50:
        return "BULL"
    if ema20 < ema50:
        return "BEAR"
    return "FLAT"


def apply_strategy(df):
    df["ema_fast"] = ta.trend.EMAIndicator(df["close"],9).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"],21).ema_indicator()
    return df


def signal(df):
    p, c = df.iloc[-2], df.iloc[-1]
    if p["ema_fast"] <= p["ema_slow"] and c["ema_fast"] > c["ema_slow"]:
        return "BUY"
    if p["ema_fast"] >= p["ema_slow"] and c["ema_fast"] < c["ema_slow"]:
        return "SELL"
    return "HOLD"
# ============================================


def print_summary():
    if not trade_log:
        print("\n⚠️ No trades taken.")
        return

    trades = np.array(trade_log)

    wins = trades[trades > 0]
    losses = trades[trades <= 0]

    win_rate = len(wins) / len(trades) * 100
    avg_win = wins.mean() if len(wins) else 0
    avg_loss = abs(losses.mean()) if len(losses) else 0
    expectancy = (win_rate/100 * avg_win) - ((1-win_rate/100) * avg_loss)

    max_dd = (peak_balance - portfolio_balance) / peak_balance * 100
    total_pnl = portfolio_balance - START_BALANCE

    print("\n" + "="*45)
    print("📊 FINAL BOT SUMMARY")
    print("="*45)
    print(f"Start Balance      : {START_BALANCE:.2f}")
    print(f"Final Balance      : {portfolio_balance:.2f}")
    print(f"Total PnL          : {total_pnl:.2f}")
    print(f"Total Trades       : {len(trades)}")
    print(f"Win Rate           : {win_rate:.2f}%")
    print(f"Avg Win            : {avg_win:.2f}")
    print(f"Avg Loss           : {avg_loss:.2f}")
    print(f"Expectancy/Trade   : {expectancy:.4f}")
    print(f"Max Drawdown       : {max_dd:.2f}%")
    print("="*45 + "\n")


print("\n🚀 AI BOT RUNNING — CTRL+C TO STOP & PRINT SUMMARY\n")

while True:
    try:
        # ===== NEW DAY RESET =====
        if date.today() != current_day:
            current_day = date.today()
            daily_start_balance = portfolio_balance
            daily_trading_enabled = True
            print("\n🌅 New trading day — limits reset")

        daily_pnl_pct = (portfolio_balance - daily_start_balance) / daily_start_balance

        if daily_trading_enabled:
            if daily_pnl_pct >= DAILY_TARGET_PCT:
                daily_trading_enabled = False
                print("🎯 Daily target hit — trading paused")
            elif daily_pnl_pct <= DAILY_STOP_PCT:
                daily_trading_enabled = False
                print("🛑 Daily stop hit — trading paused")

        peak_balance = max(peak_balance, portfolio_balance)

        print(
            f"\n⏱ {datetime.now().strftime('%H:%M:%S')} "
            f"| Bal={portfolio_balance:.2f} "
            f"| DayPnL={daily_pnl_pct*100:.2f}% "
            f"| Trading={'ON' if daily_trading_enabled else 'OFF'}"
        )

        for symbol in SYMBOLS:
            s = state[symbol]
            bias = get_5m_bias(symbol)

            df = apply_strategy(get_data(symbol))
            if len(df) < MIN_CANDLES:
                continue

            df = prepare_features(df)
            price = df.iloc[-1]["close"]
            sig = signal(df)
            ai = predict_probability(df)

            size_factor = 1.0
            if symbol == "ETH/USDT" and state["BTC/USDT"]["pos"]:
                size_factor = ETH_CORRELATION_REDUCTION

            qty = (BASE_TRADE_SIZE / price) * size_factor

            # ===== ENTRY =====
            if daily_trading_enabled and s["pos"] is None and sig in ["BUY","SELL"]:
                if bias == "FLAT":
                    continue
                if sig == "BUY" and bias != "BULL":
                    continue
                if sig == "SELL" and bias != "BEAR":
                    continue
                if ai < 0.52:
                    continue

                s.update({
                    "pos": "LONG" if sig=="BUY" else "SHORT",
                    "entry": price,
                    "qty": qty,
                    "partial": False,
                    "hold": 0
                })
                print(f"🟢 ENTRY {symbol} | AI={ai:.2f} | Bias={bias}")

            # ===== EXIT =====
            elif s["pos"]:
                s["hold"] += 1
                pnl_pct = (
                    (price - s["entry"]) / s["entry"]
                    if s["pos"]=="LONG"
                    else (s["entry"] - price) / s["entry"]
                )

                if pnl_pct <= -STOP_LOSS or ai < AI_EXIT_WEAK or s["hold"] >= MAX_HOLD_CANDLES:
                    pnl = s["qty"] * pnl_pct * price
                    portfolio_balance += pnl
                    trade_log.append(pnl)
                    s.update({"pos":None,"qty":0,"partial":False,"hold":0})
                    print(f"❌ EXIT {symbol} | PnL={pnl:.2f}")

        time.sleep(SLEEP_SECONDS)

    except KeyboardInterrupt:
        print("\n🛑 BOT STOPPED BY USER")
        print_summary()
        sys.exit(0)

    except Exception as e:
        print("⚠️ ERROR:", e)
        print_summary()
        time.sleep(10)
