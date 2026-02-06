# ===================== IMPORTS =====================
import ccxt
import ta
import time
import csv
import os
import sys
import pandas as pd
from datetime import datetime, date
from ai_predictor import prepare_features, predict_signal

# ===================== SETTINGS =====================
SYMBOL = "BTC/USDT"
EXEC_TF = "5m"
TREND_TF = "1h"

START_BALANCE = 1000.0
BASE_RISK = 0.01          # 1% risk per trade (paper)

STOP_LOSS = 0.006         # 0.6%
TAKE_PROFIT = 0.014       # 1.4%
TRAIL_AFTER = 0.006       # start trailing after +0.6%
TRAIL_DIST = 0.003        # trail distance
MAX_HOLD = 18             # candles

AI_ENTRY_MIN = 0.65       # 🔥 strict filter
AI_EXIT_MIN = 0.52

SLEEP_SECONDS = 300       # 5m

# ===================== EXCHANGE =====================
exchange = ccxt.binance({"enableRateLimit": True})

# ===================== PORTFOLIO =====================
balance = START_BALANCE
day_start_balance = START_BALANCE
current_day = date.today()

state = {
    "side": None,
    "entry": 0.0,
    "qty": 0.0,
    "best": 0.0,
    "hold": 0
}

# ===================== CSV LOGGING =====================
if not os.path.exists("paper_trades.csv"):
    with open("paper_trades.csv", "w", newline="") as f:
        csv.writer(f).writerow(
            ["time","symbol","side","entry","exit","pnl","balance","reason"]
        )

def log_trade(side, entry, exit_price, pnl, bal, reason):
    with open("paper_trades.csv", "a", newline="") as f:
        csv.writer(f).writerow(
            [
                datetime.now(),
                SYMBOL,
                side,
                round(entry,2),
                round(exit_price,2),
                round(pnl,2),
                round(bal,2),
                reason
            ]
        )

# ===================== TREND FILTER =====================
def get_trend():
    df = pd.DataFrame(
        exchange.fetch_ohlcv(SYMBOL, TREND_TF, limit=200),
        columns=["t","open","high","low","close","volume"]
    )
    ema50 = ta.trend.EMAIndicator(df["close"], 50).ema_indicator().iloc[-1]
    ema200 = ta.trend.EMAIndicator(df["close"], 200).ema_indicator().iloc[-1]
    return "UP" if ema50 > ema200 else "DOWN"

# ===================== MAIN LOOP =====================
print("\n🚀 PHASE-4 BTC-ONLY AI PAPER TRADING STARTED\n")

while True:
    try:
        # ---------- DAILY RESET ----------
        if date.today() != current_day:
            day_start_balance = balance
            current_day = date.today()
            print("\n📅 New trading day")

        # ---------- FETCH DATA ----------
        df = pd.DataFrame(
            exchange.fetch_ohlcv(SYMBOL, EXEC_TF, limit=300),
            columns=["t","open","high","low","close","volume"]
        )

        df = prepare_features(df)
        if len(df) < 120:
            time.sleep(SLEEP_SECONDS)
            continue

        price = df.iloc[-1]["close"]
        trend = get_trend()
        signal, ai = predict_signal(df)

        print(
            f"{datetime.now().strftime('%H:%M:%S')} | "
            f"BTC={price:.2f} | AI={ai:.2f} | Trend={trend} | Pos={state['side']}"
        )

        # ================= ENTRY =================
        if state["side"] is None:
            if trend != "UP":
                time.sleep(SLEEP_SECONDS)
                continue

            if signal != "LONG" or ai < AI_ENTRY_MIN:
                time.sleep(SLEEP_SECONDS)
                continue

            risk_amount = balance * BASE_RISK
            qty = risk_amount / price

            state.update({
                "side": "LONG",
                "entry": price,
                "qty": qty,
                "best": price,
                "hold": 0
            })

            print(f"🟢 ENTER LONG BTC | AI={ai:.2f}")

        # ================= MANAGEMENT =================
        else:
            state["hold"] += 1

            pnl_pct = (price - state["entry"]) / state["entry"]
            state["best"] = max(state["best"], price)

            trail_hit = (
                pnl_pct > TRAIL_AFTER
                and price <= state["best"] * (1 - TRAIL_DIST)
            )

            exit_now = (
                pnl_pct <= -STOP_LOSS
                or pnl_pct >= TAKE_PROFIT
                or trail_hit
                or ai < AI_EXIT_MIN
                or state["hold"] >= MAX_HOLD
            )

            if exit_now:
                pnl = pnl_pct * state["qty"] * price
                balance += pnl

                log_trade(
                    "LONG",
                    state["entry"],
                    price,
                    pnl,
                    balance,
                    "EXIT"
                )

                print(f"❌ EXIT BTC | PnL={pnl:.2f} | Bal={balance:.2f}")

                state.update({
                    "side": None,
                    "qty": 0.0,
                    "hold": 0
                })

        time.sleep(SLEEP_SECONDS)

    except KeyboardInterrupt:
        print("\n🛑 Bot stopped manually")
        sys.exit(0)

    except Exception as e:
        print("⚠️ Error:", e)
        time.sleep(30)
