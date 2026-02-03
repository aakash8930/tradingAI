# main.py

import ccxt
import time
import csv
import os
import sys
import pandas as pd
from datetime import datetime
from ai_predictor import prepare_features, predict_probability

# ================= SETTINGS =================
SYMBOLS = ["BTC/USDT", "ETH/USDT"]   # add more coins safely later
TIMEFRAME = "5m"

START_BALANCE = 1000.0
BASE_TRADE_USD = 150.0

STOP_LOSS = 0.012
MAX_HOLD = 12
AI_LONG = 0.60
AI_SHORT = 0.40

# ================= EXCHANGE =================
exchange = ccxt.binance({"enableRateLimit": True})

# ================= STATE =================
balance = START_BALANCE

state = {
    s: {"pos": None, "entry": 0.0, "qty": 0.0, "hold": 0}
    for s in SYMBOLS
}

# ================= CSV =================
if not os.path.exists("paper_trades.csv"):
    with open("paper_trades.csv", "w", newline="") as f:
        csv.writer(f).writerow(
            ["timestamp","symbol","side","entry","exit","pnl","balance","reason"]
        )

if not os.path.exists("diagnostics.csv"):
    with open("diagnostics.csv", "w", newline="") as f:
        csv.writer(f).writerow(
            ["time","symbol","price","ai","position","balance"]
        )

def log_trade(row):
    with open("paper_trades.csv", "a", newline="") as f:
        csv.writer(f).writerow(row)

def log_diag(row):
    with open("diagnostics.csv", "a", newline="") as f:
        csv.writer(f).writerow(row)

print("\n🚀 5m AI DIAGNOSTIC PAPER TRADING STARTED\n")

# ================= LOOP =================
while True:
    try:
        for symbol in SYMBOLS:
            s = state[symbol]
            coin = symbol.replace("/USDT", "")

            df = pd.DataFrame(
                exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=200),
                columns=["time","open","high","low","close","volume"]
            )

            df = prepare_features(df)
            if len(df) < 50:
                continue

            price = df.iloc[-1]["close"]
            ai = predict_probability(df)

            print(
                f"{coin} | Price={price:.2f} | AI={ai:.2f} | "
                f"Pos={s['pos']}"
            )

            log_diag([
                datetime.now(), symbol, round(price,2),
                round(ai,3), s["pos"], round(balance,2)
            ])

            qty = BASE_TRADE_USD / price

            # -------- ENTRY --------
            if s["pos"] is None:
                if ai >= AI_LONG:
                    s.update({"pos":"LONG","entry":price,"qty":qty,"hold":0})
                    print(f"🟢 ENTER LONG {coin}")

                elif ai <= AI_SHORT:
                    s.update({"pos":"SHORT","entry":price,"qty":qty,"hold":0})
                    print(f"🔴 ENTER SHORT {coin}")

            # -------- MANAGEMENT --------
            else:
                s["hold"] += 1

                pnl_pct = (
                    (price - s["entry"]) / s["entry"]
                    if s["pos"] == "LONG"
                    else (s["entry"] - price) / s["entry"]
                )

                if pnl_pct <= -STOP_LOSS or s["hold"] >= MAX_HOLD:
                    pnl = pnl_pct * s["qty"] * price
                    balance += pnl

                    log_trade([
                        datetime.now(), symbol, s["pos"],
                        round(s["entry"],2), round(price,2),
                        round(pnl,2), round(balance,2), "EXIT"
                    ])

                    print(f"❌ EXIT {coin} | PnL={pnl:.2f}")

                    s.update({"pos":None,"qty":0.0,"hold":0})

        time.sleep(300)

    except KeyboardInterrupt:
        print("\n🛑 Bot stopped")
        sys.exit(0)

    except Exception as e:
        print("⚠️ Error:", e)
        time.sleep(30)
