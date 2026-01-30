# ===================== IMPORTS =====================
import ccxt
import ta
import time
import csv
import os
import sys
import pandas as pd
from datetime import datetime, date
from ai_predictor import prepare_features, predict_probability

# ================= SETTINGS =================
SYMBOLS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME = "5m"

START_BALANCE = 1000.0
BASE_TRADE_SIZE = 150.0

STOP_LOSS = 0.012
TAKE_PARTIAL = 0.006
MAX_HOLD = 12

DAILY_TARGET = 0.01     # +1%
DAILY_STOP = -0.01      # -1%

ETH_REDUCE = 0.5

# ================= EXCHANGE =================
exchange = ccxt.binance({"enableRateLimit": True})

# ================= PORTFOLIO =================
balance = START_BALANCE
day_start_balance = START_BALANCE
current_day = date.today()

state = {
    s: {"pos": None, "entry": 0.0, "qty": 0.0, "partial": False, "hold": 0}
    for s in SYMBOLS
}

# ================= CSV =================
if not os.path.exists("paper_trades.csv"):
    with open("paper_trades.csv", "w", newline="") as f:
        csv.writer(f).writerow(
            ["time", "symbol", "side", "qty", "price", "pnl", "balance", "reason"]
        )

def log_trade(sym, side, qty, price, pnl, bal, reason):
    with open("paper_trades.csv", "a", newline="") as f:
        csv.writer(f).writerow(
            [datetime.now(), sym, side, round(qty, 5),
             round(price, 2), round(pnl, 2), round(bal, 2), reason]
        )

# ================= MAIN LOOP =================
print("\n🚀 5m AI Paper Trading Started\n")

while True:
    try:
        # ---------- DAILY RESET ----------
        if date.today() != current_day:
            day_start_balance = balance
            current_day = date.today()
            print("\n📅 New trading day started")

        daily_pnl = (balance - day_start_balance) / day_start_balance

        # ---------- DAILY CONTROLS ----------
        if daily_pnl >= DAILY_TARGET:
            print(f"🎯 Daily target hit (+{daily_pnl*100:.2f}%) — pausing 1h")
            time.sleep(3600)
            continue

        if daily_pnl <= DAILY_STOP:
            print(f"🛑 Daily stop hit ({daily_pnl*100:.2f}%) — pausing 1h")
            time.sleep(3600)
            continue

        print(
            f"\n⏱ {datetime.now().strftime('%H:%M:%S')} | "
            f"Balance={balance:.2f} | DailyPnL={daily_pnl*100:.2f}%"
        )

        # ---------- PER SYMBOL ----------
        for symbol in SYMBOLS:
            s = state[symbol]
            coin = symbol.replace("/USDT", "")

            df = pd.DataFrame(
                exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=300),
                columns=["time", "open", "high", "low", "close", "volume"]
            )

            df = prepare_features(df)
            if len(df) < 50:
                print(f"🪙 {coin} | warming data...")
                continue

            price = df.iloc[-1]["close"]
            ai = predict_probability(df)

            print(
                f"🪙 {coin} | Price={price:.2f} | AI={ai:.2f} | "
                f"Pos={'YES' if s['pos'] else 'NO'}"
            )

            # ---------- POSITION SIZE ----------
            size_factor = (
                ETH_REDUCE
                if symbol == "ETH/USDT" and state["BTC/USDT"]["pos"]
                else 1.0
            )
            qty = (BASE_TRADE_SIZE / price) * size_factor

            # ---------- ENTRY ----------
            if s["pos"] is None and ai > 0.52:
                s.update({
                    "pos": "LONG",
                    "entry": price,
                    "qty": qty,
                    "partial": False,
                    "hold": 0
                })
                print(f"🟢 LONG {coin} | AI={ai:.2f}")

            # ---------- MANAGEMENT ----------
            elif s["pos"]:
                s["hold"] += 1
                pnl_pct = (price - s["entry"]) / s["entry"]

                # ----- PARTIAL EXIT -----
                if pnl_pct >= TAKE_PARTIAL and not s["partial"]:
                    exit_qty = s["qty"] * 0.5
                    pnl = exit_qty * pnl_pct * price
                    balance += pnl
                    s["qty"] -= exit_qty
                    s["partial"] = True
                    log_trade(symbol, "PARTIAL", exit_qty, price, pnl, balance, "TP1")
                    print(f"➖ PARTIAL EXIT {coin}")

                # ----- FINAL EXIT -----
                if pnl_pct <= -STOP_LOSS or s["hold"] >= MAX_HOLD:
                    pnl = s["qty"] * pnl_pct * price
                    balance += pnl
                    log_trade(symbol, "EXIT", s["qty"], price, pnl, balance, "EXIT")
                    print(f"❌ EXIT {coin} | PnL={pnl:.2f}")
                    s.update({"pos": None, "qty": 0.0, "partial": False, "hold": 0})

        print("⏳ Waiting for next 5m candle...\n")
        time.sleep(300)

    except KeyboardInterrupt:
        print("\n🛑 Bot stopped manually")
        sys.exit(0)

    except Exception as e:
        print("⚠️ Error:", e)
        time.sleep(30)
