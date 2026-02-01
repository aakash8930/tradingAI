# ===================== IMPORTS =====================
import ccxt, ta, time, csv, os, sys
import pandas as pd
from datetime import datetime, date
from ai_predictor import prepare_features, predict_signal

# ================= SETTINGS =================
SYMBOLS = ["BTC/USDT", "ETH/USDT"]

TREND_TF = "1h"
EXEC_TF = "5m"

START_BALANCE = 1000.0
BASE_RISK_PCT = 0.01          # 1% per trade (paper)
MAX_LEVERAGE = 3              # dynamic leverage cap

STOP_LOSS = 0.006             # 0.6%
TAKE_PROFIT = 0.010           # 1.0%
MAX_HOLD = 18                 # candles

DAILY_TARGET = 0.01
DAILY_STOP = -0.01

exchange = ccxt.binance({"enableRateLimit": True})

# ================= PORTFOLIO =================
balance = START_BALANCE
day_start_balance = START_BALANCE
current_day = date.today()

state = {
    s: {"side": None, "entry": 0, "qty": 0, "hold": 0}
    for s in SYMBOLS
}

# ================= CSV =================
if not os.path.exists("paper_trades.csv"):
    with open("paper_trades.csv", "w", newline="") as f:
        csv.writer(f).writerow(
            ["time","symbol","side","entry","exit","pnl","balance","reason"]
        )

def log_trade(sym, side, entry, exit_price, pnl, bal, reason):
    with open("paper_trades.csv", "a", newline="") as f:
        csv.writer(f).writerow(
            [datetime.now(), sym, side,
             round(entry,2), round(exit_price,2),
             round(pnl,2), round(bal,2), reason]
        )

# ================= TREND =================
def get_trend(symbol):
    df = pd.DataFrame(
        exchange.fetch_ohlcv(symbol, TREND_TF, limit=120),
        columns=["t","o","h","l","c","v"]
    )
    ema50 = ta.trend.EMAIndicator(df["c"], 50).ema_indicator().iloc[-1]
    ema200 = ta.trend.EMAIndicator(df["c"], 200).ema_indicator().iloc[-1]

    if ema50 > ema200:
        return "UP"
    elif ema50 < ema200:
        return "DOWN"
    return "RANGE"

# ================= MAIN LOOP =================
print("\n🚀 5m AI TREND PAPER TRADING STARTED\n")

while True:
    try:
        if date.today() != current_day:
            day_start_balance = balance
            current_day = date.today()
            print("\n📅 New trading day")

        daily_pnl = (balance - day_start_balance) / day_start_balance

        if daily_pnl >= DAILY_TARGET or daily_pnl <= DAILY_STOP:
            print("🛑 Daily limit hit — pausing 1h")
            time.sleep(3600)
            continue

        for symbol in SYMBOLS:
            s = state[symbol]
            coin = symbol.replace("/USDT","")

            trend = get_trend(symbol)

            df = pd.DataFrame(
                exchange.fetch_ohlcv(symbol, EXEC_TF, limit=300),
                columns=["t","open","high","low","close","volume"]
            )

            df = prepare_features(df)
            if len(df) < 50:
                continue

            price = df.iloc[-1]["close"]
            signal, confidence = predict_signal(df)

            print(f"🪙 {coin} | Price={price:.2f} | AI={confidence:.2f} | Trend={trend}")

            # ========== ENTRY ==========
            if s["side"] is None:
                if trend == "UP" and signal == "LONG":
                    risk = balance * BASE_RISK_PCT
                elif trend == "DOWN" and signal == "SHORT":
                    risk = balance * BASE_RISK_PCT
                else:
                    continue

                leverage = min(MAX_LEVERAGE, 1 + confidence)
                qty = (risk * leverage) / price

                s.update({
                    "side": signal,
                    "entry": price,
                    "qty": qty,
                    "hold": 0
                })
                print(f"🟢 ENTER {signal} {coin} | Lev={leverage:.1f}")

            # ========== MANAGEMENT ==========
            else:
                s["hold"] += 1

                pnl_pct = (
                    (price - s["entry"]) / s["entry"]
                    if s["side"] == "LONG"
                    else (s["entry"] - price) / s["entry"]
                )

                if pnl_pct <= -STOP_LOSS or pnl_pct >= TAKE_PROFIT or s["hold"] >= MAX_HOLD:
                    pnl = pnl_pct * s["qty"] * price
                    balance += pnl

                    log_trade(
                        symbol, s["side"], s["entry"],
                        price, pnl, balance, "EXIT"
                    )

                    print(f"❌ EXIT {coin} | PnL={pnl:.2f}")
                    s.update({"side":None,"qty":0,"hold":0})

        time.sleep(300)

    except KeyboardInterrupt:
        print("\n🛑 Bot stopped")
        sys.exit(0)

    except Exception as e:
        print("⚠️ Error:", e)
        time.sleep(30)
