import ccxt, ta, time, csv, os, sys
import pandas as pd
from datetime import datetime, date
from ai_predictor import prepare_features, predict_signal

# ================== CONFIG ==================
SYMBOLS = ["BTC/USDT", "ETH/USDT"]
EXEC_TF = "5m"

START_BALANCE = 1000.0
BASE_RISK = 0.01            # 1% base risk

STOP_LOSS = 0.006           # hard stop
TRAIL_START = 0.006         # start trailing after +0.6%
TRAIL_GAP = 0.004           # initial trailing distance
MAX_HOLD = 30               # extended hold for BTC

BTC_AI_MIN = 0.62           # BTC confidence gate
ETH_AI_MIN = 0.58           # ETH entry gate
ETH_RISK_MULT = 0.35        # ETH size reduction

exchange = ccxt.binance({"enableRateLimit": True})

# ================== STATE ==================
balance = START_BALANCE
state = {
    s: {"side": None, "entry": 0, "qty": 0, "best": 0, "hold": 0}
    for s in SYMBOLS
}

# ================== CSV ==================
if not os.path.exists("paper_trades.csv"):
    with open("paper_trades.csv", "w", newline="") as f:
        csv.writer(f).writerow(
            ["time","symbol","side","entry","exit","pnl","balance","reason"]
        )

def log(sym, side, entry, exit_p, pnl, bal, reason):
    with open("paper_trades.csv", "a", newline="") as f:
        csv.writer(f).writerow(
            [datetime.now(), sym, side,
             round(entry,2), round(exit_p,2),
             round(pnl,2), round(bal,2), reason]
        )

print("\n🚀 PHASE-3 BTC-LED AI PAPER TRADING STARTED\n")

# ================== MAIN LOOP ==================
while True:
    try:
        btc_ai = 0.5

        for symbol in SYMBOLS:
            df = pd.DataFrame(
                exchange.fetch_ohlcv(symbol, EXEC_TF, limit=300),
                columns=["t","open","high","low","close","volume"]
            )

            df = prepare_features(df)
            if len(df) < 120:
                continue

            price = df.iloc[-1]["close"]
            side, ai = predict_signal(df)

            if symbol == "BTC/USDT":
                btc_ai = ai

            s = state[symbol]

            print(f"{symbol} | Price={price:.2f} | AI={ai:.2f} | Pos={s['side']}")

            # ========== ENTRY ==========
            if s["side"] is None:
                if side is None:
                    continue

                # ---- BTC FILTER ----
                if symbol == "BTC/USDT" and ai < BTC_AI_MIN:
                    continue

                # ---- ETH FILTER ----
                if symbol == "ETH/USDT":
                    if btc_ai < BTC_AI_MIN or ai < ETH_AI_MIN:
                        continue

                risk = balance * BASE_RISK
                if symbol == "ETH/USDT":
                    risk *= ETH_RISK_MULT

                qty = risk / price

                s.update({
                    "side": side,
                    "entry": price,
                    "qty": qty,
                    "best": price,
                    "hold": 0
                })

                print(f"🟢 ENTER {side} {symbol}")

            # ========== MANAGEMENT ==========
            else:
                s["hold"] += 1

                pnl_pct = (
                    (price - s["entry"]) / s["entry"]
                    if s["side"] == "LONG"
                    else (s["entry"] - price) / s["entry"]
                )

                # Track best price
                if s["side"] == "LONG":
                    s["best"] = max(s["best"], price)
                else:
                    s["best"] = min(s["best"], price)

                # ---- Dynamic trailing ----
                trail_hit = False
                if pnl_pct >= TRAIL_START:
                    trail_gap = TRAIL_GAP + (pnl_pct - TRAIL_START) * 0.5
                    if s["side"] == "LONG":
                        trail_hit = price <= s["best"] * (1 - trail_gap)
                    else:
                        trail_hit = price >= s["best"] * (1 + trail_gap)

                exit_now = (
                    pnl_pct <= -STOP_LOSS
                    or trail_hit
                    or (symbol == "ETH/USDT" and ai < 0.50)
                    or (symbol == "BTC/USDT" and ai < 0.45)
                    or s["hold"] >= MAX_HOLD
                )

                if exit_now:
                    pnl = pnl_pct * s["qty"] * price
                    balance += pnl

                    log(symbol, s["side"], s["entry"], price, pnl, balance, "EXIT")
                    print(f"❌ EXIT {symbol} | PnL={pnl:.2f}")

                    s.update({"side": None, "qty": 0, "hold": 0})

        time.sleep(300)

    except KeyboardInterrupt:
        print("\n🛑 Bot stopped")
        sys.exit(0)

    except Exception as e:
        print("⚠️ Error:", e)
        time.sleep(30)
