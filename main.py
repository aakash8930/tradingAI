import ccxt, ta, time, csv, os, sys
import pandas as pd
from datetime import datetime
from ai_predictor import prepare_features, predict_signal

# ================= SETTINGS =================
SYMBOLS = ["BTC/USDT", "ETH/USDT"]

EXEC_TF = "5m"
START_BALANCE = 1000.0
BASE_RISK = 0.01          # 1% risk per trade

STOP_LOSS = 0.006         # 0.6%
TAKE_PROFIT = 0.012       # 1.2%
TRAIL_AFTER = 0.004       # trail only after +0.4%
TRAIL_PCT = 0.002         # 0.2% trailing
MAX_HOLD = 18             # candles

exchange = ccxt.binance({"enableRateLimit": True})

# ================= STATE =================
balance = START_BALANCE
state = {
    s: {"side": None, "entry": 0.0, "qty": 0.0, "best": 0.0, "hold": 0}
    for s in SYMBOLS
}

# ================= CSV =================
if not os.path.exists("paper_trades.csv"):
    with open("paper_trades.csv", "w", newline="") as f:
        csv.writer(f).writerow(
            ["time","symbol","side","entry","exit","pnl","balance","reason"]
        )

def log_trade(sym, side, entry, exit_p, pnl, bal, reason):
    with open("paper_trades.csv", "a", newline="") as f:
        csv.writer(f).writerow(
            [datetime.now(), sym, side,
             round(entry, 2), round(exit_p, 2),
             round(pnl, 2), round(bal, 2), reason]
        )

# ================= MAIN LOOP =================
print("\n🚀 PHASE-2 AI PAPER TRADING STARTED\n")

while True:
    try:
        btc_ai = 0.5  # BTC bias reference

        for symbol in SYMBOLS:
            df = pd.DataFrame(
                exchange.fetch_ohlcv(symbol, EXEC_TF, limit=300),
                columns=["t","open","high","low","close","volume"]
            )

            df = prepare_features(df)
            if len(df) < 100:
                continue

            price = df.iloc[-1]["close"]
            vol = df.iloc[-1]["vol"]

            side, ai = predict_signal(df)

            if symbol == "BTC/USDT":
                btc_ai = ai

            s = state[symbol]

            print(f"{symbol} | Price={price:.2f} | AI={ai:.2f} | Pos={s['side']}")

            # ================= ENTRY =================
            if s["side"] is None:
                # ETH trades only when BTC AI is calm
                if symbol == "ETH/USDT" and btc_ai > 0.62:
                    continue

                if side is None:
                    continue

                # volatility filter (anti-chop)
                if vol < df["vol"].quantile(0.2):
                    continue

                risk = balance * BASE_RISK
                if symbol == "ETH/USDT":
                    risk *= 0.4  # ETH size reduction

                qty = risk / price

                s.update({
                    "side": side,
                    "entry": price,
                    "qty": qty,
                    "best": price,
                    "hold": 0
                })

                print(f"🟢 ENTER {side} {symbol}")

            # ================= MANAGEMENT =================
            else:
                s["hold"] += 1

                pnl_pct = (
                    (price - s["entry"]) / s["entry"]
                    if s["side"] == "LONG"
                    else (s["entry"] - price) / s["entry"]
                )

                # update best price
                if s["side"] == "LONG":
                    s["best"] = max(s["best"], price)
                    trail_hit = price <= s["best"] * (1 - TRAIL_PCT)
                else:
                    s["best"] = min(s["best"], price)
                    trail_hit = price >= s["best"] * (1 + TRAIL_PCT)

                exit_now = (
                    pnl_pct <= -STOP_LOSS
                    or pnl_pct >= TAKE_PROFIT
                    or (pnl_pct > TRAIL_AFTER and trail_hit)
                    or ai < 0.55
                    or s["hold"] >= MAX_HOLD
                )

                if exit_now:
                    pnl = pnl_pct * s["qty"] * price
                    balance += pnl

                    log_trade(
                        symbol, s["side"],
                        s["entry"], price,
                        pnl, balance, "EXIT"
                    )

                    print(f"❌ EXIT {symbol} | PnL={pnl:.2f}")

                    s.update({"side": None, "qty": 0.0, "hold": 0})

        time.sleep(300)

    except KeyboardInterrupt:
        print("\n🛑 Bot stopped")
        sys.exit(0)

    except Exception as e:
        print("⚠️ Error:", e)
        time.sleep(30)
