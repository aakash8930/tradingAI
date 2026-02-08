# main.py

import ccxt, ta, time, csv, os, sys
import pandas as pd
from datetime import datetime
from ai_predictor import prepare_features, predict_signal

# ================= CONFIG =================
SYMBOL = "BTC/USDT"
TF = "5m"

START_BALANCE = 1000.0
BASE_RISK = 0.01

# UP-TREND SETTINGS
UP_AI_MIN = 0.65
UP_TP = 0.012
UP_SL = 0.006
UP_MAX_HOLD = 18

# DOWN-TREND MEAN REVERSION
DOWN_AI_MIN = 0.72
DOWN_RSI_MAX = 32
DOWN_RISK_MULT = 0.4
DOWN_TP = 0.004
DOWN_SL = 0.003
DOWN_MAX_HOLD = 6
COOLDOWN_CANDLES = 6

exchange = ccxt.binance({"enableRateLimit": True})

balance = START_BALANCE
position = None
cooldown = 0

# ================= CSV =================
if not os.path.exists("paper_trades.csv"):
    with open("paper_trades.csv", "w", newline="") as f:
        csv.writer(f).writerow(
            ["time", "symbol", "side", "entry", "exit", "pnl", "balance", "reason"]
        )


def log_trade(entry, exit_p, pnl, bal, reason):
    with open("paper_trades.csv", "a", newline="") as f:
        csv.writer(f).writerow(
            [
                datetime.now(),
                SYMBOL,
                "LONG",
                round(entry, 2),
                round(exit_p, 2),
                round(pnl, 2),
                round(bal, 2),
                reason,
            ]
        )


# ================= TREND =================
def get_trend():
    df = pd.DataFrame(
        exchange.fetch_ohlcv(SYMBOL, "1h", limit=200),
        columns=["t", "o", "h", "l", "c", "v"],
    )
    ema50 = ta.trend.EMAIndicator(df["c"], 50).ema_indicator().iloc[-1]
    ema200 = ta.trend.EMAIndicator(df["c"], 200).ema_indicator().iloc[-1]
    return "UP" if ema50 > ema200 else "DOWN"


print("\n🚀 PHASE-4.1 BTC-ONLY AI PAPER TRADING STARTED\n")

# ================= MAIN LOOP =================
while True:
    try:
        if cooldown > 0:
            cooldown -= 1

        df = pd.DataFrame(
            exchange.fetch_ohlcv(SYMBOL, TF, limit=300),
            columns=["t", "open", "high", "low", "close", "volume"],
        )

        df = prepare_features(df)
        if len(df) < 120:
            time.sleep(300)
            continue

        price = df.iloc[-1]["close"]
        rsi = df.iloc[-1]["rsi"]
        _, ai = predict_signal(df)

        trend = get_trend()

        ts = datetime.now().strftime("%H:%M:%S")
        print(
            f"{ts} | BTC={price:.2f} | AI={ai:.2f} | Trend={trend} | Pos={position is not None}"
        )

        # ================= ENTRY =================
        if position is None and cooldown == 0:
            # ---- UP TREND ----
            if trend == "UP" and ai >= UP_AI_MIN:
                risk = balance * BASE_RISK
                qty = risk / price
                position = {
                    "entry": price,
                    "qty": qty,
                    "best": price,
                    "hold": 0,
                    "tp": UP_TP,
                    "sl": UP_SL,
                    "max_hold": UP_MAX_HOLD,
                }
                print("🟢 ENTER LONG (UP TREND)")

            # ---- DOWN TREND MEAN REVERSION ----
            elif trend == "DOWN" and ai >= DOWN_AI_MIN and rsi <= DOWN_RSI_MAX:
                risk = balance * BASE_RISK * DOWN_RISK_MULT
                qty = risk / price
                position = {
                    "entry": price,
                    "qty": qty,
                    "best": price,
                    "hold": 0,
                    "tp": DOWN_TP,
                    "sl": DOWN_SL,
                    "max_hold": DOWN_MAX_HOLD,
                }
                print("🟡 ENTER LONG (MEAN REVERSION)")

        # ================= MANAGEMENT =================
        elif position:
            position["hold"] += 1
            position["best"] = max(position["best"], price)

            pnl_pct = (price - position["entry"]) / position["entry"]

            exit_now = (
                pnl_pct >= position["tp"]
                or pnl_pct <= -position["sl"]
                or position["hold"] >= position["max_hold"]
            )

            if exit_now:
                pnl = pnl_pct * position["qty"] * price
                balance += pnl

                log_trade(position["entry"], price, pnl, balance, "EXIT")

                print(f"❌ EXIT | PnL={pnl:.2f} | Bal={balance:.2f}")

                position = None
                cooldown = COOLDOWN_CANDLES

        time.sleep(300)

    except KeyboardInterrupt:
        print("\n🛑 Bot stopped")
        sys.exit(0)

    except Exception as e:
        print("⚠️ Error:", e)
        time.sleep(60)
