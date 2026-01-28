# ===================== IMPORTS =====================
import ccxt
import pandas as pd
import ta
import time
import csv
import os
import sys
import requests
import numpy as np
from datetime import datetime, timezone, date

from ai_predictor import prepare_features, predict_probability


# ===================== MODE =====================
LIVE_TRADING = False   # True = Binance Testnet
# ===============================================


# ===================== SETTINGS =====================
SYMBOLS = ["BTC/USDT", "ETH/USDT"]

START_BALANCE = 1000.0
BASE_TRADE_SIZE = 100.0

STOP_LOSS = 0.015
TRAILING_STOP = 0.006
MAX_HOLD_CANDLES = 30
AI_EXIT_WEAK = 0.40

MIN_CANDLES = 210
TARGET_VOL = 0.006
AI_RISING_DELTA = 0.02

# 🔒 RISK
MAX_DRAWDOWN_PCT = 0.08
DAILY_DRAWDOWN_PCT = 0.04

# 🔗 CORRELATION
ETH_CORRELATION_REDUCTION = 0.5

# 📆 WALK-FORWARD
MODEL_REFRESH_DAYS = 30
MODEL_INFO_FILE = "model_meta.txt"

# TELEGRAM (optional)
TELEGRAM_TOKEN = ""
TELEGRAM_CHAT_ID = ""
# ================================================


# ===================== EXCHANGE =====================
exchange = ccxt.binanceusdm({
    "enableRateLimit": True,
    "options": {"defaultType": "future"},
})
if LIVE_TRADING:
    exchange.set_sandbox_mode(True)
# ================================================


# ===================== PORTFOLIO =====================
portfolio_balance = START_BALANCE
peak_balance = START_BALANCE
daily_start_balance = START_BALANCE
current_day = date.today()


# ===================== FILES =====================
def init_csv(name, header):
    if not os.path.exists(name):
        with open(name, "w", newline="") as f:
            csv.writer(f).writerow(header)

init_csv("paper_trades.csv", [
    "time","symbol","side","qty","price","pnl","balance","reason"
])

init_csv("equity_curve.csv", [
    "time","balance","drawdown_pct"
])


def log_trade(symbol, side, qty, price, pnl, balance, reason):
    with open("paper_trades.csv", "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now(), symbol, side,
            round(qty,4), round(price,2),
            round(pnl,4), round(balance,2), reason
        ])


def log_equity(balance, peak):
    dd = (peak - balance) / peak if peak > 0 else 0
    with open("equity_curve.csv", "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now(), round(balance,2), round(dd*100,2)
        ])
    return dd


def tg(msg):
    if not TELEGRAM_TOKEN:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
            timeout=5
        )
    except:
        pass


# ===================== WALK-FORWARD CHECK =====================
def model_refresh_due():
    if not os.path.exists(MODEL_INFO_FILE):
        return True
    with open(MODEL_INFO_FILE, "r") as f:
        last = date.fromisoformat(f.read().strip())
    return (date.today() - last).days >= MODEL_REFRESH_DAYS


if model_refresh_due():
    print("⚠️ MODEL REFRESH DUE — retrain ai_predictor before live deployment")
    tg("⚠️ AI model refresh due (monthly)")


# ===================== MARKET =====================
def get_data(symbol, tf="1m", limit=250):
    return pd.DataFrame(
        exchange.fetch_ohlcv(symbol, tf, limit=limit),
        columns=["time","open","high","low","close","volume"]
    )


def get_atr_5m(symbol):
    df = get_data(symbol,"5m",50)
    atr = ta.volatility.AverageTrueRange(
        df["high"],df["low"],df["close"],14
    ).average_true_range().iloc[-1]
    return atr / df.iloc[-1]["close"]


# ===================== STRATEGY =====================
def apply_strategy(df):
    df["ema_fast"] = ta.trend.EMAIndicator(df["close"],9).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"],21).ema_indicator()
    df["adx"] = ta.trend.ADXIndicator(
        df["high"],df["low"],df["close"],14
    ).adx()
    return df


def generate_signal(df):
    p,c = df.iloc[-2], df.iloc[-1]
    if p["ema_fast"] <= p["ema_slow"] and c["ema_fast"] > c["ema_slow"]:
        return "BUY"
    if p["ema_fast"] >= p["ema_slow"] and c["ema_fast"] < c["ema_slow"]:
        return "SELL"
    return "HOLD"


# ===================== AI CALIBRATION =====================
ai_buffer = []

def calibrate_ai(raw):
    ai_buffer.append(raw)
    if len(ai_buffer) > 100:
        ai_buffer.pop(0)
    z = (raw - np.mean(ai_buffer)) / (np.std(ai_buffer)+1e-6)
    return float(np.clip(0.5 + z*0.15, 0.25, 0.75))


def ai_is_rising():
    return len(ai_buffer) > 5 and ai_buffer[-1] - ai_buffer[-5] > AI_RISING_DELTA


# ===================== STATE =====================
state = {
    s: {
        "pos":None,
        "entry":0.0,
        "qty":0.0,
        "partial":False,
        "hold":0
    }
    for s in SYMBOLS
}


# ===================== MAIN =====================
print("\n🚀 AI Trading Engine — Stable Paper Mode\n")

while True:
    try:
        # DAILY RESET
        if date.today() != current_day:
            daily_start_balance = portfolio_balance
            current_day = date.today()

        peak_balance = max(peak_balance, portfolio_balance)
        log_equity(portfolio_balance, peak_balance)

        if (peak_balance - portfolio_balance) / peak_balance >= MAX_DRAWDOWN_PCT:
            tg("🛑 MAX DRAWDOWN HIT — BOT STOPPED")
            print("🛑 Max drawdown hit")
            break

        print(f"\n⏱ {datetime.now().strftime('%H:%M:%S')} | Balance={portfolio_balance:.2f}")

        for symbol in SYMBOLS:
            coin = symbol.replace("/USDT","")
            s = state[symbol]

            df = apply_strategy(get_data(symbol))
            if len(df) < MIN_CANDLES:
                continue

            df = prepare_features(df)
            price = df.iloc[-1]["close"]
            signal = generate_signal(df)
            ai = calibrate_ai(predict_probability(df))

            print(f"🪙 {coin} | AI={ai:.2f} | Signal={signal}")

            # 🔗 Correlation sizing
            size_factor = 1.0
            if symbol == "ETH/USDT" and state["BTC/USDT"]["pos"]:
                size_factor = ETH_CORRELATION_REDUCTION

            qty = (BASE_TRADE_SIZE / price) * size_factor

            # ===== ENTRY =====
            if s["pos"] is None and signal in ["BUY","SELL"]:
                if ai >= 0.47 or ai_is_rising():   # 👈 relaxed
                    direction = "LONG" if signal == "BUY" else "SHORT"
                    s.update({
                        "pos": direction,
                        "entry": price,
                        "qty": qty,
                        "partial": False,
                        "hold": 0
                    })
                    print(f"🟢 ENTRY {coin} {direction} | AI={ai:.2f}")
                    tg(f"ENTRY {coin} {direction} AI={ai:.2f}")

            # ===== MANAGEMENT =====
            elif s["pos"]:
                s["hold"] += 1

                pnl_pct = (
                    (price - s["entry"]) / s["entry"]
                    if s["pos"] == "LONG"
                    else (s["entry"] - price) / s["entry"]
                )

                # PARTIAL EXIT
                if pnl_pct > 0.006 and not s["partial"]:
                    exit_qty = s["qty"] * 0.5
                    pnl = exit_qty * pnl_pct * price
                    portfolio_balance += pnl
                    s["qty"] -= exit_qty
                    s["partial"] = True
                    log_trade(symbol,"PARTIAL",exit_qty,price,pnl,portfolio_balance,"SCALE_OUT")
                    print(f"➖ PARTIAL EXIT {coin}")

                # FINAL EXIT
                if pnl_pct <= -STOP_LOSS or ai < AI_EXIT_WEAK or s["hold"] >= MAX_HOLD_CANDLES:
                    pnl = s["qty"] * pnl_pct * price
                    portfolio_balance += pnl
                    log_trade(symbol,"EXIT",s["qty"],price,pnl,portfolio_balance,"FINAL")
                    print(f"❌ EXIT {coin} | PnL={pnl:.2f}")
                    s.update({"pos":None,"qty":0,"partial":False,"hold":0})

        time.sleep(60)

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
        sys.exit(0)

    except Exception as e:
        print("Error:", e)
        time.sleep(10)
