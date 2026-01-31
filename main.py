# main.py
import ccxt, ta, time, csv, os, sys
import pandas as pd
from datetime import datetime, date
from ai_predictor import prepare_features, predict_probability

SYMBOLS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME = "5m"

START_BALANCE = 1000.0
BASE_TRADE_SIZE = 200.0

STOP_LOSS = 0.006
TAKE_PROFIT = 0.012
MAX_HOLD = 24

exchange = ccxt.binance({"enableRateLimit": True})

balance = START_BALANCE
current_day = date.today()

state = {
    s: {"pos":None,"entry":0,"qty":0,"hold":0}
    for s in SYMBOLS
}

if not os.path.exists("paper_trades.csv"):
    with open("paper_trades.csv","w",newline="") as f:
        csv.writer(f).writerow(
            ["time","symbol","side","qty","price","pnl","balance","reason"]
        )

def log_trade(sym, side, qty, price, pnl, bal, reason):
    with open("paper_trades.csv","a",newline="") as f:
        csv.writer(f).writerow(
            [datetime.now(),sym,side,qty,price,round(pnl,2),round(bal,2),reason]
        )

print("\n🚀 5m AI TREND PAPER TRADING STARTED\n")

while True:
    try:
        for symbol in SYMBOLS:
            s = state[symbol]
            coin = symbol.replace("/USDT","")

            df5 = pd.DataFrame(
                exchange.fetch_ohlcv(symbol, "5m", limit=300),
                columns=["t","o","h","l","c","v"]
            )
            df5.columns = ["time","open","high","low","close","volume"]

            df1h = pd.DataFrame(
                exchange.fetch_ohlcv(symbol, "1h", limit=100),
                columns=["t","o","h","l","c","v"]
            )
            df1h.columns = ["time","open","high","low","close","volume"]

            df5 = prepare_features(df5)
            if len(df5) < 50:
                continue

            ema_1h = ta.trend.EMAIndicator(df1h["close"], 50).ema_indicator().iloc[-1]
            trend_ok = df5.iloc[-1]["close"] > ema_1h

            price = df5.iloc[-1]["close"]
            ai = predict_probability(df5)

            print(f"🪙 {coin} | Price={price:.2f} | AI={ai:.2f} | Trend={'UP' if trend_ok else 'DOWN'}")

            qty = BASE_TRADE_SIZE / price

            if s["pos"] is None and ai >= 0.60 and trend_ok:
                s.update({"pos":"LONG","entry":price,"qty":qty,"hold":0})
                print(f"🟢 LONG {coin}")

            elif s["pos"]:
                s["hold"] += 1
                pnl_pct = (price - s["entry"]) / s["entry"]

                if pnl_pct >= TAKE_PROFIT or pnl_pct <= -STOP_LOSS or s["hold"] >= MAX_HOLD:
                    pnl = s["qty"] * pnl_pct * price
                    balance += pnl
                    log_trade(symbol,"EXIT",s["qty"],price,pnl,balance,"EXIT")
                    print(f"❌ EXIT {coin} | PnL={pnl:.2f}")
                    s.update({"pos":None,"qty":0,"hold":0})

        time.sleep(300)

    except KeyboardInterrupt:
        print("\n🛑 Stopped")
        sys.exit(0)
