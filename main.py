import ccxt, ta, time, csv, os, sys
import pandas as pd
from datetime import datetime, date
from ai_predictor import prepare_features, predict_signal

SYMBOLS = ["BTC/USDT","ETH/USDT"]
TIMEFRAME = "5m"

START_BALANCE = 1000.0
RISK_PCT = 0.01

STOP_LOSS = 0.006
TAKE_PROFIT = 0.012
MAX_HOLD = 18

exchange = ccxt.binance({"enableRateLimit": True})

balance = START_BALANCE
day_start = START_BALANCE
current_day = date.today()

state = {s:{"side":None,"entry":0,"qty":0,"hold":0} for s in SYMBOLS}

if not os.path.exists("paper_trades.csv"):
    with open("paper_trades.csv","w",newline="") as f:
        csv.writer(f).writerow(
            ["time","symbol","side","entry","exit","pnl","balance"]
        )

def log_trade(sym, side, entry, exit_p, pnl):
    with open("paper_trades.csv","a",newline="") as f:
        csv.writer(f).writerow(
            [datetime.now(),sym,side,round(entry,2),round(exit_p,2),
             round(pnl,2),round(balance,2)]
        )

print("\n🚀 5m AI PAPER TRADING STARTED\n")

while True:
    try:
        if date.today() != current_day:
            day_start = balance
            current_day = date.today()

        for symbol in SYMBOLS:
            s = state[symbol]

            df = pd.DataFrame(
                exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=200),
                columns=["t","open","high","low","close","volume"]
            )

            df = prepare_features(df)
            if len(df) < 50:
                continue

            price = df.iloc[-1]["close"]
            signal, conf = predict_signal(df)

            print(f"{symbol} | Price={price:.2f} | AI={conf:.2f} | Pos={s['side']}")

            if s["side"] is None and signal != "NONE":
                risk = balance * RISK_PCT
                qty = risk / price
                s.update({"side":signal,"entry":price,"qty":qty,"hold":0})
                print(f"🟢 ENTER {signal} {symbol}")

            elif s["side"]:
                s["hold"] += 1
                pnl_pct = (
                    (price - s["entry"]) / s["entry"]
                    if s["side"]=="LONG"
                    else (s["entry"] - price) / s["entry"]
                )

                if pnl_pct <= -STOP_LOSS or pnl_pct >= TAKE_PROFIT or s["hold"] >= MAX_HOLD:
                    pnl = pnl_pct * s["qty"] * price
                    balance += pnl
                    log_trade(symbol,s["side"],s["entry"],price,pnl)
                    print(f"❌ EXIT {symbol} | PnL={pnl:.2f}")
                    s.update({"side":None,"qty":0,"hold":0})

        time.sleep(300)

    except KeyboardInterrupt:
        print("\n🛑 Stopped")
        sys.exit(0)
