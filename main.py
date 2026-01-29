import ccxt, ta, time, csv, os, sys
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

DAILY_TARGET = 0.01
DAILY_STOP = -0.01

ETH_REDUCE = 0.5

exchange = ccxt.binance()

balance = START_BALANCE
day_start_balance = START_BALANCE
current_day = date.today()

state = {
    s: {"pos":None,"entry":0,"qty":0,"partial":False,"hold":0}
    for s in SYMBOLS
}

# ================= CSV =================
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

# ================= LOOP =================
print("\n🚀 5m AI Paper Trading Started\n")

while True:
    try:
        if date.today() != current_day:
            day_start_balance = balance
            current_day = date.today()

        daily_pnl = (balance - day_start_balance) / day_start_balance

        if daily_pnl >= DAILY_TARGET:
            print("🎯 Daily target hit — pausing")
            time.sleep(3600)
            continue

        if daily_pnl <= DAILY_STOP:
            print("🛑 Daily stop hit — pausing")
            time.sleep(3600)
            continue

        for symbol in SYMBOLS:
            s = state[symbol]
            coin = symbol.replace("/USDT","")

            df = pd.DataFrame(
                exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=300),
                columns=["t","open","high","low","close","vol"]
            )

            df = prepare_features(df)
            if len(df) < 50:
                continue

            price = df.iloc[-1]["close"]
            ai = predict_probability(df)

            # Position size
            size_factor = ETH_REDUCE if symbol=="ETH/USDT" and state["BTC/USDT"]["pos"] else 1
            qty = (BASE_TRADE_SIZE / price) * size_factor

            # ENTRY
            if s["pos"] is None and ai > 0.48:
                s.update({"pos":"LONG","entry":price,"qty":qty,"partial":False,"hold":0})
                print(f"🟢 LONG {coin} AI={ai:.2f}")

            # MANAGEMENT
            elif s["pos"]:
                s["hold"] += 1
                pnl_pct = (price - s["entry"]) / s["entry"]

                if pnl_pct >= TAKE_PARTIAL and not s["partial"]:
                    exit_qty = s["qty"] * 0.5
                    pnl = exit_qty * pnl_pct * price
                    balance += pnl
                    s["qty"] -= exit_qty
                    s["partial"] = True
                    log_trade(symbol,"PARTIAL",exit_qty,price,pnl,balance,"TP1")
                    print(f"➖ PARTIAL {coin}")

                if pnl_pct <= -STOP_LOSS or s["hold"] >= MAX_HOLD:
                    pnl = s["qty"] * pnl_pct * price
                    balance += pnl
                    log_trade(symbol,"EXIT",s["qty"],price,pnl,balance,"EXIT")
                    print(f"❌ EXIT {coin} | PnL={pnl:.2f}")
                    s.update({"pos":None,"qty":0,"partial":False,"hold":0})

        time.sleep(300)

    except KeyboardInterrupt:
        print("\n🛑 Stopped")
        sys.exit(0)
