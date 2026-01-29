# backtest.py

import ccxt
import pandas as pd
import ta
import numpy as np

from ai_predictor import prepare_features, predict_probability

# ================= SETTINGS =================
SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT",
    "SOL/USDT", "AVAX/USDT", "MATIC/USDT",
    "DOGE/USDT", "PEPE/USDT"
]

TIMEFRAME = "1m"

START_BALANCE = 1000.0
BASE_RISK_PCT = 0.01          # 1% risk per trade (base)

STOP_LOSS = 0.012
TAKE_PROFIT = 0.040
TRAILING_STOP = 0.008
MAX_HOLD_CANDLES = 30

MIN_CANDLES = 210
AI_WINDOW = 120

# ---- Portfolio Risk ----
MAX_PORTFOLIO_DD = 0.06       # 6% hard stop
DD_LEVELS = [
    (0.01, 1.0),              # <1% DD → full risk
    (0.02, 0.7),              # 1–2% DD → 70%
    (0.03, 0.4),              # 2–3% DD → 40%
    (0.99, 0.0),              # >3% DD → stop new trades
]

MAX_COIN_EXPOSURE = 0.20      # 20% max capital per coin
TARGET_VOL = 0.006
MIN_TRADES_FOR_COIN = 30
DISABLE_SCORE = -0.002
# ============================================

exchange = ccxt.binance()

# ---------- HELPERS ----------
def coin_score(trades):
    if len(trades) < MIN_TRADES_FOR_COIN:
        return 0.0
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    win_rate = len(wins) / len(trades)
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0
    return (win_rate * avg_win - (1 - win_rate) * avg_loss) * len(trades)


def drawdown_multiplier(dd):
    for level, mult in DD_LEVELS:
        if dd <= level:
            return mult
    return 0.0


# ---------- FETCH DATA ----------
def fetch_history(symbol, tf="1m", limit=3000):
    bars = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
    return pd.DataFrame(
        bars, columns=["time", "open", "high", "low", "close", "volume"]
    )


# ---------- INDICATORS ----------
def apply_strategy(df):
    df["ema_fast"] = ta.trend.EMAIndicator(df["close"], 9).ema_indicator()
    df["ema_slow"] = ta.trend.EMAIndicator(df["close"], 21).ema_indicator()
    df["ema_200"] = ta.trend.EMAIndicator(df["close"], 200).ema_indicator()
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], 14).rsi()

    adx = ta.trend.ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=14
    )
    df["adx"] = adx.adx()

    atr = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    )
    df["atr"] = atr.average_true_range()

    return df


# ---------- 5m TREND ----------
def get_5m_trend(df_5m, t):
    window = df_5m[df_5m["time"] <= t].tail(60)
    if len(window) < 50:
        return None

    ema_fast = ta.trend.EMAIndicator(window["close"], 21).ema_indicator().iloc[-1]
    ema_slow = ta.trend.EMAIndicator(window["close"], 50).ema_indicator().iloc[-1]

    if ema_fast > ema_slow:
        return "BULL"
    if ema_fast < ema_slow:
        return "BEAR"
    return "FLAT"


# ================= PORTFOLIO BACKTEST =================
portfolio_balance = START_BALANCE
portfolio_peak = START_BALANCE
portfolio_trades = []

print("\n📊 Day 4.8 – Portfolio Capital & Risk Engine\n")

for SYMBOL in SYMBOLS:
    COIN = SYMBOL.replace("/USDT", "")
    print(f"\n🪙 Backtesting {COIN}...")

    df_1m = fetch_history(SYMBOL, "1m")
    df_5m = fetch_history(SYMBOL, "5m")

    df_1m = apply_strategy(df_1m)
    df_1m = prepare_features(df_1m)

    position = None
    entry_price = 0.0
    best_price = 0.0
    hold_candles = 0
    coin_trades = []
    coin_capital = 0.0

    for i in range(MIN_CANDLES, len(df_1m)):
        row = df_1m.iloc[i]
        price = row["close"]
        ts = row["time"]

        window_df = df_1m.iloc[i - AI_WINDOW : i + 1]
        ai_prob = predict_probability(window_df)
        ai_prev = predict_probability(df_1m.iloc[i - AI_WINDOW - 5 : i - 4])
        ai_slope = ai_prob - ai_prev

        trend_5m = get_5m_trend(df_5m, ts)
        if trend_5m is None:
            continue

        # ---- Portfolio drawdown ----
        portfolio_peak = max(portfolio_peak, portfolio_balance)
        drawdown = (portfolio_peak - portfolio_balance) / portfolio_peak
        risk_mult = drawdown_multiplier(drawdown)

        if risk_mult == 0.0:
            continue  # trading paused

        # ---- Trade sizing ----
        risk_amount = portfolio_balance * BASE_RISK_PCT * risk_mult
        coin_vol = row["atr"] / price if row["atr"] > 0 else TARGET_VOL
        trade_size = risk_amount * (TARGET_VOL / max(coin_vol, 0.002))
        trade_size *= (0.7 + ai_prob)

        if coin_capital + trade_size > portfolio_balance * MAX_COIN_EXPOSURE:
            continue

        # -------- ENTRY --------
        if position is None:
            if ai_prob >= 0.52 and ai_slope > -0.01 and trend_5m in ["BULL", "FLAT"]:
                position = "LONG"
                entry_price = price
                best_price = price
                hold_candles = 0
                coin_capital += trade_size

            elif ai_prob <= 0.48 and ai_slope < 0.01 and trend_5m in ["BEAR", "FLAT"]:
                position = "SHORT"
                entry_price = price
                best_price = price
                hold_candles = 0
                coin_capital += trade_size

        # -------- EXIT --------
        else:
            hold_candles += 1

            if position == "LONG":
                best_price = max(best_price, price)
                pnl = (price - entry_price) / entry_price
                trail_hit = price <= best_price * (1 - TRAILING_STOP)
            else:
                best_price = min(best_price, price)
                pnl = (entry_price - price) / entry_price
                trail_hit = price >= best_price * (1 + TRAILING_STOP)

            exit_now = (
                pnl <= -STOP_LOSS
                or pnl >= TAKE_PROFIT
                or (trail_hit and pnl > 0)
                or (hold_candles >= MAX_HOLD_CANDLES and pnl < 0.005)
                or ai_prob < 0.38
            )

            if exit_now:
                profit = pnl * trade_size
                portfolio_balance += profit
                portfolio_trades.append(profit)
                coin_trades.append(profit)
                position = None
                coin_capital -= trade_size

    score = coin_score(coin_trades)
    if score < DISABLE_SCORE:
        print(f"🚫 {COIN} disabled (score={score:.4f})")
        continue

    print(f"Trades: {len(coin_trades)} | PnL: {sum(coin_trades):.2f}")

# ================= RESULT =================
wins = [t for t in portfolio_trades if t > 0]
losses = [t for t in portfolio_trades if t <= 0]

win_rate = len(wins) / len(portfolio_trades)
avg_win = np.mean(wins)
avg_loss = abs(np.mean(losses))
expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

print("\n========== PORTFOLIO RESULT ==========")
print(f"Total Trades: {len(portfolio_trades)}")
print(f"Win Rate: {win_rate*100:.2f}%")
print(f"Avg Win: {avg_win:.2f}")
print(f"Avg Loss: {avg_loss:.2f}")
print(f"Expectancy / trade: {expectancy:.4f}")
print(f"Final Balance: {portfolio_balance:.2f}")
print("=====================================")
