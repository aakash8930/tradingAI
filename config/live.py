# config/live.py

import os
from dataclasses import dataclass, field


LIVE_UNLOCK_TOKEN = "YES_I_UNDERSTAND"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


@dataclass
class LiveSettings:
    mode: str = "paper"  # paper | spot | futures
    symbol: str = "BTC/USDT"
    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT"])
    timeframe: str = "15m"
    exchange_name: str = "binance"

    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True
    allow_live_trading: bool = False

    lookback: int = 300
    sleep_seconds: int = 900

    starting_balance_usdt: float = 500.0
    min_balance_usdt: float = 100.0
    cooldown_minutes: int = 30
    risk_per_trade: float = 0.01
    max_position_notional_pct: float = 0.20
    long_prob_threshold: float = 0.52
    short_prob_threshold: float = 0.48
    min_adx: float = 8.0
    take_profit_rr: float = 2.0
    breakeven_rr: float = 1.0
    max_active_positions: int = 2
    capital_allocation_per_symbol_pct: float = 0.25
    data_retry_attempts: int = 3
    max_consecutive_errors: int = 5
    require_model_quality: bool = True
    min_model_val_f1: float = 0.10
    min_model_val_precision: float = 0.10
    min_model_val_recall: float = 0.10

    @staticmethod
    def _parse_symbols(single_symbol: str) -> list[str]:
        raw_multi = os.getenv("TRADING_SYMBOLS", "")
        if raw_multi.strip():
            symbols = [item.strip().upper() for item in raw_multi.split(",") if item.strip()]
        else:
            symbols = [single_symbol.strip().upper()]

        unique_symbols: list[str] = []
        for symbol in symbols:
            if symbol not in unique_symbols:
                unique_symbols.append(symbol)
        return unique_symbols

    @classmethod
    def from_env(cls) -> "LiveSettings":
        mode = os.getenv("TRADING_MODE", "paper").strip().lower()
        symbol = os.getenv("TRADING_SYMBOL", "BTC/USDT").strip().upper()
        symbols = cls._parse_symbols(symbol)
        return cls(
            mode=mode,
            symbol=symbol,
            symbols=symbols,
            timeframe=os.getenv("TRADING_TIMEFRAME", "15m").strip(),
            exchange_name=os.getenv("EXCHANGE_NAME", "binance").strip().lower(),
            api_key=os.getenv("EXCHANGE_API_KEY", "").strip(),
            api_secret=os.getenv("EXCHANGE_API_SECRET", "").strip(),
            testnet=_env_bool("EXCHANGE_TESTNET", True),
            allow_live_trading=os.getenv("ALLOW_LIVE_TRADING", "").strip() == LIVE_UNLOCK_TOKEN,
            lookback=_env_int("LOOKBACK_BARS", 300),
            sleep_seconds=_env_int("LOOP_SLEEP_SECONDS", 900),
            starting_balance_usdt=_env_float("PAPER_STARTING_BALANCE_USDT", 500.0),
            min_balance_usdt=_env_float("MIN_BALANCE_USDT", 100.0),
            cooldown_minutes=_env_int("ENTRY_COOLDOWN_MINUTES", 30),
            risk_per_trade=_env_float("RISK_PER_TRADE", 0.01),
            max_position_notional_pct=_env_float("MAX_POSITION_NOTIONAL_PCT", 0.20),
            long_prob_threshold=_env_float("LONG_PROB_THRESHOLD", 0.52),
            short_prob_threshold=_env_float("SHORT_PROB_THRESHOLD", 0.48),
            min_adx=_env_float("MIN_ADX", 8.0),
            take_profit_rr=_env_float("TAKE_PROFIT_RR", 2.0),
            breakeven_rr=_env_float("BREAKEVEN_RR", 1.0),
            max_active_positions=_env_int("MAX_ACTIVE_POSITIONS", 2),
            capital_allocation_per_symbol_pct=_env_float("CAPITAL_ALLOCATION_PER_SYMBOL_PCT", 0.25),
            data_retry_attempts=_env_int("DATA_RETRY_ATTEMPTS", 3),
            max_consecutive_errors=_env_int("MAX_CONSECUTIVE_ERRORS", 5),
            require_model_quality=_env_bool("REQUIRE_MODEL_QUALITY", True),
            min_model_val_f1=_env_float("MIN_MODEL_VAL_F1", 0.10),
            min_model_val_precision=_env_float("MIN_MODEL_VAL_PRECISION", 0.10),
            min_model_val_recall=_env_float("MIN_MODEL_VAL_RECALL", 0.10),
        )
        
        


    def validate(self) -> None:
        if self.mode not in {"paper", "live", "futures"}:
            raise ValueError("TRADING_MODE must be 'paper', 'live', or 'futures'.")

        if not self.symbols:
            raise ValueError("No trading symbols configured. Use TRADING_SYMBOL or TRADING_SYMBOLS.")

        self.symbol = self.symbols[0]

        if self.lookback < 220:
            raise ValueError("LOOKBACK_BARS must be >= 220 for EMA200/ATR safety checks.")

        if not (0 < self.risk_per_trade <= 0.05):
            raise ValueError("RISK_PER_TRADE must be in (0, 0.05].")

        if not (0 < self.max_position_notional_pct <= 1.0):
            raise ValueError("MAX_POSITION_NOTIONAL_PCT must be in (0, 1].")

        if not (0.5 <= self.long_prob_threshold <= 0.9):
            raise ValueError("LONG_PROB_THRESHOLD must be in [0.5, 0.9].")

        if not (0.1 <= self.short_prob_threshold <= 0.5):
            raise ValueError("SHORT_PROB_THRESHOLD must be in [0.1, 0.5].")

        if self.short_prob_threshold >= self.long_prob_threshold:
            raise ValueError("SHORT_PROB_THRESHOLD must be < LONG_PROB_THRESHOLD.")

        if self.min_adx < 0:
            raise ValueError("MIN_ADX must be >= 0.")

        if self.take_profit_rr <= 0:
            raise ValueError("TAKE_PROFIT_RR must be > 0.")

        if self.breakeven_rr <= 0:
            raise ValueError("BREAKEVEN_RR must be > 0.")

        if self.max_active_positions < 1:
            raise ValueError("MAX_ACTIVE_POSITIONS must be >= 1.")

        if self.max_active_positions > len(self.symbols):
            self.max_active_positions = len(self.symbols)

        if not (0 < self.capital_allocation_per_symbol_pct <= 1.0):
            raise ValueError("CAPITAL_ALLOCATION_PER_SYMBOL_PCT must be in (0, 1].")

        if self.max_active_positions * self.capital_allocation_per_symbol_pct > 1.0:
            self.capital_allocation_per_symbol_pct = 1.0 / self.max_active_positions

        if self.data_retry_attempts < 1:
            raise ValueError("DATA_RETRY_ATTEMPTS must be >= 1")

        if self.max_consecutive_errors < 1:
            raise ValueError("MAX_CONSECUTIVE_ERRORS must be >= 1")

        if not (0.0 <= self.min_model_val_f1 <= 1.0):
            raise ValueError("MIN_MODEL_VAL_F1 must be in [0, 1].")

        if not (0.0 <= self.min_model_val_precision <= 1.0):
            raise ValueError("MIN_MODEL_VAL_PRECISION must be in [0, 1].")

        if not (0.0 <= self.min_model_val_recall <= 1.0):
            raise ValueError("MIN_MODEL_VAL_RECALL must be in [0, 1].")

        for symbol in self.symbols:
            if "/" not in symbol:
                raise ValueError(f"Invalid symbol format: {symbol}. Use format like BTC/USDT")

        if self.mode == "live":
            if not self.allow_live_trading:
                raise ValueError(
                    "Live trading is locked. Set ALLOW_LIVE_TRADING=YES_I_UNDERSTAND to unlock."
                )
            if not self.api_key or not self.api_secret:
                raise ValueError("EXCHANGE_API_KEY and EXCHANGE_API_SECRET are required for live mode.")
            
        if self.mode == "futures":
            if self.risk_per_trade > 0.005:
                raise ValueError(
                    "Futures mode blocked: RISK_PER_TRADE must be <= 0.5%"
                )

