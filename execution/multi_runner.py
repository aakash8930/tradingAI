# execution/multi_runner.py

import time
import json
from pathlib import Path
from execution.coin_selector import CoinSelector
from config.live import LiveSettings
from execution.runner import TradingRunner


class MultiSymbolTradingSystem:
    def __init__(self, settings: LiveSettings):
        self.settings = settings
        self.runners: list[TradingRunner] = []
        self.error_counts: dict[str, int] = {}
        self.disabled_symbols: set[str] = set()
        
        selector = CoinSelector(
            timeframe=settings.timeframe,
            top_k=settings.max_active_positions + 2,
        )

        selected_symbols = selector.select(settings.symbols)

        print(f"🔁 Selected symbols: {selected_symbols}")

        for symbol in selected_symbols:
            if self.settings.require_model_quality and not self._model_quality_ok(symbol):
                print(f"[{symbol}] skipped due to weak validation metrics")
                continue

            runner = TradingRunner(
                symbol=symbol,
                timeframe=settings.timeframe,
                lookback=settings.lookback,
                mode=settings.mode,
                exchange_name=settings.exchange_name,
                api_key=settings.api_key,
                api_secret=settings.api_secret,
                testnet=settings.testnet,
                allow_live_trading=settings.allow_live_trading,
                starting_balance_usdt=settings.starting_balance_usdt,
                min_balance_usdt=settings.min_balance_usdt,
                cooldown_minutes=settings.cooldown_minutes,
                risk_per_trade=settings.risk_per_trade,
                max_position_notional_pct=settings.max_position_notional_pct,
                long_prob_threshold=settings.long_prob_threshold,
                short_prob_threshold=settings.short_prob_threshold,
                min_adx=settings.min_adx,
                take_profit_rr=settings.take_profit_rr,
                breakeven_rr=settings.breakeven_rr,
                balance_allocation_pct=settings.capital_allocation_per_symbol_pct,
                data_retry_attempts=settings.data_retry_attempts,
                on_before_entry=self._can_open_new_position,
            )
            self.runners.append(runner)
            self.error_counts[symbol] = 0

        if not self.runners:
            raise RuntimeError("No symbols passed model-quality checks. Retrain models or relax thresholds.")

    def _model_quality_ok(self, symbol: str) -> bool:
        metadata_path = Path("models") / symbol.replace("/", "_") / "metadata.json"
        if not metadata_path.exists():
            return False

        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            metrics = data.get("metrics", {})
        except Exception:
            return False

        val_f1 = float(metrics.get("val_f1", 0.0))
        val_precision = float(metrics.get("val_precision", 0.0))
        val_recall = float(metrics.get("val_recall", 0.0))

        if val_f1 < self.settings.min_model_val_f1:
            return False
        if val_precision < self.settings.min_model_val_precision:
            return False
        if val_recall < self.settings.min_model_val_recall:
            return False
        return True

    def _active_positions_count(self) -> int:
        count = 0
        for runner in self.runners:
            if runner.broker.position is not None:
                count += 1
        return count

    def _can_open_new_position(self, symbol: str) -> bool:
        if self._active_positions_count() >= self.settings.max_active_positions:
            return False
        return True

    def run_loop(self) -> None:
        print(f"🚀 Multi-symbol trading started for: {', '.join(self.settings.symbols)}")

        while True:
            try:
                scores = {}

                # --- Rank symbols ---
                for runner in self.runners:
                    if runner.symbol in self.disabled_symbols:
                        continue
                    try:
                        df = runner._fetch_ohlcv_with_retry(limit=runner.lookback)
                        df = runner.strategy.compute_indicators(df)

                        score = runner.strategy.score_symbol(df)
                        scores[runner.symbol] = score

                    except Exception as e:
                        print(f"[{runner.symbol}] ranking skipped due to data error: {e}")
                        continue
                    
                # Trade only TOP 2 symbols
                top_symbols = sorted(
                    scores, key=scores.get, reverse=True
                )[:2]

                for runner in self.runners:
                    if runner.symbol in self.disabled_symbols:
                        continue

                    if runner.symbol not in top_symbols:
                        continue

                    try:
                        runner.run_once()
                        self.error_counts[runner.symbol] = 0

                    except Exception as symbol_error:
                        self.error_counts[runner.symbol] += 1
                        print(f"[{runner.symbol}] cycle error: {symbol_error}")

                        if self.error_counts[runner.symbol] >= self.settings.max_consecutive_errors:
                            self.disabled_symbols.add(runner.symbol)
                            print(
                                f"[{runner.symbol}] disabled after "
                                f"{self.error_counts[runner.symbol]} consecutive errors"
                            )

                time.sleep(self.settings.sleep_seconds)

            except KeyboardInterrupt:
                print("Stopped by user / kill-switch")
                break

